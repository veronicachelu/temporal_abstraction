import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from collections import deque
from agents.schedules import LinearSchedule, TFLinearSchedule
from PIL import Image
import scipy.stats
FLAGS = tf.app.flags.FLAGS


class ACMatrixAgent():
  def __init__(self, game, global_step, config):
    self.name = "worker_0"
    self.thread_id = 0
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.model_path = os.path.join(config.stage_logdir, "models")
    self.summary_path = os.path.join(config.stage_logdir, "summaries")
    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)
    self.increment_global_step = self.global_step.assign_add(1)
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.episode_mean_q_values = []
    self.episode_mean_returns = []
    self.episode_mean_oterms = []
    self.episode_mean_options = []
    self.episode_options = []
    self.sf_transition_matrix = deque()
    self.config = config
    self.total_steps_tensor = tf.Variable(0, dtype=tf.int32, name='total_steps_tensor', trainable=False)
    self.increment_total_steps_tensor = self.total_steps_tensor.assign_add(1)
    self.total_steps = 0
    self.action_size = game.action_space.n

    self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
    self.summary = tf.Summary()

    self.local_network = config.network(self.name, config, self.action_size, 3)

    self.update_local_vars = update_target_graph('global', self.name)
    self.env = game

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if episode_count > self.config.steps:
          return 0

        sess.run(self.update_local_vars)
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0
        old_sf = None

        s = self.env.reset()

        while not d:
          feed_dict = {self.local_network.observation: np.stack([s])}
          pi, sf, fi = sess.run([self.local_network.policy, self.local_network.sf, self.local_network.fi],
                                                     feed_dict=feed_dict)
          a = np.random.choice(pi[0], p=pi[0])
          a = np.argmax(pi == a)
          sf = sf[0, :, a]

          if old_sf is not None:
            self.build_sf_matrix(old_sf, sf)

          s1, r, d, _ = self.env.step(a)

          r = np.clip(r, -1, 1)
          self.total_steps += 1
          sess.run(self.increment_total_steps_tensor)
          episode_reward += r
          t += 1
          t_counter += 1
          s = s1
          old_sf = sf

          if self.name == "worker_0":
            print("Episode {} >>> Step {} >>> Length: {} >>> Reward: {}".format(episode_count, self.total_steps, t, episode_reward))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(t)

        # if episode_count % self.config.checkpoint_interval == 0 and self.name == 'worker_0' and \
        #         self.total_steps != 0:
        #   np.save(os.path.join(self.model_path, "sf_transition_matrix.npy"), np.asarray(self.sf_transition_matrix))

        if self.name == 'worker_0':
          sess.run(self.increment_global_step)
        episode_count += 1

  def build_sf_matrix(self, sf_old, sf_new):
    if len(self.sf_transition_matrix) == self.config.sf_transition_matrix_size:
      print("Matrix is full")
      np.save(os.path.join(self.model_path, "sf_transition_matrix.npy"), np.asarray(self.sf_transition_matrix))
      self.sf_transition_matrix.popleft()

    self.sf_transition_matrix.append(sf_new - sf_old)

