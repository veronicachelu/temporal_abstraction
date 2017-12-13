import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
import matplotlib.patches as patches
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
# import plotly.plotly as py
# import plotly.tools as tls
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib import cm
from collections import deque
from agents.schedules import LinearSchedule, TFLinearSchedule
from PIL import Image
import scipy.stats
import seaborn as sns
from auxilary.visualizer import Visualizer
sns.set()
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from auxilary.policy_iteration import PolicyIteration
FLAGS = tf.app.flags.FLAGS


class DIFAgent(Visualizer):
  def __init__(self, game, thread_id, global_step, config):
    self.name = "worker_" + str(thread_id)
    self.thread_id = thread_id
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
    self.config = config
    self.total_steps_tensor = tf.Variable(0, dtype=tf.int32, name='total_steps_tensor', trainable=False)
    self.increment_total_steps_tensor = self.total_steps_tensor.assign_add(1)
    self.total_steps = 0
    self.action_size = game.action_space.n
    self.nb_states = game.nb_states
    self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
    self.summary = tf.Summary()

    self.local_network = config.network(self.name, config, self.action_size, self.nb_states)

    self.update_local_vars = update_target_graph('global', self.name)
    self.env = game
    self.nb_states = game.nb_states

    self.matrix_path = os.path.join(self.model_path, "matrix.npy")
    if os.path.exists(self.matrix_path):
      self.matrix_sf = np.load(self.matrix_path)
      self.mat_counter = self.config.sf_transition_matrix_size
    else:
      self.matrix_sf = np.zeros((self.config.sf_transition_matrix_size, self.config.sf_layers[-1]))
      self.mat_counter = 0

  def train_sf(self, rollout, sess, bootstrap_sf, summaries=False):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0)}
    fi = sess.run(self.local_network.fi,
                          feed_dict=feed_dict)

    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0)} #,
                 # self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 # self.local_network.actions_placeholder: actions}

    _, ms, sf_loss = \
      sess.run([self.local_network.apply_grads_sf,
                self.local_network.merged_summary_sf,
                self.local_network.sf_loss],
               feed_dict=feed_dict)

    return ms, sf_loss

  def train_aux(self, sess):
    minibatch = random.sample(self.aux_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0), #,
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.actions_placeholder: actions}

    _, ms, img_summ, aux_loss = \
      sess.run([self.local_network.apply_grads_aux,
                self.local_network.merged_summary_aux,
                self.local_network.image_summaries,
                self.local_network.aux_loss],
               feed_dict=feed_dict)

    return ms, img_summ, aux_loss

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)
      ms_aux = ms_sf = None
      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if self.total_steps > self.config.steps:
          return 0

        sess.run(self.update_local_vars)
        episode_buffer = []
        self.aux_episode_buffer = deque()
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0

        s = self.env.reset()

        while not d:
          a = np.random.choice(range(self.action_size))

          # feed_dict = {self.local_network.observation: np.stack([s])}
          # sf, fi = sess.run([self.local_network.sf, self.local_network.fi],
          #                       feed_dict=feed_dict)
          # sf, fi = sf[0], fi[0]

          s1, r, d, _ = self.env.step(a)
          if d:
            s1 = s
          r = np.clip(r, -1, 1)
          self.total_steps += 1
          # if self.total_steps < self.config.training_steps:
          episode_buffer.append([s, s1, a])
          self.aux_episode_buffer.append([s, s1, a])
          episode_reward += r
          t += 1
          t_counter += 1
          s = s1

          if len(self.aux_episode_buffer) == self.config.memory_size:
            self.aux_episode_buffer.popleft()
          if self.total_steps > self.config.observation_steps and len(
              self.aux_episode_buffer) > self.config.observation_steps and \
            self.total_steps % self.config.aux_update_freq == 0:
            ms_aux, img_sum, aux_loss = self.train_aux(sess)
            if self.name == "worker_0":
              print("Episode {} >>> Step {} >>> AUX_loss {} ".format(episode_count,
                                                                     self.total_steps,
                                                                     aux_loss))

          if t_counter == self.config.max_update_freq or d:
            feed_dict = {self.local_network.observation: np.stack([s])}
            sf = sess.run(self.local_network.sf,
                                      feed_dict=feed_dict)[0]
            bootstrap_sf = np.zeros_like(sf) if d else sf
            ms_sf, sf_loss = self.train_sf(episode_buffer, sess, bootstrap_sf)

            if self.name == "worker_0":
              print("Episode {} >>> Step {} >>> SF_loss {}".format(episode_count,
                                                                   self.total_steps,
                                                                   sf_loss))

            episode_buffer = []
            t_counter = 0
          if self.name == "worker_0":
            print("Episode {} >>> Step {} >>> Length: {} >>> Reward: {}".format(episode_count, self.total_steps, t, episode_reward))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(t)

        if episode_count % self.config.checkpoint_interval == 0 and self.name == 'worker_0' and \
                self.total_steps != 0:
          saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                     global_step=self.global_step)
          print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))
          if self.mat_counter > self.config.sf_transition_matrix_size:
            np.save(self.matrix_path, self.matrix_sf)
            print("Saved Matrix at {}".format(self.matrix_path))

        if episode_count % self.config.summary_interval == 0 and self.total_steps != 0 and \
                self.name == 'worker_0':

          last_reward = self.episode_rewards[-1]
          last_length = self.episode_lengths[-1]

          self.summary.value.add(tag='Perf/Reward', simple_value=float(last_reward))
          self.summary.value.add(tag='Perf/Length', simple_value=float(last_length))

          if ms_sf is not None:
            self.summary_writer.add_summary(ms_sf, self.total_steps)
          if ms_aux is not None:
            self.summary_writer.add_summary(ms_aux, self.total_steps)

          # self.summary_writer.add_summary(img_summ, self.total_steps)

          self.summary_writer.add_summary(self.summary, self.total_steps)
          self.summary_writer.flush()

        if self.name == 'worker_0':
          sess.run(self.increment_global_step)
        episode_count += 1