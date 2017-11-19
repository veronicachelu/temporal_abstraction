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
from collections import deque
from PIL import Image
import scipy.stats
import seaborn as sns
from .base_vis_agent import BaseVisAgent
sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from auxilary.policy_iteration import PolicyIteration
FLAGS = tf.app.flags.FLAGS


class A3CSFAgent(BaseVisAgent):
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

    self.matrix_path = os.path.join(self.model_path, "matrix.npy")
    if os.path.exists(self.matrix_path):
      self.matrix_sf = np.load(self.matrix_path)
      self.mat_counter = self.config.sf_transition_matrix_size
    else:
      self.matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
      self.mat_counter = 0

  def train(self, rollout, sess, bootstrap_sf, summaries=False):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    fi = rollout[:, 1]
    next_observations = rollout[:, 2]
    actions = rollout[:, 3]

    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.actions_placeholder: actions}

    _, ms, loss, sf_loss, aux_loss = \
      sess.run([self.local_network.apply_grads,
                self.local_network.merged_summary,
                self.local_network.loss,
                self.local_network.sf_loss,
                self.local_network.aux_loss],
               feed_dict=feed_dict)

    return ms, loss, sf_loss, aux_loss

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if self.total_steps > self.config.steps:
          return 0

        sess.run(self.update_local_vars)
        episode_buffer = []
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0

        s = self.env.reset()

        while not d:
          a = np.random.choice(range(self.action_size))

          feed_dict = {self.local_network.observation: np.stack([s])}
          sf, fi = sess.run([self.local_network.sf, self.local_network.fi],
                                feed_dict=feed_dict)
          sf, fi = sf[0], fi[0]

          if self.total_steps > self.config.training_steps:
            self.matrix_sf[self.mat_counter % self.config.sf_transition_matrix_size] = sf
            self.mat_counter += 1

            if self.mat_counter == self.sf_transition_matrix_size:
              self.plot_eigenoptions("eigenoptions", sess)
              exit(0)

          s1, r, d, _ = self.env.step(a)

          r = np.clip(r, -1, 1)
          self.total_steps += 1
          if self.total_steps < self.config.training_steps:
            episode_buffer.append([s, fi, s1, a])
          episode_reward += r
          t += 1
          t_counter += 1
          s = s1

          if self.total_steps < self.config.training_steps:
            if t_counter == self.config.max_update_freq or d:
              feed_dict = {self.local_network.observation: np.stack([s])}
              sf = sess.run(self.local_network.sf,
                                        feed_dict=feed_dict)[0]
              bootstrap_sf = np.zeros_like(sf) if d else sf
              ms, loss, sf_loss, aux_loss = self.train(episode_buffer, sess, bootstrap_sf)
              if self.name == "worker_0":
                print("Episode {} >>> Step {} >>> SF_loss {} >>> AUX_loss {} ".format(episode_count, self.total_steps, sf_loss, aux_loss))

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

          self.summary_writer.add_summary(ms, self.total_steps)

          # self.summary_writer.add_summary(img_summ, self.total_steps)

          self.summary_writer.add_summary(self.summary, self.total_steps)
          self.summary_writer.flush()

        if self.name == 'worker_0':
          sess.run(self.increment_global_step)
        episode_count += 1