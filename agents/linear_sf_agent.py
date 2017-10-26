import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from collections import deque
from agents.schedules import LinearSchedule, TFLinearSchedule
from PIL import Image
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import cm
FLAGS = tf.app.flags.FLAGS


class LinearSFAgent():
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

  def train(self, rollout, sess, bootstrap_sf, summaries=False):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    # actions = rollout[:, 1]
    # sf = rollout[:, 2]
    # fi = rollout[:, 3]
    fi = np.identity(self.nb_states)[observations]
    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.identity(self.nb_states)[observations]}

    _, ms, loss, sf_loss = \
      sess.run([self.local_network.apply_grads,
                self.local_network.merged_summary,
                self.local_network.loss,
                self.local_network.sf_loss],
               feed_dict=feed_dict)
    return ms, loss, sf_loss

  def build_matrix1(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.matrix_sf = np.zeros((self.nb_states, self.nb_states))
      for s in range(self.nb_states):
        feed_dict = {self.local_network.observation: np.identity(self.nb_states)[s:s + 1]}
        self.matrix_sf[s] = sess.run(self.local_network.sf, feed_dict=feed_dict)[0]

    self.eigen_decomp(self.matrix_sf)

  def eigen_decomp(self, matrix):
    u, s, v = np.linalg.svd(matrix)
    self.plot_basis_functions(s, v)

  def plot_basis_functions(self, eigenvalues, eigenvectors):
    for k in ["poz", "neg"]:
      for i in range(len(eigenvalues)):
        Z = eigenvectors[i].reshape(self.config.input_size[0], self.config.input_size[1])
        if k in "neg":
          Z -= Z
        X, Y = np.meshgrid(np.arange(self.config.input_size[1]), np.arange(self.config.input_size[0]))

        for ii in range(len(X)):
          for j in range(int(len(X[ii]) / 2)):
            tmp = X[ii][j]
            X[ii][j] = X[ii][len(X[ii]) - j - 1]
            X[ii][len(X[ii]) - j - 1] = tmp

        # new_Z = Z[X][Y]
        plt.pcolor(X, Y, Z, cmap=cm.Blues)
        plt.colorbar()

        # my_col = cm.jet(np.random.rand(Z.shape[0], Z.shape[1]))

        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #                 cmap=cm.Blues, linewidth=0, antialiased=False)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        # plt.gca().view_init(elev=30, azim=30)
        plt.savefig(os.path.join(self.summary_path, ("Eigenvector" + str(i) + '_eig_' + k + '.png')))
        plt.close()

    plt.plot(eigenvalues, 'o')
    plt.savefig(self.summary_path + 'eigenvalues.png')

  def build_matrix(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if episode_count > self.config.steps:
          return 0

        sess.run(self.update_local_vars)
        episode_buffer = []
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0
        R = 0
        old_sf = None

        s = self.env.get_initial_state()

        while not d:
          a = np.random.choice(range(self.action_size))
          feed_dict = {self.local_network.observation: np.identity(self.nb_states)[s:s+1]}
          sf = sess.run(self.local_network.sf, feed_dict=feed_dict)[0]
          _, r, d, s1 = self.env.step(a)

          r = np.clip(r, -1, 1)
          self.total_steps += 1
          sess.run(self.increment_total_steps_tensor)
          episode_buffer.append([s])
          episode_reward += r
          t += 1
          t_counter += 1
          s = s1

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(t)

        episode_count += 1

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if episode_count > self.config.steps:
          return 0

        sess.run(self.update_local_vars)
        episode_buffer = []
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0
        R = 0
        old_sf = None

        s = self.env.get_initial_state()

        while not d:
          a = np.random.choice(range(self.action_size))
          # feed_dict = {self.local_network.observation: np.identity(self.nb_states)[s:s+1]}
          # sf = sess.run(self.local_network.sf, feed_dict=feed_dict)
          _, r, d, s1 = self.env.step(a)

          r = np.clip(r, -1, 1)
          self.total_steps += 1
          sess.run(self.increment_total_steps_tensor)
          episode_buffer.append([s])
          episode_reward += r
          t += 1
          t_counter += 1
          s = s1

          if t_counter == self.config.max_update_freq or d:
            feed_dict = {self.local_network.observation: np.identity(self.nb_states)[s:s+1]}
            sf = sess.run(self.local_network.sf,
                                      feed_dict=feed_dict)[0]
            bootstrap_sf = np.zeros_like(sf) if d else sf
            ms, img_summ, loss = self.train(episode_buffer, sess, bootstrap_sf)
            if self.name == "worker_0":
              print("Episode {} >>> Step {} >>> SF_loss {} ".format(episode_count, self.total_steps, loss))

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

  def build_sf_matrix(self, sf):
    if len(self.sf_transition_matrix) == self.config.sf_transition_matrix_size:
      print("Matrix is ready")
      self.task = 3
      self.sf_transition_matrix.popleft()

    self.sf_transition_matrix.append(sf_new - sf_old)

