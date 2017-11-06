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
sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from auxilary.policy_iteration import PolicyIteration
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
    tf.gfile.MakeDirs(os.path.join(os.path.join(config.stage_logdir, "summaries"), "sr_vectors"))

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
      for idx in range(self.nb_states):
        ii, jj = self.env.get_state_xy(idx)
        if self.env.not_wall(ii, jj):
          feed_dict = {self.local_network.observation: np.identity(self.nb_states)[idx:idx + 1]}
          self.matrix_sf[idx] = sess.run(self.local_network.sf, feed_dict=feed_dict)[0]
    # plt.pcolor(self.matrix_sf, cmap='hot', interpolation='nearest')
    # plt.savefig(os.path.join(self.summary_path, 'SR_matrix.png'))
      self.reconstruct_sr(self.matrix_sf)
    # self.plot_sr_vectors(self.matrix_sf)
    # self.plot_sr_matrix(self.matrix_sf)
    # self.eigen_decomp(self.matrix_sf)

  def reconstruct_sr(self, matrix):
    U, s, V = np.linalg.svd(matrix)
    S = np.diag(s[2:60])
    sr_r_m = np.dot(U[:, 2:60], np.dot(S, V[2:60]))
    self.plot_sr_vectors(sr_r_m, "reconstructed_sr")
    self.plot_sr_matrix(sr_r_m, "reconstructed_sr")

  def plot_sr_matrix(self, matrix, folder):
    sns.plt.clf()
    ax = sns.heatmap(matrix, cmap="Blues")

    # for s in range(self.nb_states):
    #   ii, jj = self.env.get_state_xy(s)
    #   if self.env.not_wall(ii, jj):
    #     continue
    #   else:
    #     sns.plt.gca().add_patch(
    #       patches.Rectangle(
    #         (jj, self.config.input_size[0] - ii - 1),  # (x,y)
    #         1.0,  # width
    #         1.0,  # height
    #         facecolor="gray"
    #       )
    #     )
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), folder)
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'SR_matrix.png'))
    sns.plt.close()

  def plot_vector(self, vector, i):
    sns.plt.clf()
    Z = vector.reshape(self.config.input_size[0], self.config.input_size[1])
    ax = sns.heatmap(Z, cmap="Blues")

    for idx in range(self.nb_states):
      ii, jj = self.env.get_state_xy(idx)
      if self.env.not_wall(ii, jj):
        continue
      else:
        sns.plt.gca().add_patch(
          patches.Rectangle(
            (jj, self.config.input_size[0] - ii - 1),  # (x,y)
            1.0,  # width
            1.0,  # height
            facecolor="gray"
          )
        )
    sns.plt.savefig(os.path.join(self.summary_path, ("sr_vectors/Return_VECTOR_" + str(i) + '.png')))
    sns.plt.close()

  def plot_sr_vectors(self, matrix, folder):
    sns.plt.clf()
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), folder)
    tf.gfile.MakeDirs(folder_path)
    for i in range(self.nb_states):
      aa, bb = self.env.get_state_xy(i)
      if self.env.not_wall(aa, bb):
        Z = matrix[i].reshape(self.config.input_size[0], self.config.input_size[1])
        ax = sns.heatmap(Z, cmap="Blues")

        for idx in range(self.nb_states):
          ii, jj = self.env.get_state_xy(idx)
          if self.env.not_wall(ii, jj):
            continue
          else:
            sns.plt.gca().add_patch(
              patches.Rectangle(
                (jj, self.config.input_size[0] - ii - 1),  # (x,y)
                1.0,  # width
                1.0,  # height
                facecolor="gray"
              )
            )
        sns.plt.savefig(os.path.join(folder_path, "SR_VECTOR_" + str(i) + '.png'))
        sns.plt.close()

  def eigen_decomp(self, matrix):
    u, s, v = np.linalg.svd(matrix)
    noise_reduction = s > 1
    # s = s[noise_reduction]
    # v = v[noise_reduction]
    # self.plot_basis_functions(s, v)
    self.plot_policy_and_value_function(s, v)

  def plot_basis_functions(self, eigenvalues, eigenvectors):
    sns.plt.clf()
    for k in ["poz", "neg"]:
      for i in range(len(eigenvalues)):
        Z = eigenvectors[i].reshape(self.config.input_size[0], self.config.input_size[1])
        if k == "neg":
          Z = -Z
        # sns.palplot(sns.dark_palette("purple", reverse=True))
        ax = sns.heatmap(Z, cmap="Blues")

        for idx in range(self.nb_states):
          ii, jj = self.env.get_state_xy(idx)
          if self.env.not_wall(ii, jj):
           continue
          else:
            sns.plt.gca().add_patch(
              patches.Rectangle(
                (jj, self.config.input_size[0] - ii - 1),  # (x,y)
                1.0,  # width
                1.0,  # height
                facecolor="gray"
              )
            )
        sns.plt.savefig(os.path.join(self.summary_path, ("Eigenvector" + str(i) + '_eig_' + k + '.png')))
        sns.plt.close()

    sns.plt.plot(eigenvalues, 'o')
    sns.plt.savefig(self.summary_path + 'eigenvalues.png')

  def plot_policy_and_value_function(self, eigenvalues, eigenvectors):
    epsilon = 0.0001
    options = []
    for k in ["poz", "neg"]:
      for i in range(len(eigenvalues)):
        polIter = PolicyIteration(0.9, self.env, augmentActionSet=True)
        self.env.define_reward_function(eigenvectors[i] if k == "poz" else -eigenvectors[i])
        V, pi = polIter.solvePolicyIteration()

        # Now I will eliminate any actions that may give us a small improvement.
        # This is where the epsilon parameter is important. If it is not set all
        # it will never be considered, since I set it to a very small value
        for j in range(len(V)):
          if V[j] < epsilon:
            pi[j] = len(self.env.get_action_set())

        # if plotGraphs:
        self.plot_value_function(V[0:self.nb_states], str(i) + '_' + k + "_")
        self.plot_policy(pi[0:self.nb_states], str(i) + '_' + k + "_")

        options.append(pi[0:self.nb_states])
        # optionsActionSet = self.env.get_action_set()
        # np.append(optionsActionSet, ['terminate'])
        # actionSetPerOption.append(optionsActionSet)

  def plot_value_function(self, value_function, prefix):
    '''3d plot of a value function.'''
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X, Y = np.meshgrid(np.arange(self.config.input_size[1]), np.arange(self.config.input_size[0]))
    Z = value_function.reshape(self.config.input_size[0], self.config.input_size[1])

    for i in range(len(X)):
        for j in range(int(len(X[i]) / 2)):
            tmp = X[i][j]
            X[i][j] = X[i][len(X[i]) - j - 1]
            X[i][len(X[i]) - j - 1] = tmp

    my_col = cm.jet(np.random.rand(Z.shape[0], Z.shape[1]))

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=plt.get_cmap('jet'))
    plt.gca().view_init(elev=30, azim=30)
    plt.savefig(os.path.join(self.summary_path, "SuccessorFeatures" + prefix + 'value_function.png'))
    plt.close()

  def plot_policy(self, policy, prefix):
    plt.clf()
    for idx in range(len(policy)):
        i, j = self.env.get_state_xy(idx)

        dx = 0
        dy = 0
        if policy[idx] == 0:  # up
            dy = 0.35
        elif policy[idx] == 1:  # right
            dx = 0.35
        elif policy[idx] == 2:  # down
            dy = -0.35
        elif policy[idx] == 3:  # left
            dx = -0.35
        elif self.env.not_wall(i, j) and policy[idx] == 4:  # termination
            circle = plt.Circle(
                (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.025, color='k')
            plt.gca().add_artist(circle)

        if self.env.not_wall(i, j):
            plt.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, dx, dy,
                      head_width=0.05, head_length=0.05, fc='k', ec='k')
        else:
            plt.gca().add_patch(
                patches.Rectangle(
                    (j, self.config.input_size[0] - i - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="gray"
                )
            )

    plt.xlim([0, self.config.input_size[1]])
    plt.ylim([0, self.config.input_size[0]])

    for i in range(self.config.input_size[1]):
        plt.axvline(i, color='k', linestyle=':')
    plt.axvline(self.config.input_size[1], color='k', linestyle=':')

    for j in range(self.config.input_size[0]):
        plt.axhline(j, color='k', linestyle=':')
    plt.axhline(self.config.input_size[0], color='k', linestyle=':')

    plt.savefig(os.path.join(self.summary_path, "SuccessorFeatures_" + prefix + 'policy.png'))
    plt.close()

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
          _, r, d, s1 = self.env.special_step(a, s)

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

