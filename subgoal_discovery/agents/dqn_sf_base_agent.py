import tensorflow as tf
import numpy as np
from collections import deque
import os
import pickle
from .base_vis_agent import BaseVisAgent
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import matplotlib.patches as patches
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm
from collections import deque
from PIL import Image
import scipy.stats
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from auxilary.policy_iteration import PolicyIteration
FLAGS = tf.app.flags.FLAGS


class DQNSFBaseAgent(BaseVisAgent):
  def __init__(self, game, _, global_step, config, type_of_task):
    self.config = config
    self.global_step = global_step
    self.optimizer = config.network_optimizer
    self.increment_global_step = self.global_step.assign_add(1)
    self.increment_batch_global_step = self.global_step.assign_add(self.config.batch_size)
    self.model_path = os.path.join(config.stage_logdir, "models")
    self.summary_path = os.path.join(config.stage_logdir, "summaries")
    self.buffer_path = os.path.join(self.model_path, "buffer")

    if type_of_task == "sf":
      if os.path.exists(self.buffer_path):
        self.load_buffer()
        # self.buf_counter = self.episode_buffer['counter']
      else:
        tf.gfile.MakeDirs(self.buffer_path)
        self.episode_buffer = {
          'counter': 0,
          'observations': np.zeros(
            (self.config.observation_steps, config.input_size[0], config.input_size[1], config.history_size)),
          # 'fi': np.zeros((self.config.observation_steps, self.config.sf_layers[-1])),
          'next_observations': np.zeros(
            (self.config.observation_steps, config.input_size[0], config.input_size[1], config.history_size)),
          'actions': np.zeros(
            (self.config.observation_steps,)),
          'done': np.zeros(
            (self.config.observation_steps,)),
        }
      # self.buf_counter = 0

    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)

    self.action_size = game.action_space.n
    self.actions = np.zeros([self.action_size])
    self.nb_states = game.nb_states
    self.summary_writer = tf.summary.FileWriter(self.summary_path)
    self.summary = tf.Summary()
    self.env = game

  def update_target_graph_tao(self, from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
      op_holder.append(to_var.assign((1 - FLAGS.TAO) * to_var.value() + FLAGS.TAO * from_var.value()))
    return op_holder

  def update_target_graph(self, from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
      op_holder.append(to_var.assign(from_var))
    return op_holder

  def save_model(self, sess, saver, episode_count):
    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
               global_step=self.global_step)

    print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

  def save_buffer(self):
    np.save(os.path.join(self.buffer_path, "observations.npy"), self.episode_buffer["observations"])
    # np.save(os.path.join(self.buffer_path, "fi.npy"), self.episode_buffer["fi"])
    np.save(os.path.join(self.buffer_path, "next_observations.npy"), self.episode_buffer["next_observations"])
    np.save(os.path.join(self.buffer_path, "actions.npy"), self.episode_buffer["actions"])
    np.save(os.path.join(self.buffer_path, "done.npy"), self.episode_buffer["done"])
    np.save(os.path.join(self.buffer_path, "buff_counter.npy"), self.episode_buffer["counter"])

  def load_buffer(self):
    self.episode_buffer = {
      'counter': np.load(os.path.join(self.buffer_path, "buff_counter.npy")),
      'observations': np.load(os.path.join(self.buffer_path, "observations.npy")),
      # 'fi': np.load(os.path.join(self.buffer_path, "fi.npy")),
      'next_observations': np.load(os.path.join(self.buffer_path, "next_observations.npy")),
      'actions': np.load(os.path.join(self.buffer_path, "actions.npy")),
      'done': np.load(os.path.join(self.buffer_path, "done.npy")),
    }
    # with open(self.buffer_path, "rb") as fp:
    #   self.episode_buffer = pickle.load(fp)

  def build_matrix(self, sess, coord, saver):
    matrix_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "matrix.npy")
    matrix_fi_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "matrix_fi.npy")
    if os.path.exists(matrix_path) and os.path.exists(matrix_fi_path):
      self.matrix_sf = np.load(matrix_path)
      self.matrix_fi = np.load(matrix_fi_path)
    else:
      with sess.as_default(), sess.graph.as_default():
        self.matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
        self.matrix_fi = np.zeros((self.nb_states, self.config.sf_layers[-1]))
        for idx in range(self.nb_states):
          s, ii, jj = self.env.get_state(idx)
          if self.env.not_wall(ii, jj):
            feed_dict = {self.orig_net.observation: [s]}
            fi, sf = sess.run([self.orig_net.fi, self.orig_net.sf], feed_dict=feed_dict)
            self.matrix_fi[idx], self.matrix_sf[idx] = fi[0], sf[0]

            # plt.pcolor(self.matrix_sf, cmap='hot', interpolation='nearest')
            # plt.savefig(os.path.join(self.summary_path, 'SR_matrix.png'))
        # self.reconstruct_sr(self.matrix_sf)
        np.save(matrix_path, self.matrix_sf)
        np.save(matrix_fi_path, self.matrix_fi)

        # self.plot_sr_vectors(self.matrix_sf)
        # self.plot_sr_matrix(self.matrix_sf)
        # self.eigen_decomp(self.matrix_sf)
    import seaborn as sns
    sns.plt.clf()
    ax = sns.heatmap(self.matrix_sf, cmap="Blues")
    ax.set(xlabel='SR_vect_size=128', ylabel='Grid states/positions')
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "eigenoptions")
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'SR_matrix.png'))
    sns.plt.close()
    np.savetxt(os.path.join(folder_path, 'Matrix_SF_numeric.txt'), self.matrix_sf, fmt='%-7.2f')

    sns.plt.clf()
    ax = sns.heatmap(self.matrix_fi, cmap="Blues")
    ax.set(xlabel='FI_vect_size=128', ylabel='Grid states/positions')
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "eigenoptions")
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'FI_matrix.png'))
    sns.plt.close()
    np.savetxt(os.path.join(folder_path, 'Matrix_FI_numeric.txt'), self.matrix_fi, fmt='%-7.2f')

    self.plot_eigenoptions("eigenoptions", sess)
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "policies")
    tf.gfile.MakeDirs(folder_path)
    self.plot_policy_and_value_function_approx(folder_path, sess)


  def build_matrix_approx(self, sess, coord, saver):
    matrix_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "matrix.npy")
    if os.path.exists(matrix_path):
      self.matrix_sf = np.load(matrix_path)
    else:
      with sess.as_default(), sess.graph.as_default():
        self.matrix_sf = np.zeros((self.config.sf_transition_matrix_size, self.config.sf_layers[-1]))
        mat_counter = 0
        t = 0
        while t < self.config.sf_transition_matrix_size:
          d = False

          s = self.env.reset()

          while not d:
            if t >= self.config.sf_transition_matrix_size:
              break
            a = np.random.choice(range(self.action_size))

            feed_dict = {self.orig_net.observation: np.stack([s])}
            sf = sess.run(self.orig_net.sf,
                          feed_dict=feed_dict)[0]
            self.matrix_sf[mat_counter] = sf
            mat_counter += 1
            s1, r, d, _ = self.env.step(a)
            t += 1
            s = s1
      np.save(matrix_path, self.matrix_sf)

    self.plot_eigenoptions("eigenoptions", sess)
      # self.plot_sr_vectors(self.matrix_sf, "sr_vectors")
      # self.plot_sr_matrix(self.matrix_sf, "sr_matrix")
      # self.eigen_decomp(self.matrix_sf)

  def plot_eigenoptions(self, folder, sess):
    # feed_dict = {self.orig_net.matrix_sf: self.matrix_sf}
    # s, v = sess.run([self.orig_net.s, self.orig_net.v], feed_dict=feed_dict)
    u, s, v = np.linalg.svd(self.matrix_sf, full_matrices=False)
    eigenvalues = s
    eigenvectors = v
    # U, s, V = np.linalg.svd(matrix)
    S = np.diag(s[1:])
    sr_r_m = np.dot(u[:, 1:], np.dot(S, v[1:]))
    import seaborn as sns
    sns.plt.clf()
    ax = sns.heatmap(sr_r_m, cmap="Blues")
    ax.set(xlabel='SR_vect_size=128', ylabel='Grid states/positions')
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "eigenoptions")
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'reconstructed_sr.png'))
    sns.plt.close()

    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), folder)
    tf.gfile.MakeDirs(folder_path)

    # variance_eigenvectors = []
    # for i in range(self.nb_states):
    #   variance_eigenvectors.append([])
    # for i in range(self.nb_states):
    #   variance_eigenvectors[i].append(np.var(eigenvectors[:, i]))
    #   sns.plt.clf()
    #   sns.plt.plot(variance_eigenvectors[i])
    #   sns.plt.savefig(os.path.join(folder_path, 'var_eig_' + str(i) + '.png'))
    #   sns.plt.close()


    sns.plt.clf()
    ax = sns.heatmap(eigenvectors, cmap="Blues")
    ax.set(xlabel='Eigenvector_dim=128', ylabel='Eigenvectors')

    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), folder)
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'Eigenvectors.png'))
    sns.plt.close()

    sns.plt.clf()
    sns.plt.plot(eigenvalues, 'o')
    sns.plt.savefig(os.path.join(folder_path, 'Eignevalues.png'))
    sns.plt.close()

    eigenvectors_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvectors.npy")
    eigenvalues_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvalues.npy")
    np.save(eigenvectors_path, eigenvectors)
    np.save(eigenvalues_path, eigenvalues)

  def reconstruct_sr(self, matrix):
    U, s, V = np.linalg.svd(matrix)
    S = np.diag(s[2:60])
    sr_r_m = np.dot(U[:, 2:60], np.dot(S, V[2:60]))
    self.plot_sr_vectors(sr_r_m, "reconstructed_sr")
    self.plot_sr_matrix(sr_r_m, "reconstructed_sr")

  def plot_sr_matrix(self, matrix, folder):
    plt.clf()
    ax = sns.heatmap(matrix, cmap="Blues")

    # for s in range(self.nb_states):
    #   ii, jj = self.env.get_state_xy(s)
    #   if self.env.not_wall(ii, jj):
    #     continue
    #   else:
    #     plt.gca().add_patch(
    #       patches.Rectangle(
    #         (jj, self.config.input_size[0] - ii - 1),  # (x,y)
    #         1.0,  # width
    #         1.0,  # height
    #         facecolor="gray"
    #       )
    #     )
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), folder)
    tf.gfile.MakeDirs(folder_path)
    plt.savefig(os.path.join(folder_path, 'SR_matrix.png'))
    plt.close()


  def plot_vector(self, vector, i):
    plt.clf()
    Z = vector.reshape(self.config.input_size[0], self.config.input_size[1])
    ax = sns.heatmap(Z, cmap="Blues")

    for idx in range(self.nb_states):
      ii, jj = self.env.get_state_xy(idx)
      if self.env.not_wall(ii, jj):
        continue
      else:
        plt.gca().add_patch(
          patches.Rectangle(
            (jj, self.config.input_size[0] - ii - 1),  # (x,y)
            1.0,  # width
            1.0,  # height
            facecolor="gray"
          )
        )
    plt.savefig(os.path.join(self.summary_path, ("sr_vectors/Return_VECTOR_" + str(i) + '.png')))
    plt.close()


  def plot_sr_vectors(self, matrix, folder):
    plt.clf()
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
            plt.gca().add_patch(
              patches.Rectangle(
                (jj, self.config.input_size[0] - ii - 1),  # (x,y)
                1.0,  # width
                1.0,  # height
                facecolor="gray"
              )
            )
        plt.savefig(os.path.join(folder_path, "SR_VECTOR_" + str(i) + '.png'))
        plt.close()


  def eigen_decomp(self, matrix):
    u, s, v = np.linalg.svd(matrix)
    noise_reduction = s > 1
    # s = s[noise_reduction]
    # v = v[noise_reduction]
    # self.plot_basis_functions(s, v)
    self.plot_policy_and_value_function(s, v)


  def plot_basis_functions(self, eigenvalues, eigenvectors):
    plt.clf()
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
            plt.gca().add_patch(
              patches.Rectangle(
                (jj, self.config.input_size[0] - ii - 1),  # (x,y)
                1.0,  # width
                1.0,  # height
                facecolor="gray"
              )
            )
        plt.savefig(os.path.join(self.summary_path, ("Eigenvector" + str(i) + '_eig_' + k + '.png')))
        plt.close()

    plt.plot(eigenvalues, 'o')
    plt.savefig(self.summary_path + 'eigenvalues.png')


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

  def plot_policy_and_value_function_approx(self, folder, sess):
    # feed_dict = {self.orig_net.matrix_sf: self.matrix_sf}
    # s, v = sess.run([self.orig_net.s, self.orig_net.v], feed_dict=feed_dict)
    u, s, v = np.linalg.svd(self.matrix_sf)
    eigenvalues = s
    eigenvectors = v

    epsilon = 0.0001
    options = []
    with sess.as_default(), sess.graph.as_default():
      self.env.define_network(self.orig_net)
      self.env.define_session(sess)
      for k in ["poz", "neg"]:
        for i in range(len(eigenvalues)):
          polIter = PolicyIteration(0.9, self.env, augmentActionSet=True)
          self.env.define_reward_function(eigenvectors[i] if k == "poz" else -eigenvectors[i])
          V, pi = polIter.solvePolicyIteration()

          for j in range(len(V)):
            if V[j] < epsilon:
              pi[j] = len(self.env.get_action_set())

          self.plot_value_function(V[0:self.nb_states], str(i) + '_' + k + "_", folder)
          self.plot_policy(pi[0:self.nb_states], str(i) + '_' + k + "_", folder)

          options.append(pi[0:self.nb_states])
          # optionsActionSet = self.env.get_action_set()
          # np.append(optionsActionSet, ['terminate'])
          # actionSetPerOption.append(optionsActionSet)


  def plot_value_function(self, value_function, prefix, folder=None):
    if folder is None:
      folder = self.summary_path
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
    plt.savefig(os.path.join(folder, "SuccessorFeatures" + prefix + 'value_function.png'))
    plt.close()

  def plot_options(self, sess, coord, saver):
    eigenvectors_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvectors.npy")
    eigenvalues_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvalues.npy")
    eigenvectors = np.load(eigenvectors_path)
    eigenvalues = np.load(eigenvalues_path)
    for k in ["poz", "neg"]:
      for option in range(len(eigenvalues)):
        # eigenvalue = eigenvalues[option]
        eigenvector = eigenvectors[option] if k == "poz" else -eigenvectors[option]
        prefix = str(option) + '_' + k + "_"

        plt.clf()

        with sess.as_default(), sess.graph.as_default():
          for idx in range(self.nb_states):
            dx = 0
            dy = 0
            d = False
            s, i, j = self.env.get_state(idx)

            if not self.env.not_wall(i, j):
              plt.gca().add_patch(
                patches.Rectangle(
                  (j, self.config.input_size[0] - i - 1),  # (x,y)
                  1.0,  # width
                  1.0,  # height
                  facecolor="gray"
                )
              )
              continue
            # Image.fromarray(np.asarray(scipy.misc.imresize(s, [512, 512], interp='nearest'), np.uint8)).show()
            # feed_dict = {self.orig_net.observation: np.stack([s])}
            # fi = sess.run(self.orig_net.fi,
            #               feed_dict=feed_dict)[0]
            a = self.option_policy_evaluation_eval(s, sess)
            if a == self.action_size:
              circle = plt.Circle(
                (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.025, color='k')
              plt.gca().add_artist(circle)
              continue

            if a == 0: # up
              dy = 0.35
            elif a == 1:  # right
              dx = 0.35
            elif a == 2:  # down
              dy = -0.35
            elif a == 3:  # left
              dx = -0.35

            plt.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, dx, dy,
                      head_width=0.05, head_length=0.05, fc='k', ec='k')

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

  # def plot_greedy_option_policy(self, sess, coord, saver):
  #   plot_options_greedy(self, sess, coord, saver):
  #   eigenvectors_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvectors.npy")
  #   eigenvalues_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvalues.npy")
  #   eigenvectors = np.load(eigenvectors_path)
  #   eigenvalues = np.load(eigenvalues_path)
  #   for k in ["poz", "neg"]:
  #     for option in range(len(eigenvalues)):
  #       # eigenvalue = eigenvalues[option]
  #       eigenvector = eigenvectors[option] if k == "poz" else -eigenvectors[option]
  #       prefix = str(option) + '_' + k + "_"
  #
  #       sns.plt.clf()
  #
  #       with sess.as_default(), sess.graph.as_default():
  #         s = self.env.reset()
  #
  def plot_options_greedy(self, sess, coord, saver):
    eigenvectors_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvectors.npy")
    eigenvalues_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvalues.npy")
    eigenvectors = np.load(eigenvectors_path)
    eigenvalues = np.load(eigenvalues_path)
    for k in ["poz", "neg"]:
      for option in range(len(eigenvalues)):
        # eigenvalue = eigenvalues[option]
        eigenvector = eigenvectors[option] if k == "poz" else -eigenvectors[option]
        prefix = str(option) + '_' + k + "_"

        sns.plt.clf()

        with sess.as_default(), sess.graph.as_default():
          for idx in range(self.nb_states):
            dx = 0
            dy = 0
            d = False
            s, i, j = self.env.get_state(idx)

            if not self.env.not_wall(i, j):
              sns.plt.gca().add_patch(
                patches.Rectangle(
                  (j, self.config.input_size[0] - i - 1),  # (x,y)
                  1.0,  # width
                  1.0,  # height
                  facecolor="gray"
                )
              )
              continue
            # Image.fromarray(np.asarray(scipy.misc.imresize(s, [512, 512], interp='nearest'), np.uint8)).show()
            feed_dict = {self.orig_net.observation: np.stack([s])}
            fi = sess.run(self.orig_net.fi,
                          feed_dict=feed_dict)[0]

            transitions = []
            terminations = []
            for a in range(self.action_size):
              s1, r, d, _ = self.env.fake_step(a)
              feed_dict = {self.orig_net.observation: np.stack([s1])}
              fi1 = sess.run(self.orig_net.fi,
                            feed_dict=feed_dict)[0]
              transitions.append(self.cosine_similarity((fi1 - fi), eigenvector))
              terminations.append(d)

            # transitions.append(self.cosine_similarity(np.zeros_like(fi), eigenvector))
            # terminations.append(True)

            a = np.argmax(transitions)
            # if a == 4:
            #   d = True

            if a == 0: # up
              dy = 0.35
            elif a == 1:  # right
              dx = 0.35
            elif a == 2:  # down
              dy = -0.35
            elif a == 3:  # left
              dx = -0.35

            if terminations[a] or transitions[a] == 0: # termination
              circle = sns.plt.Circle(
                (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.025, color='k')
              sns.plt.gca().add_artist(circle)
              continue

            sns.plt.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, dx, dy,
                      head_width=0.05, head_length=0.05, fc='k', ec='k')

          sns.plt.xlim([0, self.config.input_size[1]])
          sns.plt.ylim([0, self.config.input_size[0]])

          for i in range(self.config.input_size[1]):
            sns.plt.axvline(i, color='k', linestyle=':')
          sns.plt.axvline(self.config.input_size[1], color='k', linestyle=':')

          for j in range(self.config.input_size[0]):
            sns.plt.axhline(j, color='k', linestyle=':')
          sns.plt.axhline(self.config.input_size[0], color='k', linestyle=':')

          sns.plt.savefig(os.path.join(self.summary_path, "SuccessorFeatures_" + prefix + 'policy.png'))
          sns.plt.close()

  def cosine_similarity(self, next_sf, evect):
      state_dif_norm = np.linalg.norm(next_sf)
      state_dif_normalized = next_sf / (state_dif_norm + 1e-8)
      # evect_norm = np.linalg.norm(evect)
      # evect_normalized = evect / (evect_norm + 1e-8)

      return np.dot(state_dif_normalized, evect)

  def plot_policy(self, policy, prefix, folder=None):
    if folder is None:
      folder = self.summary_path
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

    plt.savefig(os.path.join(folder, "SuccessorFeatures_" + prefix + 'policy.png'))
    plt.close()


