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
from PIL import Image
FLAGS = tf.app.flags.FLAGS

class Visualizer():

  def build_matrix(self, sess, coord, saver):
    matrix_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "matrix.npy")
    if os.path.exists(matrix_path):
      self.matrix_sf = np.load(matrix_path)
    else:
      with sess.as_default(), sess.graph.as_default():
        self.matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
        for idx in range(self.nb_states):
          s, ii, jj = self.env.get_state(idx)
          if self.env.not_wall(ii, jj):
            feed_dict = {self.local_network.observation: [s]}
            sf = sess.run(self.local_network.sf, feed_dict=feed_dict)[0]
            self.matrix_sf[idx] = sf

            # plt.pcolor(self.matrix_sf, cmap='hot', interpolation='nearest')
            # plt.savefig(os.path.join(self.summary_path, 'SR_matrix.png'))
        # self.reconstruct_sr(self.matrix_sf)
        np.save(matrix_path, self.matrix_sf)
        # self.plot_sr_vectors(self.matrix_sf)
        # self.plot_sr_matrix(self.matrix_sf)
        # self.eigen_decomp(self.matrix_sf)
    import seaborn as sns
    sns.plt.clf()
    ax = sns.heatmap(self.matrix_sf, cmap="Blues")
    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "eigenoptions")
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'SR_matrix.png'))
    sns.plt.close()
    #
    # ax = sns.heatmap(self.matrix_sf, cmap="Blues")
    # folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "eigenoptions")
    # tf.gfile.MakeDirs(folder_path)
    # plt.savefig(os.path.join(folder_path, 'Matrix_SF.png'))
    np.savetxt(os.path.join(folder_path, 'Matrix_SF_numeric.txt'), self.matrix_sf, fmt='%-7.2f')

    # self.plot_eigenoptions("eigenoptions", sess)

  # def build_matrix(self, sess, coord, saver):
  #   with sess.as_default(), sess.graph.as_default():
  #     self.matrix_sf = np.zeros((self.nb_states, self.nb_states))
  #     for idx in range(self.nb_states):
  #       ii, jj = self.env.get_state_xy(idx)
  #       if self.env.not_wall(ii, jj):
  #         feed_dict = {self.local_network.observation: np.identity(self.nb_states)[idx:idx + 1]}
  #         sf = sess.run(self.local_network.sf, feed_dict=feed_dict)[0]
  #         self.matrix_sf[idx] = sf
  #         # plt.pcolor(self.matrix_sf, cmap='hot', interpolation='nearest')
  #         # plt.savefig(os.path.join(self.summary_path, 'SR_matrix.png'))
  #     self.reconstruct_sr(self.matrix_sf)
  #     # self.plot_sr_vectors(self.matrix_sf)
  #     # self.plot_sr_matrix(self.matrix_sf)
  #     # self.eigen_decomp(self.matrix_sf)

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

            feed_dict = {self.local_network.observation: np.stack([s])}
            sf = sess.run(self.local_network.sf,
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
    feed_dict = {self.local_network.matrix_sf: self.matrix_sf}
    s, v = sess.run([self.local_network.s, self.local_network.v], feed_dict=feed_dict)
    # u, s, v = np.linalg.svd(self.matrix_sf)
    eigenvalues = s
    eigenvectors = v

    sns.plt.clf()
    ax = sns.heatmap(eigenvectors, cmap="Blues")

    folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), folder)
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'Eigenvectors.png'))

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
            feed_dict = {self.local_network.observation: np.stack([s])}
            fi = sess.run(self.local_network.fi,
                          feed_dict=feed_dict)[0]

            transitions = []
            terminations = []
            for a in range(self.action_size):
              s1, r, d, _ = self.env.fake_step(a)
              feed_dict = {self.local_network.observation: np.stack([s1])}
              fi1 = sess.run(self.local_network.fi,
                            feed_dict=feed_dict)[0]
              transitions.append(self.cosine_similarity((fi1 - fi), eigenvector))
              terminations.append(d)

            transitions.append(self.cosine_similarity(np.zeros_like(fi), eigenvector))
            terminations.append(True)

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

            if terminations[a] or np.all(transitions[a] == np.zeros_like(fi)) : # termination
              circle = plt.Circle(
                (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.025, color='k')
              plt.gca().add_artist(circle)
              continue

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




  def cosine_similarity(self, next_sf, evect):
      state_dif_norm = np.linalg.norm(next_sf)
      state_dif_normalized = next_sf / (state_dif_norm + 1e-8)
      evect_norm = np.linalg.norm(evect)
      evect_normalized = evect / (evect_norm + 1e-8)

      return np.dot(state_dif_normalized, evect_normalized)

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
