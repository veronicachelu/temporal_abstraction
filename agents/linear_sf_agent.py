import numpy as np
import tensorflow as tf
from tools.agent_utils import update_target_graph, discount, make_gif
import os
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
from collections import deque
from PIL import Image
import scipy.stats
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from auxilary.policy_iteration import PolicyIteration
from online_clustering import OnlineCluster
FLAGS = tf.app.flags.FLAGS

"""This Agent corresponds to the case of performing linear approximation over states, whereas states are encoded as one-hot vectors with size corresponding to the state-space size of 169 (13x13) for the 4 Rooms Domain"""
class LinearSFAgent():
  def __init__(self, sess, game, thread_id, global_step, global_episode, config, global_network, barrier):
    self.name = "worker_" + str(thread_id)
    self.thread_id = thread_id

    self.config = config
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.global_episode = global_episode
    self.increment_global_step = self.global_step.assign_add(1)
    self.increment_global_episode = self.global_episode.assign_add(1)
    self.global_network = global_network

    """Save models in the models directory in the logdir config folder"""
    self.model_path = os.path.join(config.logdir, "models")
    """Save events file and other stats in the summaries folder of the logdir config folder"""
    self.summary_path = os.path.join(config.logdir, "summaries")
    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)
    """Save successor representation vectors plotted over the environment in the sr_vector inside the summaries folder"""
    tf.gfile.MakeDirs(os.path.join(os.path.join(config.logdir, "summaries"), "sr_vectors"))

    """Environment configuration"""
    self.action_size = game.action_space.n
    self.nb_states = config.input_size[0] * config.input_size[1]
    self.env = game
    self.sess = sess

    """Setting the summary information"""
    self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
    self.summary = tf.Summary()

    """Instantiating local network for function approximation of the policy and state space"""
    self.local_network = config.network(self.name, config, self.action_size)
    self.update_local_vars = update_target_graph('global', self.name)

  """Do one n-step update for training the agent's latent successor representation space"""
  def train(self, rollout, bootstrap_sf):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    fi = np.identity(self.nb_states)[observations]

    """Construct list of one=hot encodings for the entire trajectory"""
    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    """Construct the targets for the next step successor representations for the entire trajectory"""
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.identity(self.nb_states)[observations]}

    _, self.summaries, loss = \
      self.sess.run([self.local_network.apply_grads,
                self.local_network.merged_summary,
                self.local_network.loss],
               feed_dict=feed_dict)

  """Builds the SR matrix. Plots it. Does eigendecomposition and replots everything on the env"""
  def build_SR_matrix(self):
    """Building the SR matrix"""
    with self.sess.as_default(), self.sess.graph.as_default():
      self.matrix_sf = np.zeros((self.nb_states, self.nb_states))
      indices = []
      for idx in range(self.nb_states):
        ii, jj = self.env.get_state_xy(idx)
        if self.env.not_wall(ii, jj):
          indices.append(idx)

      feed_dict = {self.local_network.observation: np.identity(self.nb_states)[indices]}
      self.matrix_sf[indices] = self.sess.run(self.local_network.sf, feed_dict=feed_dict)

    """Plot the SR matrix"""
    self.plot_sr_matrix()
    """Plot individual SR vectors back over the 4 Rooms environment"""
    self.plot_sr_vectors("sr_vectors")
    """Do eigendecomposition and plot eigenvectors over the 4 Rooms environment"""
    self.eigen_decomp()

  """"Plots the SR matrix"""
  def plot_sr_matrix(self):
    sns.plt.clf()
    sns.set_style('ticks')
    ax = sns.heatmap(self.matrix_sf, cmap="Blues")
    ax.set(xlabel='SR_vect_size=169', ylabel='Grid states/positions')
    folder_path = os.path.join(self.summary_path, "state_space_matrix")
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'SR_matrix.png'))
    sns.plt.close()

  """Reproject and plot the individual SR vectors over the 4 Rooms environment"""
  def plot_sr_vectors(self, folder):
    sns.plt.clf()
    """"Where to save the plots"""
    folder_path = os.path.join(self.summary_path, folder)
    tf.gfile.MakeDirs(folder_path)

    for i in range(self.nb_states):
      aa, bb = self.env.get_state_xy(i)
      if self.env.not_wall(aa, bb):
        """Reproject the SR vector over the environment"""
        sr_vector = self.matrix_sf[i].reshape(self.config.input_size[0], self.config.input_size[1])
        ax = sns.heatmap(sr_vector, cmap="Blues")

        """Add borders"""
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
        """Save the plot"""
        sns.plt.savefig(os.path.join(folder_path, "SR_VECTOR_" + str(i) + '.png'))
        sns.plt.close()

  def eigen_decomp(self):
    """Where to save the eigenvectors, the policies and the value functions"""
    eigenvector_folder = os.path.join(self.summary_path, "eigenvectors")
    tf.gfile.MakeDirs(eigenvector_folder)

    policy_folder = os.path.join(self.summary_path, "policies")
    tf.gfile.MakeDirs(policy_folder)

    v_folder = os.path.join(self.summary_path, "value_functions")
    tf.gfile.MakeDirs(v_folder)

    """Perform eigendecomposition - in this case SVD for completeness"""
    u, s, v = np.linalg.svd(self.matrix_sf)
    # noise_reduction = s > 1
    # s = s[noise_reduction]
    # v = v[noise_reduction]

    """Plot eigenvectors"""
    self.plot_eigenvectors(s, v, eigenvector_folder)
    """Plot policies and value functions"""
    self.plot_policy_and_value_function(v, policy_folder, v_folder)

  """Reproject and plot eigenvectors"""
  def plot_eigenvectors(self, eigenvalues, eigenvectors, eigenvector_folder):
    sns.plt.clf()
    for k in ["poz", "neg"]:
      for i in range(len(eigenvalues)):
        reproj_eigenvector = eigenvectors[i].reshape(self.config.input_size[0], self.config.input_size[1])
        """Take both signs"""
        if k == "neg":
          reproj_eigenvector = -reproj_eigenvector
        """Plot of the eigenvector"""
        ax = sns.heatmap(reproj_eigenvector, cmap="Blues")

        """Adding borders"""
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
        """Saving plots"""
        sns.plt.savefig(os.path.join(eigenvector_folder, ("Eigenvector" + str(i) + '_eig_' + k + '.png')))
        sns.plt.close()

    """Plot also the eigenvalues"""
    sns.plt.plot(eigenvalues, 'o')
    sns.plt.savefig(os.path.join(eigenvector_folder,  'eigenvalues.png'))
    sns.plt.close()

  """Plot plicies and value functions"""
  def plot_policy_and_value_function(self, eigenvectors, policy_folder, v_folder):
    epsilon = 0.0001
    for k in ["poz", "neg"]:
      for i in range(len(eigenvectors)):
        """Do policy iteration"""
        discount = 0.9
        polIter = PolicyIteration(discount, self.env, augmentActionSet=True)
        """Use the direction of the eigenvector as intrinsic reward for the policy iteration algorithm"""
        self.env.define_reward_function(eigenvectors[i] if k == "poz" else -eigenvectors[i])
        """Get the optimal value function and policy"""
        V, pi = polIter.solvePolicyIteration()

        for j in range(len(V)):
          if V[j] < epsilon:
            pi[j] = len(self.env.get_action_set())

        """Plot them"""
        self.plot_value_function(V[0:self.nb_states], str(i) + '_' + k + "_", v_folder)
        self.plot_policy(pi[0:self.nb_states], str(i) + '_' + k + "_", policy_folder)

  """Plot value functions"""
  def plot_value_function(self, value_function, prefix, v_folder):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X, Y = np.meshgrid(np.arange(self.config.input_size[1]), np.arange(self.config.input_size[0]))
    reproj_value_function = value_function.reshape(self.config.input_size[0], self.config.input_size[1])

    """Build the support"""
    for i in range(len(X)):
        for j in range(int(len(X[i]) / 2)):
            tmp = X[i][j]
            X[i][j] = X[i][len(X[i]) - j - 1]
            X[i][len(X[i]) - j - 1] = tmp

    cm.jet(np.random.rand(reproj_value_function.shape[0], reproj_value_function.shape[1]))

    ax.plot_surface(X, Y, reproj_value_function, rstride=1, cstride=1,
                    cmap=plt.get_cmap('jet'))
    plt.gca().view_init(elev=30, azim=30)
    plt.savefig(os.path.join(v_folder, "SuccessorFeatures" + prefix + 'value_function.png'))
    plt.close()

  """Plot the policy"""
  def plot_policy(self, policy, prefix, policy_folder):
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

    plt.savefig(os.path.join(policy_folder, "SuccessorFeatures_" + prefix + 'policy.png'))
    plt.close()

  """Starting point of the agent acting in the environment"""
  def play(self, coord, saver):
    self.saver = saver

    with self.sess.as_default(), self.sess.graph.as_default():
      self.global_episode_np = self.sess.run(self.global_episode)
      self.global_step_np = self.sess.run(self.global_step)

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if self.global_step_np > self.config.steps and self.config.steps != -1 and self.name == "worker_0":
          coord.request_stop()
          return 0

        """update local network parameters from global network"""
        self.sess.run(self.update_local_vars)

        """initializations"""
        episode_buffer = []
        self.episode_reward = 0
        d = False
        self.episode_length = 0

        """Reset the environment and get the initial state"""
        s = self.env.get_initial_state()

        """While the episode does not terminate"""
        while not d:
          """act according to the behaviour policy - random walk"""
          a = np.random.choice(range(self.action_size))

          _, r, d, s1 = self.env.special_step(a, s)

          """Add transition to buffer - this case is just the stqate"""
          episode_buffer.append([s])
          self.episode_reward += r
          self.episode_length += 1
          s = s1

          """Do n-step prediction over the successor representation"""
          if len(episode_buffer) == self.config.max_update_freq or d:
            """Get the successor features of the next state for which to bootstrap from"""
            next_sf = self.sess.run(self.local_network.sf,
                               feed_dict={self.local_network.observation: np.identity(self.nb_states)[s:s+1]})[0]
            bootstrap_sf = np.zeros_like(next_sf) if d else next_sf
            """Do one update step"""
            self.train(episode_buffer, bootstrap_sf)

            """Clear buffer for the next n steps"""
            episode_buffer = []

          if self.name == "worker_0":
            self.sess.run(self.increment_global_step)
            self.global_step_np = self.global_step.eval()

        if self.name == "worker_0":
          self.sess.run(self.increment_global_episode)
          self.global_episode_np = self.global_episode.eval()

          if self.global_episode_np % self.config.checkpoint_interval == 0:
            self.save_model()

          if self.global_episode_np % self.config.summary_interval == 0:
            self.write_summaries()

  def write_summaries(self):
    self.summary.value.add(tag='Perf/Reward', simple_value=float(self.episode_reward))
    self.summary.value.add(tag='Perf/Length', simple_value=float(self.episode_length))

    self.summary_writer.add_summary(self.summaries, self.global_episode_np)
    self.summary_writer.add_summary(self.summary, self.global_episode_np)
    self.summary_writer.flush()

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-' + str(self.global_episode_np) + '.cptk',
               global_step=self.global_episode)
    print("Saved Model at {}".format(self.model_path + '/model-' + str(self.global_episode_np) + '.cptk'))

  def cluster(self, coord):
    with self.sess.as_default(), self.sess.graph.as_default():
      self.global_episode_np = self.sess.run(self.global_episode)
      self.global_step_np = self.sess.run(self.global_step)
      # c = OnlineCluster(4, self.nb_states)
      c = self.global_network.direction_clusters
      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if self.global_step_np > self.config.steps and self.config.steps != -1 and self.name == "worker_0":
          coord.request_stop()
          return 0

        """update local network parameters from global network"""
        self.sess.run(self.update_local_vars)

        d = False
        self.episode_length = 0

        """Reset the environment and get the initial state"""
        s = self.env.get_initial_state()
        """While the episode does not terminate"""
        while not d:
          """act according to the behaviour policy - random walk"""
          a = np.random.choice(range(self.action_size))
          sf = self.sess.run(self.local_network.sf,
                             feed_dict={self.local_network.observation: np.identity(self.nb_states)[s:s + 1]})[0]
          c.cluster(sf)
          _, r, d, s1 = self.env.special_step(a, s)

          self.episode_length += 1
          s = s1

          if self.name == "worker_0":
            self.sess.run(self.increment_global_step)
            self.global_step_np = self.global_step.eval()

        if self.name == "worker_0":
          self.sess.run(self.increment_global_episode)
          self.global_episode_np = self.global_episode.eval()

          if self.global_episode_np % 10 == 0:
            print("Reclustering")
            clusters = c.get_clusters()
            """Where to save the eigenvectors, the policies and the value functions"""
            clusters_folder = os.path.join(self.summary_path, "clusters")
            tf.gfile.MakeDirs(clusters_folder)

            policy_folder = os.path.join(self.summary_path, "policies_clusters")
            tf.gfile.MakeDirs(policy_folder)

            v_folder = os.path.join(self.summary_path, "value_functions_clusters")
            tf.gfile.MakeDirs(v_folder)

            self.plot_clusters(clusters, clusters_folder)
            """Plot policies and value functions"""
            self.plot_policy_and_value_function(clusters, policy_folder, v_folder)

  """Reproject and plot cluster directions"""
  def plot_clusters(self, clusters, cluster_folder):
    sns.plt.clf()
    for i in range(len(clusters)):
      reproj_eigenvector = clusters[i].reshape(self.config.input_size[0], self.config.input_size[1])
      """Take both signs"""
      """Plot of the eigenvector"""
      ax = sns.heatmap(reproj_eigenvector, cmap="Blues")

      """Adding borders"""
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
      """Saving plots"""
      sns.plt.savefig(os.path.join(cluster_folder, ("Direction" + str(i) + '.png')))
      sns.plt.close()
