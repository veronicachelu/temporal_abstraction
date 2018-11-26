import numpy as np
import tensorflow as tf
from tools.agent_utils import update_target_graph, update_target_graph_aux, update_target_graph_sf, discount, \
  make_gif
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
from PIL import Image
import scipy.stats
import seaborn as sns

sns.set()
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from auxilary.policy_iteration import PolicyIteration

FLAGS = tf.app.flags.FLAGS

"""This Agent corresponds to the case of performing non-linear approximation over pixel observations for the 4 Rooms Domain"""
class DynSRAgent():
  def __init__(self, sess, game, thread_id, global_step, global_episode, config, global_netowork, barrier):
    self.name = "worker_" + str(thread_id)
    self.thread_id = thread_id

    self.config = config
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.global_episode = global_episode
    self.increment_global_step = self.global_step.assign_add(1)
    self.increment_global_episode = self.global_episode.assign_add(1)

    """Local thread timestep counter"""
    self.total_steps = 0

    """Save models in the models directory in the logdir config folder"""
    self.model_path = os.path.join(config.logdir, "models")
    """Save events file and other stats in the summaries folder of the logdir config folder"""
    self.summary_path = os.path.join(config.logdir, "summaries")
    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)

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
    self.update_local_vars_aux = update_target_graph_aux('global', self.name)
    self.update_local_vars_sf = update_target_graph_sf('global', self.name)

    """Experience reply for the auxilary task of next frame prediction"""
    self.aux_episode_buffer = deque()

  """Do one n-step update for training the agent's latent successor representation space"""
  def train_sf(self, rollout, bootstrap_sf):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    # next_observations = rollout[:, 1]
    # actions = rollout[:, 2]

    """Get the latent representations for each state"""
    feed_dict = {self.local_network.observation: np.stack(observations, axis=0)}
    fi = self.sess.run(self.local_network.fi,
                  feed_dict=feed_dict)
    """Construct list of latent representations for the entire trajectory"""
    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    """Construct the targets for the next step successor representations for the entire trajectory"""
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0)}  # ,
    _, self.summaries_sf, sf_loss = \
      self.sess.run([self.local_network.apply_grads_sf,
                self.local_network.merged_summary_sf,
                self.local_network.sf_loss],
               feed_dict=feed_dict)

  """Do one minibatch update over the next frame prediction network"""
  def train_aux(self):
    minibatch = random.sample(self.aux_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.actions_placeholder: actions}

    aux_loss, _, self.summaries_aux = \
      self.sess.run([self.local_network.aux_loss,
                     self.local_network.apply_grads_aux,
                     self.local_network.merged_summary_aux],
               feed_dict=feed_dict)

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
        self.sess.run(self.update_local_vars_aux)
        self.sess.run(self.update_local_vars_sf)

        """initializations"""
        episode_buffer = []
        self.episode_reward = 0
        d = False
        self.episode_length = 0

        """Reset the environment and get the initial state"""
        s = self.env.reset()

        """While the episode does not terminate"""
        while not d:
          """Every once in a while update local network parameters from global network"""
          if self.total_steps % self.config.target_update_iter_aux == 0:
            self.sess.run(self.update_local_vars_aux)
          if self.total_steps % self.config.target_update_iter_sf == 0:
            self.sess.run(self.update_local_vars_sf)

          """act according to the behaviour policy - random walk"""
          a = np.random.choice(range(self.action_size))

          s1, r, d, _ = self.env.step(a)

          """If the episode ended make the last state absorbing"""
          if d:
            s1 = s

          """If the next state prediction buffer is full override the oldest memories"""
          if len(self.aux_episode_buffer) == self.config.memory_size:
            self.aux_episode_buffer.popleft()
          """Append transition to experience reply buffer for next state prediction"""
          self.aux_episode_buffer.append([s, s1, a])

          self.episode_reward += r
          self.episode_length += 1


          """When we are done observing - warm start"""
          if self.total_steps > self.config.observation_steps:
            """Add transition to buffer"""
            episode_buffer.append([s, s1, a])

            """If the experience buffer has sufficient experience in it, every so often do an update with a batch of transition from it for next state prediction"""
            if len(self.aux_episode_buffer) > self.config.observation_steps and \
                        self.total_steps % self.config.aux_update_freq == 0:
              self.train_aux()

            """Do n-step update over the successor representation"""
            if len(episode_buffer) >= self.config.max_update_freq or d:
              """Get the successor features of the next state for which to bootstrap from"""
              next_sf = self.sess.run(self.local_network.sf,
                                        feed_dict={self.local_network.observation: [s]})[0]
              bootstrap_sf = np.zeros_like(next_sf) if d else next_sf
              self.train_sf(episode_buffer, bootstrap_sf)

              """Clear buffer for the next n steps"""
              episode_buffer = []
              
          s = s1
          self.total_steps += 1


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

    if self.summaries_sf is not None:
      self.summary_writer.add_summary(self.summaries_sf, self.global_episode_np)
    if self.summaries_aux is not None:
      self.summary_writer.add_summary(self.summaries_aux, self.global_episode_np)
    self.summary_writer.add_summary(self.summary, self.total_steps)

    self.summary_writer.flush()

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-' + str(self.global_episode_np) + '.cptk',
                    global_step=self.global_episode)
    print("Saved Model at {}".format(self.model_path + '/model-' + str(self.global_episode_np) + '.cptk'))

  """Builds the SR matrix. Plots it. Does eigendecomposition and replots everything on the env"""
  def build_SR_matrix(self):
    """If the matrices have already been constructed load them"""
    numpy_models_path = os.path.join(self.config.logdir, "numpy_models")
    tf.gfile.MakeDirs(numpy_models_path)

    sr_matrix_path = os.path.join(numpy_models_path, "sr_matrix.npy")
    fi_matrix_path = os.path.join(numpy_models_path, "fi_matrix.npy")

    if os.path.exists(fi_matrix_path) and os.path.exists(sr_matrix_path):
      self.matrix_sf = np.load(sr_matrix_path)
      self.matrix_fi = np.load(fi_matrix_path)
    else:
      """Otherwise build the SR matrix from scratch"""
      with self.sess.as_default(), self.sess.graph.as_default():
        self.matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
        self.matrix_fi = np.zeros((self.nb_states, self.config.sf_layers[-1]))
        indices = []
        states = []
        for idx in range(self.nb_states):
          s, ii, jj = self.env.get_state(idx)
          if self.env.not_wall(ii, jj):
            indices.append(idx)
            states.append(s)

        feed_dict = {self.local_network.observation: states}
        fi, sf = self.sess.run([self.local_network.fi, self.local_network.sf], feed_dict=feed_dict)
        self.matrix_fi[indices] = fi
        self.matrix_sf[indices] = sf

        np.save(sr_matrix_path, self.matrix_sf)
        np.save(fi_matrix_path, self.matrix_fi)

    """Plot the SR matrix"""
    self.plot_sr_matrix()
    """Do eigendecomposition and plot eigenvectors over the 4 Rooms environment"""
    self.eigen_decomp()

  """"Plots the SR matrix"""
  def plot_sr_matrix(self):
    import seaborn as sns
    sns.plt.clf()
    ax = sns.heatmap(self.matrix_sf, cmap="Blues")
    ax.set(xlabel='SR_vect_size=128', ylabel='Grid states/positions')
    folder_path = os.path.join(self.summary_path, "state_space_matrix")
    tf.gfile.MakeDirs(folder_path)
    sns.plt.savefig(os.path.join(folder_path, 'SR_matrix.png'))
    sns.plt.close()

  def eigen_decomp(self):
    """Where to save the eigenvectors, the policies and the value functions"""
    eigenvector_folder = os.path.join(self.summary_path, "eigenvectors")
    tf.gfile.MakeDirs(eigenvector_folder)

    policy_folder = os.path.join(self.summary_path, "policies")
    tf.gfile.MakeDirs(policy_folder)

    v_folder = os.path.join(self.summary_path, "value_functions")
    tf.gfile.MakeDirs(v_folder)

    """Perform eigendecomposition"""
    u, s, v = np.linalg.svd(self.matrix_sf, full_matrices=False)

    """Plot eigenvectors"""
    self.plot_eigenvectors(s, v, eigenvector_folder)

    """Plot policies and value functions"""
    self.plot_policy_and_value_function(s, v, policy_folder, v_folder)

  """Plot eigenvectors"""
  def plot_eigenvectors(self, eigenvalues, eigenvectors, eigenvector_folder):
    sns.plt.clf()
    ax = sns.heatmap(eigenvectors, cmap="Blues")
    ax.set(xlabel='Eigenvector_dim=128', ylabel='Eigenvectors')
    sns.plt.savefig(os.path.join(eigenvector_folder, 'Eigenvectors.png'))

    """Plot also the eigenvalues"""
    sns.plt.plot(eigenvalues, 'o')
    sns.plt.savefig(os.path.join(eigenvector_folder, 'eigenvalues.png'))

    sns.plt.close()

    eigenvectors_path = os.path.join(eigenvector_folder, "eigenvectors.npy")
    eigenvalues_path = os.path.join(eigenvector_folder, "eigenvalues.npy")
    np.save(eigenvectors_path, eigenvectors)
    np.save(eigenvalues_path, eigenvalues)

  """Plot plicies and value functions"""
  def plot_policy_and_value_function(self,  eigenvalues, eigenvectors, policy_folder, v_folder):
    epsilon = 0.0001
    with self.sess.as_default(), self.sess.graph.as_default():
      self.env.define_network(self.local_network)
      self.env.define_session(self.sess)
      for k in ["poz", "neg"]:
        for i in range(len(eigenvalues)):
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
