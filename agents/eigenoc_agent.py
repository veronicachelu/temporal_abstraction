import numpy as np
import tensorflow as tf
from tools.agent_utils import get_mode, update_target_graph_aux, update_target_graph_sf, \
  update_target_graph_option, discount, reward_discount, set_image, make_gif
import os

import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
from collections import deque
import seaborn as sns

sns.set()
import random
import matplotlib.pyplot as plt
import copy
from threading import Barrier, Thread
from tools.timer import Timer
from auxilary.policy_iteration import PolicyIteration

FLAGS = tf.app.flags.FLAGS

"""This Agent corresponds is can be used in more than one test case:
* if config.use_eigendirections is set to False it defaults to the classic Option-Critic
* if config.use_eigendirections is set to True than the options folllow eigendirections of the svd decomposition of the Successor Representation matrix buffer to which it keeps appending as it explores the environment"""
class EigenOCAgent():
  def __init__(self, sess, game, thread_id, global_step, global_episode, config, global_network, barrier):
    self.name = "worker_" + str(thread_id)
    self.thread_id = thread_id

    self.config = config
    self.barrier = barrier
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.global_episode = global_episode
    self.increment_global_step = self.global_step.assign_add(1)
    self.increment_global_episode = self.global_episode.assign_add(1)

    """Local thread timestep and episode counter"""
    self.total_steps = 0
    self.total_episodes = 0

    """Save models in the models directory in the logdir config folder"""
    self.model_path = os.path.join(config.logdir, "models")
    """Save events file and other stats in the summaries folder of the logdir config folder"""
    self.summary_path = os.path.join(config.logdir, "summaries")
    """Save eval/test files and other stats in the test folder of the logdir config folder"""
    self.test_path = os.path.join(config.logdir, "test")
    """Save stats in the stats folder of the summary config folder"""
    self.stats_path = os.path.join(self.summary_path, 'stats')
    tf.gfile.MakeDirs(self.stats_path)
    tf.gfile.MakeDirs(self.test_path)
    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)

    """Environment configuration"""
    self.action_size = game.action_space.n
    self.nb_states = config.input_size[0] * config.input_size[1]
    self.env = game
    self.sess = sess
    self.nb_options = config.nb_options

    self.global_network = global_network
    self.load_eigendirections()

    """Buffers for keeping stats"""
    self.episode_mean_q_values = []
    self.episode_mean_eigen_q_values = []
    self.episode_mean_returns = []
    self.episode_mean_oterms = []
    self.episode_mean_options = []
    self.episode_mean_actions = []
    self.episode_options = []
    self.episode_actions = []
    self.episode_term_prob = []
    self.episode_primtive_action_prob = []
    self.episode_mean_values = []

    """Setting the summary information"""
    self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
    self.summary = tf.Summary()

    """Instantiating local network for function approximation of the policy and state space"""
    self.local_network = config.network(self.name, config, self.action_size)
    self.update_local_vars_aux = update_target_graph_aux('global', self.name)
    self.update_local_vars_sf = update_target_graph_sf('global', self.name)
    self.update_local_vars_option = update_target_graph_option('global', self.name)

    """Experience reply for the auxilary task of next frame prediction"""
    self.aux_episode_buffer = deque()

  """Load eigendirections from the global shared network"""
  def load_eigendirections(self):
    """If we use a successor representation buffer, load the eigen directions"""
    if self.config.sr_matrix is not None:
      self.directions = self.global_network.directions

  """Sync thread params with the global network"""
  def sync_threads(self, force=False):
    if force:
      self.sess.run(self.update_local_vars_sf)
      self.sess.run(self.update_local_vars_aux)
      self.sess.run(self.update_local_vars_option)
    else:
      if self.total_steps % self.config.target_update_iter_sf == 0:
        self.sess.run(self.update_local_vars_sf)
      if self.total_steps % self.config.target_update_iter_aux == 0:
        self.sess.run(self.update_local_vars_aux)
      if self.total_steps % self.config.target_update_iter_option == 0:
        self.sess.run(self.update_local_vars_option)

  """Initialize agent"""
  def init_agent(self):
    self.global_episode_np = self.sess.run(self.global_episode)
    self.global_step_np = self.sess.run(self.global_step)

    tf.logging.info("Starting worker " + str(self.thread_id))

    """In the continual learning scenario, every 'move_goal_nb_of_ep' we change the position of the goal
    and such the reward signal"""
    self.goal_position = self.env.set_goal(self.total_episodes, self.config.move_goal_nb_of_ep)

    """We only let the first thread write stats"""
    if self.name == "worker_0":
      self.init_tracker()

  """Redo the SVD decomposition of the successor representation buffer matrix"""
  def recompute_eigendirections(self):
    if self.name == "worker_0" and self.global_episode_np > 0 and \
        self.config.use_eigendirections:
      self.recompute_eigenvectors_svd()

  """Initialize a new episode"""
  def init_episode(self):
    """Keep episode buffers for storing n transitions in order to do n-step prediction of different estimators"""
    self.episode_buffer_sf = []
    self.episode_buffer_option = []
    self.episode_values = []
    self.episode_q_values = []
    self.episode_eigen_q_values = []
    self.episode_oterm = []
    self.episode_options = []
    self.episode_actions = []

    self.episode_reward = 0
    self.done = False
    self.o_term = True
    self.episode_length = 0
    self.s_idx = None

    """initialization for the stats tracker"""
    col_size = self.nb_options + self.action_size if self.config.include_primitive_options else self.nb_options
    self.o_tracker_chosen = np.zeros((col_size,), dtype=np.int32)
    self.o_tracker_steps = np.zeros(col_size, dtype=np.int32)
    self.o_tracker_len = [[] for _ in range(col_size)]
    self.termination_counter = 0
    self.primitive_action_counter = 0
    self.stats_options = np.zeros((self.nb_states, col_size))
    self.stats_actions = np.zeros((self.nb_states, self.action_size))

    """If we set the decrease_option_prob in the config file to True, than we should decrease the probability of taking a random option at each episode"""
    if self.config.decrease_option_prob and self.total_episodes < self.config.explore_options_episodes:
      self.sess.run(self.local_network.decrease_prob_of_random_option)

    """Initialize summaries"""
    self.summaries_aux = self.summaries_critic = self.summaries_option = self.summaries_sf = self.summaries_termination = None
    self.R = self.eigen_R = None
    self.eigen_q_value = None

  def add_stats_to_tracker(self):
    if self.s_idx is not None:
      self.stats_actions[self.s_idx][self.action] += 1
      self.stats_options[self.s_idx][self.option] += 1

  """Do an update for the representation latent using 1-step next frame prediction"""
  def next_frame_prediction(self):
    if len(self.aux_episode_buffer) > self.config.observation_steps and \
                self.total_steps % self.config.aux_update_freq == 0:
      self.train_aux()

  """Do n-step prediction for the returns and update the option policies and critics"""
  def option_prediction(self, s, s1):
    """If we use eigendirections as basis and the option chosen was not primitive, than we can construct
    the mixed reward signal to pass to the eigen intra-option critics."""
    if self.config.use_eigendirections and not self.primitive_action:
      feed_dict = {self.local_network.observation: np.stack([s, s1])}
      fi = self.sess.run(self.local_network.fi,
                         feed_dict=feed_dict)
      """The internal reward will be the cosine similary between the direction in latent space and the 
      eigen direction corresponding to the current option"""
      r_i = self.cosine_similarity((fi[1] - fi[0]), self.directions[self.option])
      r_mix = self.config.alpha_r * r_i + (1 - self.config.alpha_r) * self.reward
    else:
      r_mix = self.reward

    """Adding to the transition buffer for doing n-step prediction on critics and policies"""
    self.episode_buffer_option.append(
      [s, self.option, self.action, self.reward, r_mix, self.primitive_action, s1])

    if len(self.episode_buffer_option) >= self.config.max_update_freq or self.done or (
          self.o_term and len(self.episode_buffer_option) >= self.config.min_update_freq):
      """Get the bootstrap option-value functions for the next time step"""
      if self.done:
        bootstrap_Q = 0
        bootstrap_eigen_Q = 0
      else:
        feed_dict = {self.local_network.observation: [s1]}
        if self.config.use_eigendirections:
          value, eigen_value, q_value, eigen_q_value = self.sess.run(
            [self.local_network.v, self.local_network.eigenv, self.local_network.q_val,
             self.local_network.eigen_q_val],
            feed_dict=feed_dict)
          q_value = q_value[0, self.option]
          value = value[0]

          if self.primitive_action:
            bootstrap_eigen_Q = value if self.o_term else q_value
          else:
            eigen_q_value = eigen_q_value[0, self.option]
            eigen_value = eigen_value[0]
            bootstrap_eigen_Q = eigen_value if self.o_term else eigen_q_value
        else:
          value, q_value = self.sess.run(
            [self.local_network.v, self.local_network.q_val],
            feed_dict=feed_dict)
          q_value = q_value[0, self.option]
          value = value[0]
        bootstrap_Q = value if self.o_term else q_value

        if not self.config.use_eigendirections:
          bootstrap_eigen_Q = bootstrap_Q

      self.train_option(bootstrap_Q, bootstrap_eigen_Q)
      self.episode_buffer_option = []

    return r_mix

  """Starting point of the agent acting in the environment"""
  def play(self, coord, saver):
    self.saver = saver

    with self.sess.as_default(), self.sess.graph.as_default():
      self.init_agent()

      with coord.stop_on_exception():
        while not coord.should_stop():
          if (self.config.steps != -1 and \
                  (self.global_step_np > self.config.steps and self.name == "worker_0")) or \
              (self.global_episode_np > len(self.config.goal_locations) * self.config.move_goal_nb_of_ep and
                   self.name == "worker_0" and self.config.multi_task):
            coord.request_stop()
            return 0

          """update local network parameters from global network"""
          self.sync_threads()

          self.recompute_eigendirections()
          self.load_eigendirections()
          self.init_episode()

          """Reset the environment and get the initial state"""
          s = self.env.reset()

          """Choose an option"""
          self.option_evaluation(s)
          """Increase the timesteps for the options - for statistics purposes"""
          self.o_tracker_steps[self.option] += 1
          """While the episode does not terminate"""
          while not self.done:
            """update local network parameters from global network"""
            self.sync_threads()

            """Choose an action from the current intra-option policy"""
            self.policy_evaluation(s)
            self.add_stats_to_tracker()

            s1, r, self.done, self.s1_idx = self.env.step(self.action)

            self.crt_op_length += 1
            self.episode_reward += r
            self.reward = np.clip(r, -1, 1)

            """Check if the option terminates at the next state"""
            self.option_terminate(s1)

            """If we use deliberation costs than the value of the reward is dependent upon option termination"""
            self.reward_deliberation()

            """If the episode ended make the last state absorbing"""
            if self.done:
              s1 = s
              self.s1_idx = self.s_idx

            """If the next state prediction buffer is full override the oldest memories"""
            if len(self.aux_episode_buffer) == self.config.memory_size:
              self.aux_episode_buffer.popleft()
            if self.config.history_size <= 3:
              self.aux_episode_buffer.append([s, s1, self.action])
            else:
              self.aux_episode_buffer.append([s, s1[:, :, -2:-1], self.action])

            """If we used eigen directions as basis for the options that store transitions for n-step successor representation predictions"""
            if self.config.use_eigendirections:
              self.episode_buffer_sf.append([s, s1, self.action])
              self.sf_prediction(s1)

            """If the experience buffer has sufficient experience in it, every so often do an update with a batch of transition from it for next state prediction"""
            self.next_frame_prediction()

            """Do n-step prediction for the returns"""
            r_mix = self.option_prediction(s, s1)

            """If the option terminated or the option was primitive, sample another option"""
            if not self.done and (self.o_term or self.primitive_action):
              self.option_evaluation(s1)

            if not self.done:
              """Increase the timesteps for the options - for statistics purposes"""
              self.o_tracker_steps[self.option] += 1

            if self.total_steps % self.config.step_summary_interval == 0 and self.name == 'worker_0':
              self.write_step_summary(r, r_mix)

            s = s1
            self.s_idx = self.s1_idx
            self.episode_length += 1
            self.total_steps += 1

            if self.name == "worker_0":
              self.sess.run(self.increment_global_step)
              self.global_step_np = self.global_step.eval()

          self.update_episode_stats()

          if self.name == "worker_0":
            self.sess.run(self.increment_global_episode)
            self.global_episode_np = self.global_episode.eval()

            if self.global_episode_np % self.config.checkpoint_interval == 0:
              self.save_model()

            if self.global_episode_np % self.config.summary_interval == 0:
              self.write_summaries()

          """If it's time to change the task - move the goal, wait for all other threads to finish the current task"""
          if self.total_episodes % self.config.move_goal_nb_of_ep == 0 and \
                  self.total_episodes != 0:
            tf.logging.info("Moving GOAL....")
            self.barrier.wait()
            self.goal_position = self.env.set_goal(self.total_episodes, self.config.move_goal_nb_of_ep)

          self.total_episodes += 1

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-{}.cptk'.format(self.global_episode_np),
                    global_step=self.global_episode)
    tf.logging.info(
      "Saved Model at {}".format(self.model_path + '/model-{}.cptk'.format(self.global_episode_np)))

  def write_summaries(self):
    self.tracker()
    self.summary = tf.Summary()
    self.summary.value.add(tag='Perf/Reward', simple_value=float(self.episode_reward))
    self.summary.value.add(tag='Perf/Length', simple_value=float(self.episode_length))

    for sum in [self.summaries_sf, self.summaries_aux, self.summaries_termination, self.summaries_critic, self.summaries_option]:
      if sum is not None:
        self.summary_writer.add_summary(sum, self.global_episode_np)

    if len(self.episode_term_prob) != 0:
      mean_term_prob = np.mean(self.episode_term_prob[-self.config.step_summary_interval:])
      self.summary.value.add(tag='Perf/Term_prob', simple_value=float(mean_term_prob))
    if len(self.episode_primtive_action_prob) != 0:
      mean_primitive_prob = np.mean(self.episode_primtive_action_prob[-self.config.step_summary_interval:])
      self.summary.value.add(tag='Perf/Primitive_prob', simple_value=float(mean_primitive_prob))
    if len(self.episode_mean_values) != 0:
      last_mean_value = np.mean(self.episode_mean_values[-self.config.step_summary_interval:])
      self.summary.value.add(tag='Perf/Value', simple_value=float(last_mean_value))
    if len(self.episode_mean_q_values) != 0:
      last_mean_q_value = np.mean(self.episode_mean_q_values[-self.config.step_summary_interval:])
      self.summary.value.add(tag='Perf/QValue', simple_value=float(last_mean_q_value))
    if self.config.eigen and len(self.episode_mean_eigen_q_values) != 0:
      last_mean_eigen_q_value = np.mean(self.episode_mean_eigen_q_values[-self.config.step_summary_interval:])
      self.summary.value.add(tag='Perf/EigenQValue', simple_value=float(last_mean_eigen_q_value))
    if len(self.episode_mean_oterms) != 0:
      last_mean_oterm = self.episode_mean_oterms[-1]
      self.summary.value.add(tag='Perf/Oterm', simple_value=float(last_mean_oterm))
    if len(self.episode_mean_options) != 0:
      last_frequent_option = self.episode_mean_options[-1]
      self.summary.value.add(tag='Perf/FreqOptions', simple_value=last_frequent_option)
    if len(self.episode_mean_options) != 0:
      last_frequent_action = self.episode_mean_actions[-1]
      self.summary.value.add(tag='Perf/FreqActions', simple_value=last_frequent_action)

    self.summary_writer.add_summary(self.summary, self.global_episode_np)
    self.summary_writer.flush()

  """Decrease the reward in accordance to the option termination probability.
     If we terminated it means we need to pay a deliberation cost for having to choose the next optimal option"""
  def reward_deliberation(self):
    self.reward = float(self.reward) - self.config.discount * (
      float(self.o_term) * self.config.delib_cost * (1 - float(self.done)))

  """Take an option using an epsilon-greedy approach with respect to the option-value functions"""
  def option_evaluation(self, s):
    feed_dict = {self.local_network.observation: np.stack([s])}
    self.option, self.primitive_action = self.sess.run(
      [self.local_network.current_option, self.local_network.primitive_action], feed_dict=feed_dict)
    self.option, self.primitive_action = self.option[0], self.primitive_action[0]

    """Keep track of statistics for the chosen options"""
    self.o_tracker_chosen[self.option] += 1
    self.episode_options.append(self.option)
    self.primitive_action_counter += self.primitive_action * (1 - self.done)
    self.crt_op_length = 0

  """Check is the option terminates at the next state"""
  def option_terminate(self, s1):
    """If we took a primitive option, termination is assured"""
    if self.config.include_primitive_options and self.primitive_action:
      self.o_term = True
    else:
      feed_dict = {self.local_network.observation: [s1]}
      o_term = self.sess.run(self.local_network.termination, feed_dict=feed_dict)
      self.o_term = o_term[0, self.option] > np.random.uniform()
      self.prob_terms = o_term[0]

    """Stats for tracking option termination"""
    self.termination_counter += self.o_term * (1 - self.done)
    self.episode_oterm.append(self.o_term)
    self.o_tracker_len[self.option].append(self.crt_op_length)

  """Sample an action from the current option's policy"""
  def policy_evaluation(self, s):
    feed_dict = {self.local_network.observation: [s]}
    """If we use eigendirections as basis for the options"""
    if self.config.use_eigendirections:
      tensor_list = [self.local_network.options,
                     self.local_network.v,
                     self.local_network.q_val,
                     self.local_network.eigen_q_val,
                     self.local_network.eigenv]
      options,\
      value,\
      q_value,\
      eigen_q_value,\
      evalue = self.sess.run(tensor_list, feed_dict=feed_dict)
      """If the current option is not a primitive action"""
      if not self.primitive_action:
        """Add the eigen option-value function to the buffer in order to add stats to tensorboad at the end of the episode"""
        self.eigen_q_value = eigen_q_value[0, self.option]
        self.episode_eigen_q_values.append(self.eigen_q_value)

        """Get the intra-option policy for the current option"""
        pi = options[0, self.option]
        """Sample an action"""
        self.action = np.random.choice(pi, p=pi)
        self.action = np.argmax(pi == self.action)

        """Get also the state value function corresponding to the mixed reward signal"""
        self.evalue = evalue[0]
      else:
        """If the option is a primitve action"""
        self.action = self.option - self.nb_options
    else:
      """If we do not use eigen directions, default behaviour for the classic option-critic"""
      tensor_list = [self.local_network.options,
                     self.local_network.v,
                     self.local_network.q_val]
      options,\
      value,\
      q_value = self.sess.run(tensor_list, feed_dict=feed_dict)

      """If we included primitve options and the option taken is a primitive action"""
      if self.config.include_primitive_options and self.primitive_action:
        self.action = self.option - self.nb_options
      else:
        """Get the intra-option policy for the current option and sample an action according to it"""
        pi = options[0, self.option]
        self.action = np.random.choice(pi, p=pi)
        self.action = np.argmax(pi == self.action)

    """Get the option-value function for the external reward signal corresponding to the current option"""
    self.q_value = q_value[0, self.option]
    """Store also all the option-value functions for the external reward signal"""
    self.q_values = q_value[0]
    """Get the state value function corresponding to the external reward signal"""
    self.value = value[0]

    """Store information in buffers for stats in tensorboard"""
    self.episode_values.append(self.value)
    self.episode_q_values.append(self.q_value)
    self.episode_actions.append(self.action)

  "Do SVD decomposition on the SR matrix buffer"
  def recompute_eigenvectors_svd(self):
    """Keep track of the eigendirection before the update"""
    old_directions = copy.deepcopy(self.global_network.directions)

    """Compute the static SR_matrix"""
    self.matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
    indices = []
    states = []
    for idx in range(self.nb_states):
      s, ii, jj = self.env.get_state(idx)
      if self.env.not_wall(ii, jj):
        indices.append(idx)
        states.append(s)

    feed_dict = {self.local_network.observation: states}
    sf = self.sess.run(self.local_network.sf, feed_dict=feed_dict)
    self.matrix_sf[indices] = sf

    """Plot the SR matrix"""
    self.plot_sr_matrix()

    """Do SVD decomposition"""
    feed_dict = {self.local_network.matrix_sf: [self.matrix_sf]}
    eigenvect = self.sess.run(self.local_network.eigenvectors,
                              feed_dict=feed_dict)
    eigenvect = eigenvect[0]

    """If this is not the first time we initialize eigendirections, that map them to the closest directions, so as not to change option basis too abruptly"""
    if self.global_network.directions_init:
      self.global_network.directions = self.associate_closest_vectors(old_directions, eigenvect)
    else:
      """Otherwise just map them from the first eigenoption, taking both directions"""
      new_eigenvectors = eigenvect[self.config.first_eigenoption: (self.config.nb_options // 2) + self.config.first_eigenoption]
      self.global_network.directions = np.concatenate((new_eigenvectors, (-1) * new_eigenvectors))
      self.global_network.directions_init = True

    self.directions = self.global_network.directions

    """Track statistics in tensorboard about the change over time in directions"""
    min_similarity = np.min(
      [self.cosine_similarity(a, b) for a, b in zip(old_directions, self.directions)])
    max_similarity = np.max(
      [self.cosine_similarity(a, b) for a, b in zip(old_directions, self.directions)])
    mean_similarity = np.mean(
      [self.cosine_similarity(a, b) for a, b in zip(old_directions, self.directions)])
    self.summary = tf.Summary()
    self.summary.value.add(tag='Eigenvectors/Min similarity', simple_value=float(min_similarity))
    self.summary.value.add(tag='Eigenvectors/Max similarity', simple_value=float(max_similarity))
    self.summary.value.add(tag='Eigenvectors/Mean similarity', simple_value=float(mean_similarity))
    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()

    """Do n-step prediction for the successor representation latent"""

  def sf_prediction(self, s1):
    if len(self.episode_buffer_sf) == self.config.max_update_freq or self.done:
      """Get the successor features of the next state for which to bootstrap from"""
      feed_dict = {self.local_network.observation: [s1]}
      next_sf = self.sess.run(self.local_network.sf,
                              feed_dict=feed_dict)[0]
      bootstrap_sf = np.zeros_like(next_sf) if self.done else next_sf
      self.train_sf(bootstrap_sf)
      self.episode_buffer_sf = []

  """Map ejgen directions to the closest old directions in terms of cosine similarity, 
  so as not to change option basis too abruptly"""
  def associate_closest_vectors(self, old, new):
    to_return = copy.deepcopy(old)
    skip_list = []

    featured = new[self.config.first_eigenoption: (self.config.nb_options // 2) + self.config.first_eigenoption]
    featured = np.concatenate((featured, (-1) * featured))

    for d in featured:
      distances = []
      for old_didx, old_d in enumerate(old):
        if old_didx in skip_list:
          distances.append(-np.inf)
        else:
          distances.append(self.cosine_similarity(d, old_d))
      closest_distance_idx = np.argmax(distances)
      skip_list.append(closest_distance_idx)
      to_return[closest_distance_idx] = d
    return to_return

  """Do one n-step update for training the agent's latent successor representation space"""
  def train_sf(self, bootstrap_sf):
    rollout = np.array(self.episode_buffer_sf)
    observations = rollout[:, 0]

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
      self.sess.run([self.local_network.aux_loss, self.local_network.apply_grads_aux,
                     self.local_network.merged_summary_aux],
                    feed_dict=feed_dict)

  """Do n-step prediction on the critics and policies"""
  def train_option(self, bootstrap_value, bootstrap_value_mix):  #
    rollout = np.array(
      self.episode_buffer_option)
    observations = rollout[:, 0]
    options = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    eigen_rewards = rollout[:, 4]
    primitive_actions = rollout[:, 5]
    next_observations = rollout[:, 6]

    """Construct list of discounted returns for the entire n-step trajectory"""
    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_returns = reward_discount(rewards_plus, self.config.discount)[:-1]

    """Construct list of discounted returns using mixed reward signals for the entire n-step trajectory"""
    eigen_rewards_plus = np.asarray(eigen_rewards.tolist() + [bootstrap_value_mix])
    discounted_eigen_returns = reward_discount(eigen_rewards_plus, self.config.discount)[:-1]

    """Do an update on the option-value function critic"""
    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.options_placeholder: options,
                 }

    _, self.summaries_critic = self.sess.run([self.local_network.apply_grads_critic,
                                       self.local_network.merged_summary_critic,
                                       ], feed_dict=feed_dict)

    """Do an update on the option termination conditions"""
    feed_dict = {
      self.local_network.observation: np.stack(next_observations, axis=0),
      self.local_network.options_placeholder: options,
      self.local_network.primitive_actions_placeholder: primitive_actions
    }

    _, self.summaries_termination = self.sess.run([self.local_network.apply_grads_term,
                                     self.local_network.merged_summary_term,
                                    ], feed_dict=feed_dict)

    """Do an update on the intra-option policies"""
    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.target_eigen_return: discounted_eigen_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.options_placeholder: options,
                 self.local_network.primitive_actions_placeholder: primitive_actions
                 }

    _, self.summaries_option = self.sess.run([self.local_network.apply_grads_option,
                                       self.local_network.merged_summary_option,
                                       ], feed_dict=feed_dict)

    """Store the bootstrap target returns at the end of the trajectory"""
    self.R = discounted_returns[-1]
    self.eigen_R = discounted_eigen_returns[-1]

  """Initialize the tracker responsible for writing different stats in csv files in order to keep track of option termination statistics and such"""
  def init_tracker(self):
    csv_things = ["episode", "total_steps", "episode_steps", "reward", "term_prob", "primitive_prob"]
    nb_cols = self.nb_options + self.action_size if self.config.include_primitive_options else self.nb_options
    csv_things += ["opt_chosen" + str(ccc) for ccc in range(nb_cols)]
    csv_things += ["opt_steps" + str(ccc) for ccc in range(nb_cols)]
    with open(os.path.join(self.summary_path, "data.csv"), "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things]) + "\n")
    csv_things = ["episode", "episode_steps", "value"]
    csv_things += ["A_" + str(ccc) for ccc in range(nb_cols)]
    csv_things += ["prob_term_" + str(ccc) for ccc in range(nb_cols)]
    with open(os.path.join(self.summary_path, "q_values.csv"), "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things]) + "\n")
    csv_things = ["episode", "episode_steps"]
    csv_things += ["mean_o_len" + str(ccc) for ccc in range(nb_cols)]
    csv_things += ["max_o_len" + str(ccc) for ccc in range(nb_cols)]
    csv_things += ["min_o_len" + str(ccc) for ccc in range(nb_cols)]
    with open(os.path.join(self.summary_path, "o_lens.csv"), "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things]) + "\n")

  def cosine_similarity(self, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    res = dot_product / ((norm_a + 1e-8) * (norm_b + 1e-8))
    if np.isnan(res):
      print("NAN")
    return res

  """Track stats and add to csv file"""
  def tracker(self):
    term_prob = float(self.termination_counter) / self.episode_length * 100
    primitive_prob = self.primitive_action_counter / self.episode_length * 100
    csv_things = [self.total_steps, self.total_steps, self.episode_length, self.episode_reward,
                  round(term_prob, 1), primitive_prob] + list(self.o_tracker_chosen) + list(
      self.o_tracker_steps)

    with open(os.path.join(self.summary_path, "data.csv"), "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things]) + "\n")
    self.advantages = [q - self.value for q in self.q_values]
    csv_things = [self.total_steps, self.episode_length, self.value] + list(self.advantages) + list(self.prob_terms)
    with open(os.path.join(self.summary_path, "q_values.csv"), "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things]) + "\n")

    mean_lengths = [0 if len(o_len) == 0 else get_mode(o_len) for o_len in self.o_tracker_len]
    max_lengths = [0 if len(o_len) == 0 else np.max(o_len) for o_len in self.o_tracker_len]
    min_lengths = [0 if len(o_len) == 0 else np.min(o_len) for o_len in self.o_tracker_len]
    csv_things = [self.total_steps, self.episode_length] + list(mean_lengths) + list(max_lengths) + list(min_lengths)
    with open(os.path.join(self.summary_path, "o_lens.csv"), "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things]) + "\n")

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

  def update_episode_stats(self):
    term_prob = float(self.termination_counter) / self.episode_length * 100
    primitive_action_prob = self.primitive_action_counter / self.episode_length * 100
    self.episode_term_prob.append(term_prob)
    self.episode_primtive_action_prob.append(primitive_action_prob)
    if len(self.episode_values) != 0:
      self.episode_mean_values.append(np.mean(self.episode_values))
    if len(self.episode_q_values) != 0:
      self.episode_mean_q_values.append(np.mean(self.episode_q_values))
    if self.config.use_eigendirections and len(self.episode_eigen_q_values) != 0:
      self.episode_mean_eigen_q_values.append(np.mean(self.episode_eigen_q_values))
    if len(self.episode_oterm) != 0:
      self.episode_mean_oterms.append(get_mode(self.episode_oterm))
    if len(self.episode_options) != 0:
      self.episode_mean_options.append(get_mode(self.episode_options))
    if len(self.episode_actions) != 0:
      self.episode_mean_actions.append(get_mode(self.episode_actions))

  def write_step_summary(self, r, r_mix=None):
    # self.write_eigendirection_maps()
    self.summary = tf.Summary()
    # for sum in [self.summaries_sf, self.summaries_aux, self.summaries_aux, self.summaries_termination, self.summaries_critic]:
      # if sum is not None:
      #   self.summary_writer.add_summary(sum, self.total_steps)

    # self.summary.value.add(tag='Step/Reward', simple_value=r)
    # if self.config.use_eigendirections and not self.primitive_action and r_mix is not None:
    #   self.summary.value.add(tag='Step/EigReward', simple_value=r_mix)
    self.summary.value.add(tag='Step/Action', simple_value=self.action)
    self.summary.value.add(tag='Step/Option', simple_value=self.option)
    self.summary.value.add(tag='Step/Q', simple_value=self.q_value)
    try:
      if self.config.use_eigendirections and not self.primitive_action and self.eigen_q_value is not None:
        self.summary.value.add(tag='Step/EigenQ', simple_value=self.eigen_q_value)
        # self.summary.value.add(tag='Step/EigenV', simple_value=self.evalue)
    except:
      print("errr")
    self.summary.value.add(tag='Step/V', simple_value=self.value)
    self.summary.value.add(tag='Step/Term', simple_value=int(self.o_term))
    if self.R:
      self.summary.value.add(tag='Step/Target_Q', simple_value=self.R)
    if self.config.use_eigendirections and self.eigen_R:
      self.summary.value.add(tag='Step/Target_EigenQ', simple_value=self.eigen_R)

    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()
    # tf.logging.warning("Writing step summary....")

