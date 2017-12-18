import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph_aux, update_target_graph_sf, \
  update_target_graph_option, discount
import os
from tools.ring_buffer import RingBuffer
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

def get_mode(arr):
  if len(arr) != 0:
    u, indices = np.unique(arr, return_inverse=True)
    return u[np.argmax(np.bincount(indices))]
  else:
    return -1


class EigenOCAgent(Visualizer):
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

    self.update_local_vars_aux = update_target_graph_aux('global', self.name)
    self.update_local_vars_sf = update_target_graph_sf('global', self.name)
    self.update_local_vars_option = update_target_graph_option('global', self.name)
    self.env = game
    self.nb_states = game.nb_states
    self.sr_matrix_buffer = RingBuffer((self.config.sf_matrix_size, self.config.sf_layers[-1]))
    self.eigenvectors = np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
    self.should_consider_eigenvectors = False

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.sess = sess
      self.saver = saver
      self.episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      ms_aux = ms_sf = ms_option = None
      print("Starting worker " + str(self.thread_id))
      self.aux_episode_buffer = deque()

      while not coord.should_stop():
        if self.episode_count > self.config.episodes:
          return 0
        sess.run(self.update_local_vars_aux)
        sess.run(self.update_local_vars_sf)
        sess.run(self.update_local_vars_option)

        self.episode_buffer_sf = []
        self.episode_buffer_option = []
        self.episode_values = []
        self.episode_q_values = []
        self.episode_oterm = []
        self.episode_options = []
        self.episode_reward = 0
        self.episode_option_histogram = np.zeros(self.config.nb_options)
        d = False
        t = 0
        t_counter_sf = 0
        t_counter_option = 0

        s = self.env.reset()
        # if self.total_steps > self.config.eigen_exploration_steps:
        self.option_evaluation(s)
        while not d:
          if self.total_steps % self.config.target_update_iter_aux == 0:
            sess.run(self.update_local_vars_aux)
          if self.total_steps % self.config.target_update_iter_sf == 0:
            sess.run(self.update_local_vars_sf)

          self.policy_evaluation(s)
          s1, r, d, _ = self.env.step(self.action)
          r = np.clip(r, -1, 1)
          if d:
            s1 = s

          self.store_general_info(s, s1, self.action, r)

          if self.total_steps > self.config.observation_steps:
            t_counter_sf += 1
            if len(self.aux_episode_buffer) > self.config.observation_steps and \
                        self.total_steps % self.config.aux_update_freq == 0:
              ms_aux, aux_loss = self.train_aux()
              if self.name == "worker_0":
                print("Episode {} >> Step {} >>> AUX_loss {} ".format(self.episode_count, self.total_steps, aux_loss))
            if t_counter_sf == self.config.max_update_freq or d:
              feed_dict = {self.local_network.observation: np.stack([s1])}
              sf = sess.run(self.local_network.sf,
                            feed_dict=feed_dict)[0]
              bootstrap_sf = np.zeros_like(sf) if d else sf
              ms_sf, sf_loss = self.train_sf(bootstrap_sf)
              if self.name == "worker_0":
                print("Episode {} >> Step {} >>> SF_loss {}".format(self.episode_count, self.total_steps, sf_loss))
              self.episode_buffer_sf = []
              t_counter_sf = 0

            if self.total_steps > self.config.eigen_exploration_steps:
              t_counter_option += 1
              self.add_current_state_SR(s)

              self.store_option_info(s, s1, self.action, r)

              if t_counter_option == self.config.max_update_freq or d or (
                self.o_term and t_counter_option >= self.config.min_update_freq):
                if d:
                  R = 0
                else:
                  feed_dict = {self.local_network.observation: np.stack([s1])}
                  value, q_value = sess.run([self.local_network.v, self.local_network.q_val],
                                            feed_dict=feed_dict)
                  q_value = q_value[0, self.option]
                  value = value[0]

                  R = value if self.o_term else q_value
                # ms_option, option_loss, policy_loss, entropy_loss, critic_loss, eigen_critic_loss, term_loss = self.train_option(R)
                ms_option, option_loss, policy_loss, entropy_loss, critic_loss, term_loss = self.train_option(R)
                if self.name == "worker_0":
                  tf.logging.info("Episode {} >> Step {} >>> option_loss {}".format(self.episode_count, self.total_steps, option_loss))

                self.episode_buffer_option = []
                t_counter_option = 0

              if not d:
                if self.o_term:
                  self.option_evaluation(s)

            if self.total_steps % self.config.steps_checkpoint_interval == 0 and self.name == 'worker_0':
              self.save_model()
              # if self.mat_counter > self.config.sf_transition_matrix_size:
              #   np.save(self.matrix_path, self.matrix_sf)
              #   tf.logging.info("Saved Matrix at {}".format(self.matrix_path))

            if self.total_steps % self.config.steps_summary_interval == 0 and self.name == 'worker_0':
              self.write_step_summary(ms_sf, ms_aux, ms_option)


          s = s1
          t += 1
          self.total_steps += 1
          sess.run(self.increment_total_steps_tensor)

        if self.name == "worker_0":
          tf.logging.info("Episode {} >> Step {} >> Length: {} >>> Reward: {}".format(self.episode_count,
                                                                                              self.total_steps, t,
                                                                                           self.episode_reward))

        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(t)
        if len(self.episode_values) != 0:
          self.episode_mean_values.append(np.mean(self.episode_values))
        if len(self.episode_q_values) != 0:
          self.episode_mean_q_values.append(np.mean(self.episode_q_values))
        if len(self.episode_oterm) != 0:
          self.episode_mean_oterms.append(np.mean(self.episode_oterm))
        if len(self.episode_options) != 0:
          self.episode_mean_options.append(get_mode(self.episode_options))

        # if episode_count % self.config.eval_interval == 0 and self.total_steps != 0 and \
        #         self.name == 'worker_0':
        #   eval_reward = self.evaluate_agent(sess)
        #   self.summary.value.add(tag='Perf/EvalReward', simple_value=float(eval_reward))
        #   self.summary_writer.add_summary(self.summary, self.total_steps)
        #   self.summary_writer.flush()

        if self.episode_count % self.config.episode_checkpoint_interval == 0 and self.name == 'worker_0':
         self.save_model()

        if self.episode_count % self.config.episode_summary_interval == 0 and self.total_steps != 0 and \
                self.name == 'worker_0':
          self.write_episode_summary(ms_sf, ms_aux, ms_option)

        if self.name == 'worker_0':
          sess.run(self.increment_global_step)
          self.episode_count += 1

  def option_evaluation(self, s):
    feed_dict = {self.local_network.observation: np.stack([s])}
    self.option = self.sess.run([self.local_network.current_option], feed_dict=feed_dict)[0][0]
    self.episode_options.append(self.option)
    self.episode_option_histogram[self.option] += 1

  def policy_evaluation(self, s):
    if self.total_steps > self.config.eigen_exploration_steps:
      feed_dict = {self.local_network.observation: np.stack([s])}
      options, value, q_value, o_term = self.sess.run([self.local_network.options, self.local_network.v,
                                                       self.local_network.q_val, self.local_network.termination],
                                                      feed_dict=feed_dict)
      self.o_term = o_term[0, self.option] > np.random.uniform()
      self.q_value = q_value[0, self.option]
      self.value = value[0]
      pi = options[0, self.option]
      self.action = np.random.choice(pi, p=pi)
      self.action = np.argmax(pi == self.action)
    else:
      self.action = np.random.choice(range(self.action_size))

  def store_general_info(self, s, s1, a, r):
    self.episode_buffer_sf.append([s, s1, a])
    if len(self.aux_episode_buffer) == self.config.memory_size:
      self.aux_episode_buffer.popleft()
    self.aux_episode_buffer.append([s, s1, a])

  def store_option_info(self, s, s1, a, r):
    if self.sr_matrix_buffer.full:
      self.recompute_eigenvectors()

    # if self.should_consider_eigenvectors:
    #   feed_dict = {self.local_network.observation: np.stack([s, s1])}
    #   fi = self.sess.run(self.local_network.fi,
    #                      feed_dict=feed_dict)
    #   eigen_r = self.cosine_similarity((fi[1] - fi[0]), self.eigenvectors[self.option])
    #   r_i = self.config.alpha_r * eigen_r + (1 - self.config.alpha_r) * r
    # else:
    #   r_i = r
    r_i = r
    self.episode_buffer_option.append(
      [s, self.option, self.action, r, r_i])
    self.episode_values.append(self.value)
    self.episode_q_values.append(self.q_value)
    self.episode_reward += r
    self.episode_oterm.append(self.o_term)

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps),
               global_step=self.global_step)
    tf.logging.info("Saved Model at {}".format(self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps)))

  def write_step_summary(self, ms_sf, ms_aux, ms_option):
    if ms_sf is not None:
      self.summary_writer.add_summary(ms_sf, self.total_steps)
    if ms_aux is not None:
      self.summary_writer.add_summary(ms_aux, self.total_steps)
    if ms_option is not None:
      self.summary_writer.add_summary(ms_option, self.total_steps)

    self.summary_writer.flush()

  def write_episode_summary(self, ms_sf, ms_aux, ms_option):
    if len(self.episode_rewards) != 0:
      last_reward = self.episode_rewards[-1]
      self.summary.value.add(tag='Perf/Reward', simple_value=float(last_reward))
    if len(self.episode_lengths) != 0:
      last_length = self.episode_lengths[-1]
      self.summary.value.add(tag='Perf/Length', simple_value=float(last_length))
    if len(self.episode_mean_values) != 0:
      last_mean_value = self.episode_mean_values[-1]
      self.summary.value.add(tag='Perf/Value', simple_value=float(last_mean_value))
    if len(self.episode_mean_q_values) != 0:
      last_mean_q_value = self.episode_mean_q_values[-1]
      self.summary.value.add(tag='Perf/QValue', simple_value=float(last_mean_q_value))
    if len(self.episode_mean_oterms) != 0:
      last_mean_oterm = self.episode_mean_oterms[-1]
      self.summary.value.add(tag='Perf/Oterm', simple_value=float(last_mean_oterm))
    if len(self.episode_mean_options) != 0:
      last_frequent_option = self.episode_mean_options[-1]
      self.summary.value.add(tag='Perf/FreqOptions', simple_value=last_frequent_option)

    if len(self.episode_options) != 0:
      counts, bin_edges = np.histogram(self.episode_options,
                                       bins=list(range(self.config.nb_options)) + [self.config.nb_options])

      hist = tf.HistogramProto(min=np.min(self.episode_options),
                               max=np.max(self.episode_options),
                               num=len(self.episode_options),
                               sum=np.sum(self.episode_options),
                               sum_squares=np.sum([e ** 2 for e in self.episode_options])
                               )
      bin_edges = bin_edges[1:]
      # Add bin edges and counts
      for edge in bin_edges:
        hist.bucket_limit.append(edge)
      for c in counts:
        hist.bucket.append(c)

      self.summary.value.add(tag='Perf/OptionsHist', histo=hist)
      self.summary_writer.add_summary(self.summary, self.total_steps)

    self.write_step_summary(ms_sf, ms_aux, ms_option)
    self.summary_writer.flush()

  def add_current_state_SR(self, s):
    feed_dict = {self.local_network.observation: np.stack([s])}
    sf = self.sess.run(self.local_network.sf,
                  feed_dict=feed_dict)[0]
    self.sr_matrix_buffer.append(sf)

  def recompute_eigenvectors(self):
    self.should_consider_eigenvectors = False
    feed_dict = {self.local_network.matrix_sf: self.sr_matrix_buffer.get()}
    eigenval, eigenvect = self.sess.run([self.local_network.eigenvalues, self.local_network.eigenvectors],
                       feed_dict=feed_dict)
    # u, s, v = np.linalg.svd(self.sr_matrix_buffer.get(), full_matrices=False)
    eigenvalues = eigenval[1:self.config.nb_options]
    self.eigenvectors = eigenvect[1:self.config.nb_options]

  def cosine_similarity(self, next_sf, evect):
    state_dif_norm = np.linalg.norm(next_sf)
    state_dif_normalized = next_sf / (state_dif_norm + 1e-8)
    # evect_norm = np.linalg.norm(evect)
    # evect_normalized = evect / (evect_norm + 1e-8)
    res = np.dot(state_dif_normalized, evect)
    return res

  def train_sf(self, bootstrap_sf):
    rollout = np.array(self.episode_buffer_sf)
    observations = rollout[:, 0]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0)}
    fi = self.sess.run(self.local_network.fi,
                       feed_dict=feed_dict)

    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0)}  # ,

    _, ms, sf_loss = \
      self.sess.run([self.local_network.apply_grads_sf,
                     self.local_network.merged_summary_sf,
                     self.local_network.sf_loss],
                    feed_dict=feed_dict)

    return ms, sf_loss

  def train_aux(self):
    minibatch = random.sample(self.aux_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.actions_placeholder: actions}

    aux_loss, _, ms = \
      self.sess.run([self.local_network.aux_loss, self.local_network.apply_grads_aux,
                     self.local_network.merged_summary_aux],
                    feed_dict=feed_dict)
    return ms, aux_loss

  def train_option(self, bootstrap_value):
    rollout = np.array(self.episode_buffer_option) #s, self.option, self.action, r, r_i
    observations = rollout[:, 0]
    options = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    eigen_rewards = rollout[:, 4]

    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_returns = discount(rewards_plus, self.config.discount)[:-1]

    # eigen_rewards_plus = np.asarray(eigen_rewards.tolist() + [bootstrap_value])
    # discounted_eigen_returns = discount(eigen_rewards_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_return: discounted_returns,
                 # self.local_network.target_eigen_return: discounted_eigen_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.options_placeholder: options}

    # _, ms_option, option_loss, policy_loss, entropy_loss, critic_loss, eigen_critic_loss, term_loss = \
    _, ms_option, option_loss, policy_loss, entropy_loss, critic_loss, term_loss = \
      self.sess.run([self.local_network.apply_grads_option,
                self.local_network.merged_summary_option,
                self.local_network.option_loss,
                self.local_network.policy_loss,
                self.local_network.entropy_loss,
                self.local_network.critic_loss,
                # self.local_network.eigen_critic_loss,
                self.local_network.term_loss],
               feed_dict=feed_dict)
    # return ms_option, option_loss, policy_loss, entropy_loss, critic_loss, eigen_critic_loss, term_loss
    return ms_option, option_loss, policy_loss, entropy_loss, critic_loss, term_loss
