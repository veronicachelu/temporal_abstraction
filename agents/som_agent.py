import numpy as np
import tensorflow as tf
from tools.agent_utils import get_mode, update_target_graph_aux, update_target_graph_sf, \
  update_target_graph_option, discount, reward_discount, set_image, make_gif
import os

from agents.base_agent import BaseAgent
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
from collections import deque
import seaborn as sns
sns.set()
import random
# import matplotlib.pyplot as plt
import copy
from tools.agent_utils import update_target_graph_reward
FLAGS = tf.app.flags.FLAGS
import csv
from tools.timer import Timer


class SomAgent(BaseAgent):
  def __init__(self, game, thread_id, global_step, config, global_network, barrier):
    super(SomAgent, self).__init__(game, thread_id, global_step, config, global_network)
    self.update_local_vars_reward = update_target_graph_reward('global', self.name)
    # self.stats_path = os.path.join(self.summary_path, 'stats')
    # tf.gfile.MakeDirs(self.stats_path)
    self.barrier = barrier

  def init_play(self, sess, saver):
    self.sess = sess
    self.saver = saver
    self.episode_count = sess.run(self.global_step)

    if self.config.move_goal_nb_of_ep and self.config.multi_task:
      self.goal_position = self.env.set_goal(self.episode_count, self.config.move_goal_nb_of_ep)

    self.total_steps = sess.run(self.total_steps_tensor)
    self.eigen_q_value = None
    self.evalue = None
    tf.logging.info("Starting worker " + str(self.thread_id))
    self.aux_episode_buffer = deque()
    self.reward_pred_episode_buffer = deque()
    self.reward_i_pred_episode_buffer = deque()
    self.ms_aux = self.ms_sf = self.ms_option = self.ms_reward = self.ms_reward_i = None
    # self.stats_options = np.zeros((self.nb_states, self.nb_options + self.action_size))

  def init_episode(self):
    self.episode_buffer_sf = []
    self.episode_values = []
    self.episode_q_values = []
    self.episode_eigen_q_values = []
    self.episode_oterm = []
    self.episode_options = []
    self.episode_actions = []
    # self.episode_options_lengths = [[] for o in range(self.config.nb_options)]
    self.episode_reward = 0
    # self.episode_option_histogram = np.zeros(self.config.nb_options)
    self.done = False
    self.episode_len = 0
    self.sf_counter = 0
    self.R = 0
    self.eigen_R = 0

    # if self.config.decrease_option_prob:
    #   self.sess.run(self.local_network.decrease_prob_of_random_option)
    # self.stats_options = np.zeros((self.nb_states, self.nb_options + self.action_size))
    # self.ms_aux = self.ms_sf = self.ms_reward = self.ms_option = None

  def SF_option_prediction(self, s, o, s1, o1, a, primitive):
    self.episode_buffer_sf.append((s, o, s1, o1, a, primitive))
    self.sf_counter += 1
    if self.config.eigen and (self.sf_counter == self.config.max_update_freq or self.done or (
            self.o_term and self.sf_counter >= self.config.min_update_freq)):
      feed_dict = {self.local_network.observation: [s1], self.local_network.options_placeholder: [o1]}
      sf_o = self.sess.run(self.local_network.sf_o,
                    feed_dict=feed_dict)[0]

      bootstrap_sf = np.zeros_like(sf_o) if self.done else sf_o
      self.ms_sf, self.sf_loss = self.train_sf(bootstrap_sf)
      self.ms_option, self.option_loss = self.train_option()

      self.episode_buffer_sf = []
      self.sf_counter = 0

  def next_frame_prediction(self):
    if len(self.aux_episode_buffer) > self.config.observation_steps and \
                self.total_steps % self.config.aux_update_freq == 0:
      self.ms_aux, self.aux_loss = self.train_aux()

  def store_reward_info(self, s, o, a, s1, r, primitive):
    self.episode_reward += r
    if self.config.eigen and not self.primitive_action:
      feed_dict = {self.local_network.observation: np.stack([s, s1])}
      fi = self.sess.run(self.local_network.fi,
                         feed_dict=feed_dict)
      eigen_r = self.cosine_similarity((fi[1] - fi[0]), self.directions[o])
      r_i = self.config.alpha_r * eigen_r + (1 - self.config.alpha_r) * r
      if len(self.reward_i_pred_episode_buffer) == self.config.memory_size:
        self.reward_i_pred_episode_buffer.popleft()
      self.reward_i_pred_episode_buffer.append([s, s1, r_i, o, a])
    else:
      r_i = r

    if len(self.reward_pred_episode_buffer) == self.config.memory_size:
      self.reward_pred_episode_buffer.popleft()

    self.reward_pred_episode_buffer.append([s, s1, r])


  def reward_prediction(self):
    if len(self.reward_pred_episode_buffer) > self.config.observation_steps and \
                self.total_steps % self.config.reward_update_freq == 0:
      self.ms_reward, self.r_loss = self.train_reward_prediction()
    if len(self.reward_i_pred_episode_buffer) > self.config.observation_steps and \
                self.total_steps % self.config.reward_update_freq == 0:
      self.ms_reward_i, self.r_i_loss = self.train_reward_i_prediction()


  def sync_threads(self, force=False):
    if force:
      self.sess.run(self.update_local_vars_aux)
      self.sess.run(self.update_local_vars_sf)
      self.sess.run(self.update_local_vars_option)
      self.sess.run(self.update_local_vars_reward)
    else:
      if self.total_steps % self.config.target_update_iter_aux == 0:
        self.sess.run(self.update_local_vars_aux)
      if self.total_steps % self.config.target_update_iter_reward == 0:
        self.sess.run(self.update_local_vars_reward)
      if self.total_steps % self.config.target_update_iter_sf == 0:
        self.sess.run(self.update_local_vars_sf)
      if self.total_steps % self.config.target_update_iter_option == 0:
        self.sess.run(self.update_local_vars_option)

  def play(self, sess, coord, saver):
    _t = {'recompute_eigenvectors_classic': Timer(), "next_frame_prediction": Timer(), 'reward_prediction': Timer(), 'SF_option_prediction': Timer()}
    with sess.as_default(), sess.graph.as_default():
      self.init_play(sess, saver)

      with coord.stop_on_exception():
        while not coord.should_stop():
          if (self.config.steps != -1 and \
              (self.total_steps > self.config.steps and self.name == "worker_0")) or \
              (self.episode_count > len(self.config.goal_locations) * self.config.move_goal_nb_of_ep and
                   self.name == "worker_0"):
            coord.request_stop()
            return 0

          self.sync_threads()

          if self.name == "worker_0" and self.episode_count > 0:
            # plotting = self.episode_count % self.config.plot_every == 0
            plotting = False
            if self.config.logging:
              _t['recompute_eigenvectors_classic'].tic()
            self.recompute_eigenvectors_classic(False)
            if self.config.logging:
              _t['recompute_eigenvectors_classic'].toc()

          self.load_directions()
          self.init_episode()

          s = self.env.reset()
          self.option_evaluation(s)
          while not self.done:
            self.sync_threads()
            self.policy_evaluation(s)

            s1, r, self.done, s1_idx = self.env.step(self.action)

            r = np.clip(r, -1, 1)
            if self.done:
              s1 = s

            self.store_general_info(s, s1, self.action, r)
            self.store_reward_info(s, self.option, self.action, s1, r, self.primitive_action)
            # self.log_timestep()

            if self.total_steps > self.config.observation_steps:
              if self.config.logging:
                _t['next_frame_prediction'].tic()
              self.next_frame_prediction()
              if self.config.logging:
                _t['next_frame_prediction'].toc()
                _t['reward_prediction'].tic()
              self.reward_prediction()
              if self.config.logging:
                _t['reward_prediction'].toc()
              self.old_option = self.option
              self.old_primitive_action = self.primitive_action

              if not self.done and (self.o_term or self.primitive_action):
                # if not self.primitive_action:
                #   self.episode_options_lengths[self.option][-1] = self.episode_len - \
                #                                                   self.episode_options_lengths[self.option][-1]
                self.option_evaluation(s1)

              if self.config.logging:
                _t['SF_option_prediction'].tic()
              self.SF_option_prediction(s, self.old_option, s1, self.option, self.action, self.old_primitive_action)
              if self.config.logging:
                _t['SF_option_prediction'].tic()

              if self.total_steps % self.config.steps_checkpoint_interval == 0 and self.name == 'worker_0':
                self.save_model()

              if self.total_steps % self.config.steps_summary_interval == 0 and self.name == 'worker_0':
                self.write_step_summary(r)

            s = s1
            self.episode_len += 1
            self.total_steps += 1
            sess.run(self.increment_total_steps_tensor)

          # self.log_episode()
          self.update_episode_stats()

          # if self.episode_count % self.config.episode_eval_interval == 0 and \
          #         self.name == 'worker_0' and self.episode_count != 0:
          #   tf.logging.info("Evaluating agent....")
          #   eval_episodes_won, mean_ep_length = self.evaluate_agent()
          #   self.write_eval_summary(eval_episodes_won, mean_ep_length)

          if self.episode_count % self.config.move_goal_nb_of_ep == 0 and \
                  self.episode_count != 0:
            tf.logging.info("Moving GOAL....")
            self.barrier.wait()
            self.goal_position = self.env.set_goal(self.episode_count, self.config.move_goal_nb_of_ep)

          if self.episode_count % self.config.episode_checkpoint_interval == 0 and self.name == 'worker_0' and \
                  self.episode_count != 0:
            self.save_model()

          if self.episode_count % self.config.episode_summary_interval == 0 and self.total_steps != 0 and \
                  self.name == 'worker_0' and self.episode_count != 0:
            self.write_episode_summary(self.episode_reward)

          if self.name == 'worker_0':
            sess.run(self.increment_global_step)
            if self.config.logging:
              tf.logging.info('recompute_eigenvectors_classic time is %f' % _t['recompute_eigenvectors_classic'].average_time)
              tf.logging.info('next_frame_prediction time is %f' % _t['next_frame_prediction'].average_time)
              tf.logging.info('reward_prediction time is %f' % _t['reward_prediction'].average_time)
              tf.logging.info('SF_option_prediction time is %f' % _t['SF_option_prediction'].average_time)
          self.episode_count += 1

  def option_evaluation(self, s):
    feed_dict = {self.local_network.observation: np.stack([s])}
    self.option, self.primitive_action = self.sess.run(
      [self.local_network.current_option, self.local_network.primitive_action], feed_dict=feed_dict)
    self.option, self.primitive_action = self.option[0], self.primitive_action[0]

    # self.stats_options[s_idx][self.option] += 1

    # self.episode_options.append(self.option)
    # if not self.primitive_action:
    #   self.episode_options_lengths[self.option].append(self.episode_len)

  def policy_evaluation(self, s):
    if self.total_steps > self.config.eigen_exploration_steps:
      feed_dict = {self.local_network.observation: np.stack([s])}

      tensor_list = [self.local_network.options, self.local_network.v, self.local_network.q_val,
                self.local_network.termination]
      options, value, q_value, o_term = self.sess.run(tensor_list, feed_dict=feed_dict)
      if not self.primitive_action or not self.config.include_primitive_options:
        pi = options[0, self.option]
        self.action = np.random.choice(pi, p=pi)
        self.action = np.argmax(pi==self.action)
        self.o_term = o_term[0, self.option] > np.random.uniform()
      else:
        self.action = self.option - self.nb_options
        self.o_term = True
      self.q_value = q_value[0, self.option]
      self.value = value[0]
    else:
      self.action = np.random.choice(range(self.action_size))
    self.episode_actions.append(self.action)

  def store_general_info(self, s, s1, a, r):
    # if self.config.eigen:
    #   self.episode_buffer_sf.append([s, s1, a, self.option])
    if len(self.aux_episode_buffer) == self.config.memory_size:
      self.aux_episode_buffer.popleft()

    self.aux_episode_buffer.append([s, s1, a])

  # def recompute_eigenvectors_classic(self, plotting=False):
  #   if self.config.eigen:
  #     self.new_eigenvectors = copy.deepcopy(self.global_network.directions)
  #     # matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
  #     states = []
  #     for idx in range(self.nb_states):
  #       s, ii, jj = self.env.fake_get_state(idx)
  #       if self.env.not_wall(ii, jj):
  #         states.append(s)
  #
  #
  #     feed_dict = {self.local_network.observation: states}
  #     sfs = self.sess.run(self.local_network.sf, feed_dict=feed_dict)
  #
  #     def move_option(sf):
  #       sf = sf[:self.nb_options]
  #       sf_norm = np.linalg.norm(sf, axis=1, keepdims=True)
  #       sf_normalized = sf / (sf_norm + 1e-8)
  #       # sf_normalized = tf.nn.l2_normalize(sf, axis=1)
  #       self.new_eigenvectors = self.config.tau * sf_normalized + (1 - self.config.tau) * self.new_eigenvectors
  #       new_eigenvectors_norm = np.linalg.norm(self.new_eigenvectors, axis=1, keepdims=True)
  #       self.new_eigenvectors = self.new_eigenvectors / (new_eigenvectors_norm + 1e-8)
  #
  #     for sf in sfs:
  #       move_option(sf)
  #
  #     if plotting:
  #       # self.plot_sr_vectors(sfs, "sr_stats")
  #       self.plot_sr_matrix(sfs, "sr_stats")
  #
  #     min_similarity = np.min(
  #       [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, self.new_eigenvectors)])
  #     max_similarity = np.max(
  #       [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, self.new_eigenvectors)])
  #     mean_similarity = np.mean(
  #       [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, self.new_eigenvectors)])
  #     self.summary = tf.Summary()
  #     self.summary.value.add(tag='Eigenvectors/Min similarity', simple_value=float(min_similarity))
  #     self.summary.value.add(tag='Eigenvectors/Max similarity', simple_value=float(max_similarity))
  #     self.summary.value.add(tag='Eigenvectors/Mean similarity', simple_value=float(mean_similarity))
  #     self.summary_writer.add_summary(self.summary, self.episode_count)
  #     self.summary_writer.flush()
  #     self.global_network.directions = self.new_eigenvectors
  #     self.directions = self.global_network.directions
  #
  #     if plotting:
  #       self.plot_basis_functions(self.directions, "sr_stats")

  def recompute_eigenvectors_classic(self, plotting=False):
    if self.config.eigen:
      # self.new_eigenvectors = copy.deepcopy(self.global_network.directions)
      # matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
      states = []
      for idx in range(self.nb_states):
        s, ii, jj = self.env.fake_get_state(idx)
        if self.env.not_wall(ii, jj):
          states.append(s)

      feed_dict = {self.local_network.observation: states}
      sfs = self.sess.run(self.local_network.sf, feed_dict=feed_dict)

      sfs = np.transpose(sfs, [1, 0, 2])
      sfs = sfs[:self.nb_options]

      feed_dict = {self.local_network.matrix_sf: sfs}
      eigenvect = self.sess.run(self.local_network.eigenvectors,
                                feed_dict=feed_dict)
      new_eigenvectors = eigenvect[:, 0]

      min_similarity = np.min(
        [np.min(self.cosine_similarity_option(self.global_network.directions[:, i], new_eigenvectors[:, i])) for i in range(self.config.sf_layers[-1])])
      max_similarity = np.max(
        [np.max(self.cosine_similarity_option(self.global_network.directions[:, i], new_eigenvectors[:, i])) for i in range(self.config.sf_layers[-1])])
      mean_similarity = np.mean(
        [np.mean(self.cosine_similarity_option(self.global_network.directions[:, i], new_eigenvectors[:, i])) for i in range(self.config.sf_layers[-1])])
      self.summary = tf.Summary()
      self.summary.value.add(tag='Eigenvectors/Min similarity', simple_value=float(min_similarity))
      self.summary.value.add(tag='Eigenvectors/Max similarity', simple_value=float(max_similarity))
      self.summary.value.add(tag='Eigenvectors/Mean similarity', simple_value=float(mean_similarity))
      self.summary_writer.add_summary(self.summary, self.episode_count)
      self.summary_writer.flush()
      # tf.logging.warning("Min cosine similarity between old eigenvectors and recomputed onesis {}".format(min_similarity))
      self.global_network.directions = new_eigenvectors
      self.directions = self.global_network.directions


  def train_sf(self, bootstrap_sf):
    rollout = np.array(self.episode_buffer_sf)

    observations = rollout[:, 0]
    options = rollout[:, 1]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0)}
    fi = self.sess.run(self.local_network.fi,
                       feed_dict=feed_dict)
    # fi = fi[np.arange(len(observations)), np.stack(options, axis=0)]

    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.options_placeholder: np.stack(options, axis=0)}  # ,

    _, ms, sf_loss, self.sf_td_error = \
      self.sess.run([self.local_network.apply_grads_sf,
                     self.local_network.merged_summary_sf,
                     self.local_network.sf_loss,
                     self.local_network.sf_td_error],
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

  def train_reward_prediction(self):
    minibatch = random.sample(self.reward_pred_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    r = rollout[:, 2]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.target_r: r}


    r_loss, _, ms_r = \
      self.sess.run([self.local_network.reward_loss,
                     self.local_network.apply_grads_reward,
                     self.local_network.merged_summary_reward],
                    feed_dict=feed_dict)

    return ms_r, r_loss,

  def train_reward_i_prediction(self):
    minibatch = random.sample(self.reward_i_pred_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    r_i = rollout[:, 2]
    o = rollout[:, 3]
    a = rollout[:, 4]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.options_placeholder: o,
                 self.local_network.actions_placeholder: a,
                 self.local_network.target_r_i: r_i}

    r_i_loss, _, ms_r_i = \
      self.sess.run([self.local_network.reward_i_loss,
                     self.local_network.apply_grads_reward_i,
                     self.local_network.merged_summary_reward_i],
                    feed_dict=feed_dict)

    return ms_r_i, r_i_loss

  def train_option(self):
    rollout = np.array(self.episode_buffer_sf)  # s, self.option, self.action, r, r_i
    observations = rollout[:, 0]
    options = rollout[:, 1]
    next_observations = rollout[:, 2]
    next_options = rollout[:, 3]
    actions = rollout[:, 4]
    primitive = rollout[:, 5]

    notprimitve = list(np.logical_not(primitive))
    observations_not_primitive = observations[notprimitve]
    if len(observations_not_primitive) == 0:
      option_loss = ms_option = None
    else:
      options_not_primitive = options[notprimitve]
      actions_not_primitive = actions[notprimitve]
      sf_td_error_not_primitive = self.sf_td_error[notprimitve]

      # if np.any(np.array(options_not_primitive) >= 4):
      #   print("ERROR option!!!!!!!!!")
      feed_dict = {self.local_network.sf_td_error_target: sf_td_error_not_primitive,
                   self.local_network.observation: np.stack(observations_not_primitive, axis=0),
                   self.local_network.options_placeholder: options_not_primitive,
                   self.local_network.actions_placeholder: actions_not_primitive}

      _, option_loss, ms_option = self.sess.run([
                self.local_network.apply_grads_option,
                self.local_network.option_loss,
                self.local_network.merged_summary_option,
                ], feed_dict=feed_dict)
    return ms_option, option_loss

  def evaluate_agent(self):
    episodes_won = 0
    episode_lengths = []
    for i in range(self.config.nb_test_ep):
      episode_reward = 0
      s = self.env.reset()
      feed_dict = {self.local_network.observation: np.stack([s])}
      option, primitive_action = self.sess.run([self.local_network.max_options, self.local_network.primitive_action],
                                               feed_dict=feed_dict)
      option, primitive_action = option[0], primitive_action[0]
      primitive_action = option >= self.config.nb_options
      d = False
      episode_length = 0
      # if i == 0:
      #   episode_frames = []
      while not d:
        feed_dict = {self.local_network.observation: np.stack([s])}
        options, o_term = self.sess.run([self.local_network.options, self.local_network.termination],
                                        feed_dict=feed_dict)

        if primitive_action:
          action = option - self.nb_options
          o_term = True
        else:
          pi = options[0, option]
          action = np.random.choice(pi, p=pi)
          action = np.argmax(pi == action)
          o_term = o_term[0, option] > np.random.uniform()

        # if i == 0 and self.episode_count > 500:
        #   episode_frames.append(set_image(s, option, action, episode_length, primitive_action))
        s1, r, d, _ = self.env.step(action)

        r = np.clip(r, -1, 1)
        episode_reward += r
        episode_length += 1

        if not d and (o_term or primitive_action):
          feed_dict = {self.local_network.observation: np.stack([s1])}
          option, primitive_action = self.sess.run(
            [self.local_network.max_options, self.local_network.primitive_action], feed_dict=feed_dict)
          option, primitive_action = option[0], primitive_action[0]
          primitive_action = option >= self.config.nb_options
        s = s1
        if episode_length > self.config.max_length_eval:
          break

        # if i == 0 and self.episode_count > 500:
        #   images = np.array(episode_frames)
        #   make_gif(images[:100], os.path.join(self.test_path, 'eval_episode_{}.gif'.format(self.episode_count)),
        #            duration=len(images[:100]) * 0.1, true_image=True)

      episodes_won += episode_reward
      episode_lengths.append(episode_length)

    return episodes_won, np.mean(episode_lengths)

  def eval(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.sess = sess
      self.saver = saver
      self.episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      tf.logging.info("Starting eval agent")
      ep_rewards = []
      ep_lengths = []
      # episode_frames = []
      for i in range(self.config.nb_test_ep):
        episode_reward = 0
        s = self.env.reset()
        feed_dict = {self.local_network.observation: np.stack([s])}
        option, primitive_action = self.sess.run(
          [self.local_network.max_options, self.local_network.primitive_action], feed_dict=feed_dict)
        option, primitive_action = option[0], primitive_action[0]
        primitive_action = option >= self.config.nb_options
        d = False
        episode_length = 0
        while not d:
          feed_dict = {self.local_network.observation: np.stack([s])}
          options, o_term = self.sess.run([self.local_network.options, self.local_network.termination],
                                          feed_dict=feed_dict)

          if primitive_action:
            action = option - self.nb_options
            o_term = True
          else:
            pi = options[0, option]
            action = np.random.choice(pi, p=pi)
            action = np.argmax(pi == action)
            o_term = o_term[0, option] > np.random.uniform()

          # episode_frames.append(set_image(s, option, action, episode_length, primitive_action))
          s1, r, d, _ = self.env.step(action)

          r = np.clip(r, -1, 1)
          episode_reward += r
          episode_length += 1

          if not d and (o_term or primitive_action):
            feed_dict = {self.local_network.observation: np.stack([s1])}
            option, primitive_action = self.sess.run(
              [self.local_network.max_options, self.local_network.primitive_action], feed_dict=feed_dict)
            option, primitive_action = option[0], primitive_action[0]
            primitive_action = option >= self.config.nb_options
          s = s1
          if episode_length > self.config.max_length_eval:
            break

        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_length)
        tf.logging.info("Ep {} finished in {} steps with reward {}".format(i, episode_length, episode_reward))
      # images = np.array(episode_frames)
      # make_gif(images, os.path.join(self.test_path, 'test_episodes.gif'),
      #          duration=len(images) * 1.0, true_image=True)
      tf.logging.info("Won {} episodes of {}".format(ep_rewards.count(1), self.config.nb_test_ep))

  def write_step_summary(self, r):
    self.summary = tf.Summary()
    if self.ms_sf is not None:
      self.summary_writer.add_summary(self.ms_sf, self.total_steps)
    if self.ms_aux is not None:
      self.summary_writer.add_summary(self.ms_aux, self.total_steps)
    if self.ms_reward is not None:
      self.summary_writer.add_summary(self.ms_reward, self.total_steps)
    if self.ms_reward_i is not None:
      self.summary_writer.add_summary(self.ms_reward_i, self.total_steps)
    if self.ms_option is not None:
      self.summary_writer.add_summary(self.ms_option, self.total_steps)

    if self.total_steps > self.config.eigen_exploration_steps:
      self.summary.value.add(tag='Step/Reward', simple_value=r)
      self.summary.value.add(tag='Step/Action', simple_value=self.action)
      self.summary.value.add(tag='Step/Option', simple_value=self.option)
      self.summary.value.add(tag='Step/Q', simple_value=self.q_value)
      self.summary.value.add(tag='Step/V', simple_value=self.value)
      self.summary.value.add(tag='Step/Term', simple_value=int(self.o_term))

    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()
    # tf.logging.warning("Writing step summary....")

  def write_episode_summary(self, r):
    # self.write_episode_summary_stats()
    self.summary = tf.Summary()
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
    if self.config.eigen and len(self.episode_mean_eigen_q_values) != 0:
      last_mean_eigen_q_value = self.episode_mean_eigen_q_values[-1]
    if len(self.episode_mean_oterms) != 0:
      last_mean_oterm = self.episode_mean_oterms[-1]
      self.summary.value.add(tag='Perf/Oterm', simple_value=float(last_mean_oterm))
    if len(self.episode_mean_options) != 0:
      last_frequent_option = self.episode_mean_options[-1]
      self.summary.value.add(tag='Perf/FreqOptions', simple_value=last_frequent_option)
    if len(self.episode_mean_options) != 0:
      last_frequent_action = self.episode_mean_actions[-1]
      self.summary.value.add(tag='Perf/FreqActions', simple_value=last_frequent_action)
    for op in range(self.config.nb_options):
      self.summary.value.add(tag='Perf/Option_length_{}'.format(op), simple_value=self.episode_mean_options_lengths[op])

    self.summary_writer.add_summary(self.summary, self.episode_count)
    self.summary_writer.flush()
    self.write_step_summary(r)

  def update_episode_stats(self):
    self.episode_rewards.append(self.episode_reward)
    self.episode_lengths.append(self.episode_len)
    if len(self.episode_values) != 0:
      self.episode_mean_values.append(np.mean(self.episode_values))
    if len(self.episode_q_values) != 0:
      self.episode_mean_q_values.append(np.mean(self.episode_q_values))
    if len(self.episode_oterm) != 0:
      self.episode_mean_oterms.append(get_mode(self.episode_oterm))
    if len(self.episode_options) != 0:
      self.episode_mean_options.append(get_mode(self.episode_options))
    if len(self.episode_actions) != 0:
      self.episode_mean_actions.append(get_mode(self.episode_actions))
    # for op, option_lengths in enumerate(self.episode_options_lengths):
    #   if len(option_lengths) != 0:
    #     self.episode_mean_options_lengths[op] = np.mean(option_lengths)

  def write_episode_summary_stats(self):
    # with open(os.path.join(self.stats_path, 'summary_stats.csv'), 'w', newline='') as csvfile:
    #   fieldnames = ['State', 'Option_0', 'Option_1', 'Option_2', 'Option_3',
    #                 'Option_4', 'Option_5', 'Option_6', 'Option_7']
    #   writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #   writer.writeheader()
      # total_timesteps_in_state = np.sum(self.stats_options, axis=1)[..., None]
      # stats_options = self.stats_options / (total_timesteps_in_state + 1e-12)
    most_chosen_options = np.argmax(self.stats_options, axis=1)
    #   for s in range(self.nb_states):
    #     writer.writerow({'State': str(s), 'Option_0': self.stats_options[s, 0], 'Option_1': self.stats_options[s, 1],
    #                      'Option_2': self.stats_options[s, 2], 'Option_3': self.stats_options[s, 3],
    #                      'Option_4': self.stats_options[s, 4], 'Option_5': self.stats_options[s, 5],
    #                      'Option_6': self.stats_options[s, 6], 'Option_7': self.stats_options[s, 7]})
    #
    # with open('summary_stats.csv', 'w', newline='') as csvfile:
    #   spamwriter = csv.writer(csvfile, delimiter=' ',
    #                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #   spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #   spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

    plt.clf()
    option_colors = [(0.1, 0.2, 0.5, 0.4), (0.1, 0.2, 0.5, 0.6), (0.1, 0.2, 0.5, 0.8), (0.1, 0.2, 0.5, 1),
                     (0.5, 0.2, 0.1, 0.4), (0.5, 0.2, 0.1, 0.6), (0.5, 0.2, 0.1, 0.8), (0.5, 0.2, 0.1, 1)]
    for idx in range(self.nb_states):
      dx = 0
      dy = 0
      o = most_chosen_options[idx]
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
      plt.gca().add_patch(
        patches.Rectangle(
          (j, self.config.input_size[0] - i - 1),  # (x,y)
          1.0,  # width
          1.0,  # height
          facecolor=option_colors[o],
        )
      )

      x, y = self.env.get_state_xy(idx)
      states = [] #up, right, down, leftƒpo
      if x - 1 > 0:
        states.append((x-1, y))
      if y + 1 < self.config.input_size[1]:
        states.append((x, y + 1))
      if x + 1 < self.config.input_size[0]:
        states.append((x + 1, y))
      if y - 1 > 0:
        states.append((x, y - 1))

      if o >= self.nb_options:
        a = o - self.nb_options
      else:
        state_idxs = [self.env.get_state_index(x, y) for x, y in states]
        possible_next_states = [self.env.fake_get_state(idx)[0] for idx in state_idxs]

        feed_dict = {self.local_network.observation: np.stack([s] + possible_next_states)}
        fis = self.sess.run(self.local_network.fi, feed_dict=feed_dict)
        fi_s = fis[0]

        fi_diffs = fi_s - fis[1:]
        cosine_sims = [self.cosine_similarity(d, self.directions[o]) for d in fi_diffs]
        a = np.argmax(cosine_sims)

      if a  == 0:  # up
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

    plt.savefig(os.path.join(self.stats_path,  "Option_map.png"))
    plt.close()

  def cosine_similarity_option(self, next_sf, evect):
    sim = []
    for option in range(len(next_sf)):
      state_dif_norm = np.linalg.norm(next_sf[option])
      state_dif_normalized = next_sf[option] / (state_dif_norm + 1e-8)

      evect_norm = np.linalg.norm(evect[option])
      evect_normalized = evect[option] / (evect_norm + 1e-8)
      # evect_norm = np.linalg.norm(evect)
      # evect_normalized = evect / (evect_norm + 1e-8)
      res = np.dot(state_dif_normalized, evect_normalized)
      sim.append(res)
    return sim