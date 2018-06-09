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
import matplotlib.pyplot as plt
import copy
from threading import Barrier, Thread
from tools.timer import Timer

FLAGS = tf.app.flags.FLAGS


class EigenOCAgent(BaseAgent):
  def __init__(self, game, thread_id, global_step, config, lr, network_optimizer, global_network, barrier):
    super(EigenOCAgent, self).__init__(game, thread_id, global_step, config, lr, network_optimizer, global_network)
    self.barrier = barrier

  def init_play(self, sess, saver):
    self.sess = sess
    self.saver = saver
    self.episode_count = sess.run(self.global_step)

    if self.config.move_goal_nb_of_ep and self.config.multi_task:
      self.goal_position = self.env.set_goal(self.episode_count, self.config.move_goal_nb_of_ep)

    self.total_steps = sess.run(self.total_steps_tensor)
    self.eigen_q_value = None
    # self.evalue = None
    tf.logging.info("Starting worker " + str(self.thread_id))
    self.aux_episode_buffer = deque()
    self.ms_aux = self.ms_sf = self.ms_option = self.ms_term = self.ms_critic = None

    if self.name == "worker_0":
      self.init_tracker()

  def init_episode(self):
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
    self.episode_len = 0
    self.sf_counter = 0
    self.option_counter = 0
    self.R = 0
    self.eigen_R = 0
    col_size = self.nb_options + self.action_size if self.config.include_primitive_options else self.nb_options
    self.o_tracker_chosen = np.zeros((col_size,), dtype=np.int32)
    self.o_tracker_steps = np.zeros(col_size, dtype=np.int32)
    self.termination_counter = 0
    self.primitive_action_counter = 0
    self.stats_options = np.zeros((self.nb_states, col_size))
    self.stats_actions = np.zeros((self.nb_states, self.action_size))

    if self.config.decrease_option_prob and self.episode_count < self.config.explore_options_episodes:
      self.sess.run(self.local_network.decrease_prob_of_random_option)

  def SF_prediction(self, s1):
    self.sf_counter += 1
    if self.config.eigen and (self.sf_counter == self.config.max_update_freq or self.done):
      feed_dict = {self.local_network.observation: np.stack([s1])}
      sf = self.sess.run(self.local_network.sf,
                         feed_dict=feed_dict)[0]
      bootstrap_sf = np.zeros_like(sf) if self.done else sf
      self.train_sf(bootstrap_sf)
      self.episode_buffer_sf = []
      self.sf_counter = 0

  def next_frame_prediction(self):
    if len(self.aux_episode_buffer) > self.config.observation_steps and \
                self.total_steps % self.config.aux_update_freq == 0:
      self.train_aux()

  def option_prediction(self, s, s1):
    self.option_counter += 1
    self.store_option_info(s, s1, self.action, self.reward)

    if self.option_counter == self.config.max_update_freq or self.done or (
          self.o_term and self.option_counter >= self.config.min_update_freq):
      if self.done:
        R = 0
        R_mix = 0
      else:
        feed_dict = {self.local_network.observation: np.stack([s1])}
        if self.config.eigen:
          value, evalue, q_value, q_eigen = self.sess.run(
            [self.local_network.v, self.local_network.eigenv, self.local_network.q_val,
             self.local_network.eigen_q_val],
            feed_dict=feed_dict)
          q_value = q_value[0, self.option]
          value = value[0]
          evalue = evalue[0]

          if self.primitive_action:
            R_mix = value if self.o_term else q_value
            # print("primitive_action {} {}".format(R_mix, self.o_term))
          else:
            q_eigen = q_eigen[0, self.option]
            R_mix = evalue if self.o_term else q_eigen
            # print("eigen {} {}".format(R_mix, self.o_term))
            # R_mix = value if self.o_term else q_eigen
        else:
          value, q_value = self.sess.run(
            [self.local_network.v, self.local_network.q_val],
            feed_dict=feed_dict)
          q_value = q_value[0, self.option]
          value = value[0]
        R = value if self.o_term else q_value
        if not self.config.eigen:
          R_mix = R
      # print("-------------------------------------------------------- {}".format(R))
      self.train_option(R, R_mix)

      self.episode_buffer_option = []
      self.option_counter = 0

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.init_play(sess, saver)

      with coord.stop_on_exception():
        while not coord.should_stop():
          if (self.config.steps != -1 and \
                  (self.total_steps > self.config.steps and self.name == "worker_0")) or \
              (self.episode_count > len(self.config.goal_locations) * self.config.move_goal_nb_of_ep and
                   self.name == "worker_0" and self.config.multi_task):
            coord.request_stop()
            return 0

          self.sync_threads()

          if self.name == "worker_0" and self.episode_count > 0 and self.config.eigen and self.config.behaviour_agent is None:
            if self.config.eigen_approach == "SVD":
              self.recompute_eigenvectors_svd()
            else:
              self.recompute_eigenvectors_NN()

          if self.config.sr_matrix is not None:
            self.load_directions()
          self.init_episode()

          s = self.env.reset()
          s_idx = None
          self.option_evaluation(s)
          self.o_tracker_steps[self.option] += 1
          while not self.done:
            self.sync_threads()
            self.policy_evaluation(s)
            if s_idx is not None:
              self.stats_actions[s_idx][self.action] += 1
              self.stats_options[s_idx][self.option] += 1

            s1, r, self.done, s1_idx = self.env.step(self.action)

            self.episode_reward += r
            self.reward = np.clip(r, -1, 1)

            self.option_terminate(s1)

            self.reward_deliberation()

            if self.done:
              s1 = s
              s1_idx = s_idx

            self.store_general_info(s, s1, self.action)
            self.log_timestep()

            # if self.total_steps > self.config.observation_steps:
            if self.config.behaviour_agent is None and self.config.eigen:
              self.SF_prediction(s1)
            self.next_frame_prediction()

            self.option_prediction(s, s1)

            if not self.done and (self.o_term or self.primitive_action):
              self.option_evaluation(s1)

            if not self.done:
              self.o_tracker_steps[self.option] += 1

            if self.total_steps % self.config.steps_checkpoint_interval == 0 and self.name == 'worker_0':
              self.save_model()

            if self.total_steps % self.config.steps_summary_interval == 0 and self.name == 'worker_0':
              self.write_step_summary(r)

            s = s1
            s_idx = s1_idx
            self.episode_len += 1
            self.total_steps += 1

            sess.run(self.increment_total_steps_tensor)

          self.log_episode()
          self.update_episode_stats()

          if self.episode_count % self.config.episode_eval_interval == 0 and \
                  self.name == 'worker_0' and self.episode_count != 0 and self.config.evaluation:
            tf.logging.info("Evaluating agent....")
            eval_episodes_won, mean_ep_length = self.evaluate_agent()
            self.write_eval_summary(eval_episodes_won, mean_ep_length)

          if self.episode_count % self.config.move_goal_nb_of_ep == 0 and \
                  self.episode_count != 0:
            tf.logging.info("Moving GOAL....")
            self.barrier.wait()
            self.goal_position = self.env.set_goal(self.episode_count, self.config.move_goal_nb_of_ep)

          if self.episode_count % self.config.episode_checkpoint_interval == 0 and self.name == 'worker_0' and \
                  self.episode_count != 0:
            self.save_model()

          if self.episode_count % self.config.episode_summary_interval == 0 and self.total_steps != 0 and \
                  self.name == 'worker_0':
            self.write_episode_summary(r)

          if self.name == 'worker_0':
            sess.run(self.increment_global_step)

          self.episode_count += 1

  def reward_deliberation(self):
    self.original_reward = self.reward
    self.reward = float(self.reward) - self.config.discount * (
      float(self.o_term) * self.config.delib_margin * (1 - float(self.done)))

  def option_evaluation(self, s):
    feed_dict = {self.local_network.observation: np.stack([s])}
    self.option, self.primitive_action = self.sess.run(
      [self.local_network.current_option, self.local_network.primitive_action], feed_dict=feed_dict)
    self.option, self.primitive_action = self.option[0], self.primitive_action[0]
    self.o_tracker_chosen[self.option] += 1
    self.episode_options.append(self.option)
    self.primitive_action_counter += self.primitive_action * (1 - self.done)


  def option_terminate(self, s1):
    if self.config.include_primitive_options and self.primitive_action:
      self.o_term = True
    else:
      feed_dict = {self.local_network.observation: np.stack([s1])}
      o_term = self.sess.run(self.local_network.termination, feed_dict=feed_dict)
      self.o_term = o_term[0, self.option] > np.random.uniform()
      self.prob_terms = o_term[0]
    self.termination_counter += self.o_term * (1 - self.done)
    self.episode_oterm.append(self.o_term)

  def policy_evaluation(self, s):
    feed_dict = {self.local_network.observation: np.stack([s])}

    if self.config.eigen:
      tensor_list = [self.local_network.options, self.local_network.v, self.local_network.q_val,
                     self.local_network.eigen_q_val, self.local_network.eigenv]
      options, value, q_value, eigen_q_value, evalue = self.sess.run(tensor_list, feed_dict=feed_dict)
      if not self.primitive_action:
        self.eigen_q_value = eigen_q_value[0, self.option]
        self.episode_eigen_q_values.append(self.eigen_q_value)
        pi = options[0, self.option]
        self.action = np.random.choice(pi, p=pi)
        self.action = np.argmax(pi == self.action)
        self.evalue = evalue[0]
      else:
        self.action = self.option - self.nb_options
      self.q_value = q_value[0, self.option]
      self.q_values = q_value[0]
      self.value = value[0]
    else:
      tensor_list = [self.local_network.options, self.local_network.v, self.local_network.q_val]
      options, value, q_value = self.sess.run(tensor_list, feed_dict=feed_dict)

      if self.config.include_primitive_options and self.primitive_action:
        self.action = self.option - self.nb_options
      else:
        pi = options[0, self.option]
        self.action = np.random.choice(pi, p=pi)
        self.action = np.argmax(pi == self.action)

      self.q_value = q_value[0, self.option]
      self.q_values = q_value[0]
      self.value = value[0]
      self.episode_values.append(self.value)
      self.episode_q_values.append(self.q_value)
    self.episode_actions.append(self.action)

  def store_general_info(self, s, s1, a):
    if self.config.eigen:
      self.episode_buffer_sf.append([s, s1, a])
    if len(self.aux_episode_buffer) == self.config.memory_size:
      self.aux_episode_buffer.popleft()
    self.aux_episode_buffer.append([s, s1, a])

  def store_option_info(self, s, s1, a, r):
    if self.config.eigen and not self.primitive_action:
      feed_dict = {self.local_network.observation: np.stack([s, s1])}
      fi = self.sess.run(self.local_network.fi,
                         feed_dict=feed_dict)
      eigen_r = self.cosine_similarity((fi[1] - fi[0]), self.directions[self.option])
      r_i = self.config.alpha_r * eigen_r + (1 - self.config.alpha_r) * r
      self.r_i = r_i
      # print(r_i)
      if r_i > 1:
        print("ERRROR r_i = {}".format(r_i))

      self.episode_buffer_option.append(
        [s, self.option, a, r, r_i, self.primitive_action, s1])
    else:
      r_i = r
      self.episode_buffer_option.append(
        [s, self.option, a, r, r_i,
         self.primitive_action, s1])

  def recompute_eigenvectors_NN(self):
    if self.config.eigen:
      new_eigenvectors = copy.deepcopy(self.global_network.directions)
      # matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
      for idx in range(self.nb_states):
        s, ii, jj = self.env.fake_get_state(idx)
        if self.env.not_wall(ii, jj):
          feed_dict = {self.local_network.observation: [s]}
          sf = self.sess.run(self.local_network.sf, feed_dict=feed_dict)[0]
          ci = np.argmax(
            [self.cosine_similarity(sf, d) for d in self.global_network.directions])

          sf_norm = np.linalg.norm(sf)
          sf_normalized = sf / (sf_norm + 1e-8)
          new_eigenvectors[ci] = self.config.tau * sf_normalized + (1 - self.config.tau) * \
                                                                   self.global_network.directions[ci]

      # feed_dict = {self.local_network.matrix_sf: [matrix_sf]}
      # eigenval, eigenvect = self.sess.run([self.local_network.eigenvalues, self.local_network.eigenvectors],
      #                                     feed_dict=feed_dict)
      # eigenval, eigenvect = eigenval[0], eigenvect[0]
      # _, eigenval, eigenvect = np.linalg.svd(matrix_sf, full_matrices=False)
      # eigenvalues = eigenval[self.config.first_eigenoption:self.config.nb_options + self.config.first_eigenoption]
      # new_eigenvectors = eigenvect[self.config.first_eigenoption:self.config.nb_options + self.config.first_eigenoption]

      min_similarity = np.min(
        [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
      max_similarity = np.max(
        [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
      mean_similarity = np.mean(
        [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
      self.summary = tf.Summary()
      self.summary.value.add(tag='Eigenvectors/Min similarity', simple_value=float(min_similarity))
      self.summary.value.add(tag='Eigenvectors/Max similarity', simple_value=float(max_similarity))
      self.summary.value.add(tag='Eigenvectors/Mean similarity', simple_value=float(mean_similarity))
      self.summary_writer.add_summary(self.summary, self.episode_count)
      self.summary_writer.flush()
      # tf.logging.warning("Min cosine similarity between old eigenvectors and recomputed onesis {}".format(min_similarity))
      self.global_network.directions = new_eigenvectors
      self.directions = self.global_network.directions

  def recompute_eigenvectors_svd(self):
    if self.config.eigen:
      # new_eigenvectors = copy.deepcopy(self.global_network.directions)
      # matrix_sf = []
      states = []
      for idx in range(self.nb_states):
        s, ii, jj = self.env.fake_get_state(idx)
        if self.env.not_wall(ii, jj):
          states.append(s)

      feed_dict = {self.local_network.observation: states}
      sfs = self.sess.run(self.local_network.sf, feed_dict=feed_dict)
      # _, eigenval, eigenvect = np.linalg.svd(sfs, full_matrices=False)
      feed_dict = {self.local_network.matrix_sf: [sfs]}
      eigenvect = self.sess.run(self.local_network.eigenvectors,
                                feed_dict=feed_dict)
      eigenvect = eigenvect[0]

      new_eigenvectors = copy.deepcopy(
        eigenvect[self.config.first_eigenoption:self.config.nb_options + self.config.first_eigenoption])

      min_similarity = np.min(
        [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
      max_similarity = np.max(
        [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
      mean_similarity = np.mean(
        [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
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

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0)}
    fi = self.sess.run(self.local_network.fi,
                       feed_dict=feed_dict)

    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0)}  # ,

    _, self.ms_sf = \
      self.sess.run([self.local_network.apply_grads_sf,
                     self.local_network.merged_summary_sf],
                    feed_dict=feed_dict)

  def train_aux(self):
    minibatch = random.sample(self.aux_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.actions_placeholder: actions}

    _, _, self.ms_aux = \
      self.sess.run([self.local_network.aux_loss, self.local_network.apply_grads_aux,
                     self.local_network.merged_summary_aux],
                    feed_dict=feed_dict)

  def train_option(self, bootstrap_value, bootstrap_value_mix):  #
    rollout = np.array(
      self.episode_buffer_option)  # s, self.option, a, r, r_i, self.primitive_action, s1
    observations = rollout[:, 0]
    options = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    eigen_rewards = rollout[:, 4]
    primitive_actions = rollout[:, 5]
    next_observations = rollout[:, 6]
    # print(eigen_rewards)

    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_returns = reward_discount(rewards_plus, self.config.discount)[:-1]

    eigen_rewards_plus = np.asarray(eigen_rewards.tolist() + [bootstrap_value_mix])
    discounted_eigen_returns = reward_discount(eigen_rewards_plus, self.config.discount)[:-1]

    # if self.config.eigen:
      # feed_dict = {
      #   self.local_network.observation: np.concatenate((np.stack(observations, 0), np.stack(next_observations, 0)),
      #                                                  axis=0)}
      # fi = self.sess.run(self.local_network.fi,
      #                    feed_dict=feed_dict)
      # fi_next = fi[len(observations):]
      # fi = fi[:len(observations)]
      # real_directions = fi_next - fi
      # real_approx_options = []
      # for i, d in enumerate(real_directions):
      #   if primitive_actions[i]:
      #     real_approx_options.append(options[i])
      #   else:
      #     real_approx_options.append(np.argmax([self.cosine_similarity(d, self.directions[o]) for o in
      #                                           range(self.nb_options)]) if self.episode_count > 0 else options[i])

    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 # self.local_network.options_placeholder: real_approx_options if self.config.eigen else options,
                 self.local_network.options_placeholder: options,
                 }

    _, self.ms_critic = self.sess.run([self.local_network.apply_grads_critic,
                                       self.local_network.merged_summary_critic,
                                       ], feed_dict=feed_dict)

    feed_dict = {
      self.local_network.observation: np.stack(next_observations, axis=0),
      self.local_network.options_placeholder: options,
      # self.local_network.options_placeholder: real_approx_options if self.config.eigen else options,
      self.local_network.primitive_actions_placeholder: primitive_actions
    }

    _, self.ms_term = self.sess.run([self.local_network.apply_grads_term,
                                     self.local_network.merged_summary_term,
                                    ], feed_dict=feed_dict)

    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.target_eigen_return: discounted_eigen_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.options_placeholder: options,
                 self.local_network.primitive_actions_placeholder: primitive_actions
                 }

    _, self.ms_option = self.sess.run([self.local_network.apply_grads_option,
                                       self.local_network.merged_summary_option,
                                       ], feed_dict=feed_dict)


    self.R = discounted_returns[-1]
    self.eigen_R = discounted_eigen_returns[-1]

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
