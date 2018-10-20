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
from agents.eigenoc_agent_dynamic import EigenOCAgentDyn
import copy
from threading import Barrier, Thread

FLAGS = tf.app.flags.FLAGS

"""This Agent is a specialization of the successor representation direction based agent with buffer SR matrix, but instead of choosing from discreate options that are grounded in the SR basis only by means of the pseudo-reward, it keeps a singly intra-option policy whose context is changed by means of the option given as embedding (the embedding being the direction given by the spectral decomposition of the SR matrix)"""
class AttentionAgent(EigenOCAgentDyn):
  def __init__(self, sess, game, thread_id, global_step, global_episode, config, global_network, barrier):
    super(AttentionAgent, self).__init__(sess, game, thread_id, global_step, global_episode, config, global_network, barrier)

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

          self.init_episode()

          """Reset the environment and get the initial state"""
          s = self.env.reset()

          """While the episode does not terminate"""
          while not self.done:
            """update local network parameters from global network"""
            self.sync_threads()

            """Choose an action from the current intra-option policy"""
            self.policy_evaluation(s)

            s1, r, self.done, self.s1_idx = self.env.step(self.action)

            self.episode_reward += r
            self.reward = np.clip(r, -1, 1)

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

            self.episode_buffer_sf.append([s, s1, self.action, self.reward, self.fi])
            self.sf_prediction(s1)

            """If the experience buffer has sufficient experience in it, every so often do an update with a batch of transition from it for next state prediction"""
            self.next_frame_prediction()

            """Do n-step prediction for the returns"""
            # r_mix = self.option_prediction(s, s1)
            r_mix = 0

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

  """Check is the option terminates at the next state"""
  def option_terminate(self, s1):
    """If we took a primitive option, termination is assured"""
    if self.config.include_primitive_options and self.primitive_action:
      self.o_term = True
    else:
      feed_dict = {self.local_network.observation: [s1],
                   self.local_network.option_direction_placeholder: [self.global_network.directions[self.option]]}
      o_term = self.sess.run(self.local_network.termination, feed_dict=feed_dict)
      self.prob_terms = [o_term[0]]
      self.o_term = o_term[0] > np.random.uniform()

    """Stats for tracking option termination"""
    self.termination_counter += self.o_term * (1 - self.done)
    self.episode_oterm.append(self.o_term)
    self.o_tracker_len[self.option].append(self.crt_op_length)

  """Sample an action from the current option's policy"""
  def policy_evaluation(self, s):

    feed_dict = {self.local_network.observation: [s],
                 self.local_network.matrix_sf: [self.global_network.sf_matrix_buffer]}

    tensor_list = [self.local_network.fi,
                   self.local_network.sf,]
                   # self.local_network.current_option_direction,
                   # self.local_network.eigen_q_val,
                   # self.local_network.option_policy]
    try:
      results = self.sess.run(tensor_list, feed_dict=feed_dict)
    except:
      print("pam pam")

    fi,\
    sf, \
      = results
    # current_option_direction,\
    # eigen_q_value,\
    # option_policy \

    """Add the eigen option-value function to the buffer in order to add stats to tensorboad at the end of the episode"""
    # self.eigen_q_value = eigen_q_value[0]
    # self.episode_eigen_q_values.append(self.eigen_q_value)
    # self.current_option_direction = current_option_direction[0]

    # """Get the intra-option policy for the current option"""
    # if np.isnan(self.current_option_direction[0]):
    #   print("NAN error")
		#
		# pi = option_policy[0]
		# """Sample an action"""
    # self.action = np.random.choice(pi, p=pi)
    # self.action = np.argmax(pi == self.action)

    ###### EXECUTE RANDOM ACTION TODO ####
    self.action = np.random.choice(range(self.action_size))

    sf = sf[0]
    self.fi = fi[0]
    # self.add_SF(sf)

    """Store information in buffers for stats in tensorboard"""
    self.episode_actions.append(self.action)

  """Do n-step prediction for the returns and update the option policies and critics"""
  def option_prediction(self, s, s1):
    """construct the mixed reward signal to pass to the eigen intra-option critics."""
    feed_dict = {self.local_network.observation: np.stack([s, s1])}
    fi = self.sess.run(self.local_network.fi,
                       feed_dict=feed_dict)
    """The internal reward will be the cosine similary between the direction in latent space and the 
         eigen direction corresponding to the current option"""
    r_i = self.cosine_similarity((fi[1] - fi[0]), self.current_option_direction)
    r_mix = self.config.alpha_r * r_i + (1 - self.config.alpha_r) * self.reward

    """Adding to the transition buffer for doing n-step prediction on critics and policies"""
    self.episode_buffer_option.append(
      [s, self.current_option_direction, self.action, self.reward, r_mix, s1])

    if len(self.episode_buffer_option) >= self.config.max_update_freq or self.done or (
          self.o_term and len(self.episode_buffer_option) >= self.config.min_update_freq):
      """Get the bootstrap option-value functions for the next time step"""
      if self.done:
        bootstrap_eigen_Q = 0
      else:
        feed_dict = {self.local_network.observation: [s1],
                     self.local_network.matrix_sf: [self.global_network.sf_matrix_buffer]}
        q_eigen = self.sess.run(self.local_network.eigen_q_val, feed_dict=feed_dict)

        q_eigen = q_eigen[0]
        bootstrap_eigen_Q = q_eigen

      self.train_option(bootstrap_eigen_Q)

      self.episode_buffer_option = []

  """Do n-step prediction for the successor representation latent and an update for the representation latent using 1-step next frame prediction"""
  def sf_prediction(self, s1):
    if len(self.episode_buffer_sf) == self.config.max_update_freq or self.done:
      """Get the successor features of the next state for which to bootstrap from"""
      feed_dict = {self.local_network.observation: [s1]}
      next_sf = self.sess.run(self.local_network.sf,
                         feed_dict=feed_dict)[0]
      bootstrap_sf = np.zeros_like(next_sf) if self.done else next_sf
      self.train_sf(bootstrap_sf)
      self.episode_buffer_sf = []

  """Do one n-step update for training the agent's latent successor representation space and an update for the next frame prediction"""
  def train_sf(self, bootstrap_sf):
    rollout = np.array(self.episode_buffer_sf)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    fi = rollout[:, 4]

    """Construct list of latent representations for the entire trajectory"""
    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    """Construct the targets for the next step successor representations for the entire trajectory"""
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0)}

    # _, self.summaries_sf, sf_loss, _, self.summaries_aux, aux_loss = \
    _, self.summaries_sf, sf_loss = \
      self.sess.run([self.local_network.apply_grads_sf,
                     self.local_network.merged_summary_sf,
                     self.local_network.sf_loss,
                     # self.local_network.apply_grads_aux,
                     # self.local_network.merged_summary_aux,
                     # self.local_network.aux_loss
                     ],
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
  def train_option(self, bootstrap_value_mix):
    rollout = np.array(self.episode_buffer_option)
    observations = rollout[:, 0]
    option_directions = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    eigen_rewards = rollout[:, 4]
    next_observations = rollout[:, 5]

    """Construct list of discounted returns using mixed reward signals for the entire n-step trajectory"""
    eigen_rewards_plus = np.asarray(eigen_rewards.tolist() + [bootstrap_value_mix])
    discounted_eigen_returns = reward_discount(eigen_rewards_plus, self.config.discount)[:-1]

    feed_dict = {
                 self.local_network.target_eigen_return: discounted_eigen_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.matrix_sf: [self.global_network.sf_matrix_buffer]
                 # self.local_network.current_option_direction: option_directions,
                 }

    """Do an update on the intra-option policies"""
    _, self.summaries_option = self.sess.run([self.local_network.apply_grads_option,
                                       self.local_network.merged_summary_option,
                                       ], feed_dict=feed_dict)

    """Store the bootstrap target returns at the end of the trajectory"""
    self.eigen_R = discounted_eigen_returns[-1]

  def write_step_summary(self, r, r_mix=None):
    self.summary = tf.Summary()
    self.summary.value.add(tag='Step/Action', simple_value=self.action)
    self.summary.value.add(tag='Step/EigenQ', simple_value=self.eigen_q_value)
    self.summary.value.add(tag='Step/Target_EigenQ', simple_value=self.eigen_R)

    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()

  def update_episode_stats(self):
    if len(self.episode_eigen_q_values) != 0:
      self.episode_mean_eigen_q_values.append(np.mean(self.episode_eigen_q_values))
    if len(self.episode_actions) != 0:
      self.episode_mean_actions.append(get_mode(self.episode_actions))

  def write_summaries(self):
    self.summary = tf.Summary()
    self.summary.value.add(tag='Perf/Reward', simple_value=float(self.episode_reward))
    self.summary.value.add(tag='Perf/Length', simple_value=float(self.episode_length))

    for sum in [self.summaries_sf, self.summaries_aux, self.summaries_critic, self.summaries_option]:
      if sum is not None:
        self.summary_writer.add_summary(sum, self.global_episode_np)

    if len(self.episode_mean_eigen_q_values) != 0:
      last_mean_eigen_q_value = np.mean(self.episode_mean_eigen_q_values[-self.config.step_summary_interval:])
      self.summary.value.add(tag='Perf/EigenQValue', simple_value=float(last_mean_eigen_q_value))
    if len(self.episode_mean_actions) != 0:
      last_frequent_action = self.episode_mean_actions[-1]
      self.summary.value.add(tag='Perf/FreqActions', simple_value=last_frequent_action)

    self.summary_writer.add_summary(self.summary, self.global_episode_np)
    self.summary_writer.flush()
