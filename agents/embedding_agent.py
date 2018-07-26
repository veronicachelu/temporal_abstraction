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
class EmbeddingAgent(EigenOCAgentDyn):
  def __init__(self, sess, game, thread_id, global_step, global_episode, config, global_network, barrier):
    super(EmbeddingAgent, self).__init__(sess, game, thread_id, global_step, global_episode, config, global_network, barrier)

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

            """If we used eigen directions as basis for the options that store transitions for n-step successor representation predictions"""
            if self.config.use_eigendirections:
              self.episode_buffer_sf.append([s, s1, self.action, self.reward, self.fi])
              self.sf_prediction(s1)

            """Keep track of previous option and the indicator of whether it was primitive or not"""
            self.old_option = self.option
            self.old_primitive_action = self.primitive_action

            """If the option terminated or the option was primitive, sample another option"""
            if not self.done and (self.o_term or self.primitive_action):
              self.option_evaluation(s1)

            if not self.done:
              """Increase the timesteps for the options - for statistics purposes"""
              self.o_tracker_steps[self.option] += 1

            """Do n-step prediction for the returns"""
            r_mix = self.option_prediction(s, s1)

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
    feed_dict = {self.local_network.observation: [s]}

    tensor_list = [self.local_network.fi,
                   self.local_network.sf,
                   self.local_network.v,
                   self.local_network.q_val]

    if not self.primitive_action:
      """If the current option is not a primitive action, than add the option direction and the eigen option-value function of the critic"""
      feed_dict[self.local_network.option_direction_placeholder] = [self.directions[self.option]]
      tensor_list += [self.local_network.eigen_q_val, self.local_network.option]

    results = self.sess.run(tensor_list, feed_dict=feed_dict)

    if not self.primitive_action:
      fi,\
      sf,\
      value,\
      q_value,\
      eigen_q_value,\
      option_policy = results
      """Add the eigen option-value function to the buffer in order to add stats to tensorboad at the end of the episode"""
      self.eigen_q_value = eigen_q_value[0]
      self.episode_eigen_q_values.append(self.eigen_q_value)

      """Get the intra-option policy for the current option"""
      pi = option_policy[0]
      """Sample an action"""
      self.action = np.random.choice(pi, p=pi)
      self.action = np.argmax(pi == self.action)
    else:
      """If the option is a primitve action"""
      fi,\
      sf,\
      value,\
      q_value = results
      self.action = self.option - self.nb_options

    """Get the option-value function for the external reward signal corresponding to the current option"""
    self.q_value = q_value[0, self.option]
    """Store also all the option-value functions for the external reward signal"""
    self.q_values = q_value[0]
    """Get the state value function corresponding to the external reward signal"""
    self.value = value[0]

    sf = sf[0]
    self.fi = fi[0]
    self.add_SF(sf)

    """Store information in buffers for stats in tensorboard"""
    self.episode_values.append(self.value)
    self.episode_q_values.append(self.q_value)
    self.episode_actions.append(self.action)

  """Do n-step prediction for the returns and update the option policies and critics"""
  def option_prediction(self, s, s1):
    """If the option chosen was not primitive, than we can construct
        the mixed reward signal to pass to the eigen intra-option critics."""
    if not self.old_primitive_action:
      feed_dict = {self.local_network.observation: np.stack([s, s1])}
      fi = self.sess.run(self.local_network.fi,
                         feed_dict=feed_dict)
      """The internal reward will be the cosine similary between the direction in latent space and the 
           eigen direction corresponding to the current option"""
      r_i = self.cosine_similarity((fi[1] - fi[0]), self.directions[self.old_option])
      r_mix = self.config.alpha_r * r_i + (1 - self.config.alpha_r) * self.reward
    else:
      r_mix = self.reward

    """Adding to the transition buffer for doing n-step prediction on critics and policies"""
    self.episode_buffer_option.append(
      [s, self.old_option, self.action, self.reward, r_mix, self.old_primitive_action, s1])

    if len(self.episode_buffer_option) >= self.config.max_update_freq or self.done or (
          self.o_term and len(self.episode_buffer_option) >= self.config.min_update_freq):
      """Get the bootstrap option-value functions for the next time step"""
      if self.done:
        bootstrap_Q = 0
        bootstrap_eigen_Q = 0
      else:
        feed_dict = {self.local_network.observation: [s1]}
        to_run = [self.local_network.v,
                  self.local_network.q_val]
        """If the previous option was not primitive than it makes sense to plug in the previous's option direction in order to compute the eigen option-value function for the critic"""
        if not self.old_primitive_action:
          feed_dict[self.local_network.option_direction_placeholder] = [self.directions[self.old_option]]
          to_run.append(self.local_network.eigen_q_val)

        results = self.sess.run(to_run, feed_dict=feed_dict)

        if self.old_primitive_action:
          value, q_values = results
          q_value = q_values[0, self.old_option]
          value = value[0]
          """In the previous option was primitve than the bootstrap return for the eigen option is the same as the bootstrap for the option"""
          bootstrap_eigen_Q = value if self.o_term else q_value
        else:
          """Otherwise we have to compute the bootstrap return for the eigen option"""
          value, q_values, q_eigen = results
          q_value = q_values[0, self.old_option]
          value = value[0]
          q_eigen = q_eigen[0]
          """Not sure this is the right way to do it.
          We construct the expected value of the next state under the mixed reward signal by taking the expectation of the eigen option-value functions using all possible directions, i.e. all possible options, weighting them under a tilted softmax"""
          if self.o_term:
            feed_dict = {self.local_network.observation: np.repeat([s1], self.nb_options, 0),
                         self.local_network.option_direction_placeholder: self.directions,
                         }
            eigen_qs, random_option_prob = self.sess.run([self.local_network.eigen_q_val, self.local_network.random_option_prob], feed_dict=feed_dict)
            random_option_prob = random_option_prob
            if self.config.include_primitive_options:
              concat_eigen_qs = np.concatenate((eigen_qs, q_values[0, self.config.nb_options:]))
            else:
              concat_eigen_qs = eigen_qs
            evalue = np.max(concat_eigen_qs) * (1 - random_option_prob) + random_option_prob * np.mean(concat_eigen_qs)
            bootstrap_eigen_Q = evalue
          else:
            bootstrap_eigen_Q = q_eigen

        bootstrap_Q = value if self.o_term else q_value

      self.train_option(bootstrap_Q, bootstrap_eigen_Q)

      self.episode_buffer_option = []

  """Do n-step prediction for the successor representation latent and an update for the representation latent using 1-step next frame prediction"""
  def sf_prediction(self, s1):
    if self.config.use_eigendirections and (len(self.episode_buffer_sf) == self.config.max_update_freq or self.done):
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

    _, self.summaries_sf, sf_loss, _, self.summaries_aux, aux_loss = \
      self.sess.run([self.local_network.apply_grads_sf,
                     self.local_network.merged_summary_sf,
                     self.local_network.sf_loss,
                     self.local_network.apply_grads_aux,
                     self.local_network.merged_summary_aux,
                     self.local_network.aux_loss
                     ],
                    feed_dict=feed_dict)

  """Do n-step prediction on the critics and policies"""
  def train_option(self, bootstrap_value, bootstrap_value_mix):
    rollout = np.array(self.episode_buffer_option)
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

    """Get the real directions executed in the environment, not the ones corresponding to the options of the high-level policy, since the former might not be the ones that need to be assigned credit for the return"""
    feed_dict = {
      self.local_network.observation: np.concatenate((np.stack(observations, 0), np.stack(next_observations, 0)),
                                                     axis=0)}
    fi = self.sess.run(self.local_network.fi,
                       feed_dict=feed_dict)
    fi_next = fi[len(observations):]
    fi = fi[:len(observations)]
    real_directions = fi_next - fi
    real_approx_options, directions = [], []
    for i, d in enumerate(real_directions):
      if primitive_actions[i]:
        real_approx_options.append(options[i])
        directions.append(np.zeros((self.config.sf_layers[-1])))
      else:
        directions.append(self.global_network.directions[options[i]])
        real_approx_options.append(np.argmax([self.cosine_similarity(d, self.directions[o]) for o in
                                              range(self.nb_options)]) if self.total_episodes > 0 else options[i])

    """Do an update on the option-value function critic"""
    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 # self.local_network.options_placeholder: real_approx_options,
                 self.local_network.options_placeholder: options,
                 # self.local_network.option_direction_placeholder: real_directions
                 self.local_network.option_direction_placeholder: directions
                 }

    _, self.summaries_critic = self.sess.run([self.local_network.apply_grads_critic,
                                       self.local_network.merged_summary_critic,
                                       ], feed_dict=feed_dict)

    """Do an update on the option termination conditions"""
    feed_dict = {
      self.local_network.observation: np.stack(next_observations, axis=0),
      # self.local_network.options_placeholder: real_approx_options,
      self.local_network.options_placeholder: options,
      # self.local_network.option_direction_placeholder: real_directions,
      self.local_network.option_direction_placeholder: directions,
      self.local_network.primitive_actions_placeholder: primitive_actions
    }

    _, self.summaries_termination = self.sess.run([self.local_network.apply_grads_term,
                                     self.local_network.merged_summary_term,
                                    ], feed_dict=feed_dict)

    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.target_eigen_return: discounted_eigen_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.options_placeholder: options,
                 self.local_network.option_direction_placeholder: directions,
                 self.local_network.primitive_actions_placeholder: primitive_actions
                 }

    """Do an update on the intra-option policies"""
    _, self.summaries_option = self.sess.run([self.local_network.apply_grads_option,
                                       self.local_network.merged_summary_option,
                                       ], feed_dict=feed_dict)

    """Store the bootstrap target returns at the end of the trajectory"""
    self.R = discounted_returns[-1]
    self.eigen_R = discounted_eigen_returns[-1]
