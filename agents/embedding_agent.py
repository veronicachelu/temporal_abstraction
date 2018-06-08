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
from agents.eigenoc_agent_dynamic import EigenOCAgentDyn
import copy
from threading import Barrier, Thread

FLAGS = tf.app.flags.FLAGS


class EmbeddingAgent(EigenOCAgentDyn):
  def __init__(self, game, thread_id, global_step, config, lr, network_optimizer, global_network, barrier):
    super(EmbeddingAgent, self).__init__(game, thread_id, global_step, config, lr, network_optimizer, global_network,
                                         barrier)
    self.barrier = barrier

  def init_play(self, sess, saver):
    super(EmbeddingAgent, self).init_play(sess, saver)
    self.ms_critic = self.ms_eigen_critic = None

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

          self.sync_threads(force=True)

          if self.name == "worker_0" and self.episode_count > 0 and self.config.eigen and self.config.behaviour_agent is None:
            if self.config.eigen_approach == "SVD":
              self.recompute_eigenvectors_dynamic_SVD()

          if self.config.sr_matrix is not None:
            self.load_directions()
          self.init_episode()

          s = self.env.reset()
          self.option_evaluation(s, None)
          self.o_tracker_steps[self.option] += 1
          while not self.done:
            self.sync_threads()
            self.policy_evaluation(s)

            s1, r, self.done, s1_idx = self.env.step(self.action)

            self.episode_reward += r
            self.reward = np.clip(r, -1, 1)

            self.option_terminate(s1)

            self.reward_deliberation()

            if self.done:
              s1 = s

            self.episode_buffer_sf.append([s, s1, self.action, self.reward, self.fi])

            self.log_timestep()

            self.SF_prediction(s1)
            # self.next_frame_prediction()

            self.old_option = self.option
            self.old_primitive_action = self.primitive_action

            if not self.done and (self.o_term or self.primitive_action):
              self.option_evaluation(s1, s1_idx)

            if not self.done:
              self.o_tracker_steps[self.option] += 1

            self.option_prediction(s, s1, r)

            if self.total_steps % self.config.steps_checkpoint_interval == 0 and self.name == 'worker_0':
              self.save_model()

            if self.total_steps % self.config.steps_summary_interval == 0 and self.name == 'worker_0':
              self.write_step_summary(r)

            s = s1
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

  def option_terminate(self, s1):
    if self.config.include_primitive_options and self.primitive_action:
      self.o_term = True
    else:
      feed_dict = {self.local_network.observation: np.stack([s1]),
                   self.local_network.option_direction_placeholder: [self.global_network.directions[self.option]]}
      o_term = self.sess.run(self.local_network.termination, feed_dict=feed_dict)
      self.prob_terms = [o_term[0]]
      self.o_term = o_term[0] > np.random.uniform()

    self.termination_counter += self.o_term * (1 - self.done)
    self.episode_oterm.append(self.o_term)

  def add_SF(self, sf):
    if self.config.eigen_approach == "SVD":
      self.global_network.sf_matrix_buffer[0] = sf.copy()
      self.global_network.sf_matrix_buffer = np.roll(self.global_network.sf_matrix_buffer, 1, 0)
    else:
      ci = np.argmax(
        [self.cosine_similarity(sf, d) for d in self.global_network.directions])

      sf_norm = np.linalg.norm(sf)
      sf_normalized = sf / (sf_norm + 1e-8)
      self.global_network.directions[ci] = self.config.tau * sf_normalized + (1 - self.config.tau) * \
                                                                             self.global_network.directions[ci]
      self.directions = self.global_network.directions

  def policy_evaluation(self, s):
    feed_dict = {self.local_network.observation: np.stack([s])}

    tensor_list = [self.local_network.fi, self.local_network.sf, self.local_network.v, self.local_network.q_val]
    if not self.primitive_action:
      feed_dict[self.local_network.option_direction_placeholder] = [self.global_network.directions[self.option]]
      tensor_list += [self.local_network.eigen_q_val, self.local_network.option]

    results = self.sess.run(tensor_list, feed_dict=feed_dict)

    if not self.primitive_action:
      fi, sf, value, q_value, eigen_q_value, option_policy = results
      self.eigen_q_value = eigen_q_value[0]
      self.episode_eigen_q_values.append(self.eigen_q_value)
      pi = option_policy[0]
      self.action = np.random.choice(pi, p=pi)
      self.action = np.argmax(pi == self.action)
    else:
      fi, sf, value, q_value = results
      self.action = self.option - self.nb_options
    self.q_value = q_value[0, self.option]
    self.q_values = q_value[0]
    self.value = value[0]

    sf = sf[0]
    self.fi = fi[0]
    self.add_SF(sf)
    self.episode_actions.append(self.action)
    self.episode_values.append(self.value)
    self.episode_q_values.append(self.q_value)

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps),
                    global_step=self.global_step)
    tf.logging.info(
      "Saved Model at {}".format(self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps)))

    if self.config.sr_matrix is not None:
      self.save_SF_matrix()
    if self.config.eigen:
      self.save_eigen_directions()

  def store_option_info(self, s, s1, a, r):
    if self.config.eigen and not self.primitive_action:
      feed_dict = {self.local_network.observation: np.stack([s, s1])
                   }

      fi = self.sess.run(self.local_network.fi,
                         feed_dict=feed_dict)
      eigen_r = self.cosine_similarity((fi[1] - fi[0]), self.directions[self.option])
      r_i = self.config.alpha_r * eigen_r + (1 - self.config.alpha_r) * r
      if np.isnan(r_i):
        print("NAN")
      self.episode_buffer_option.append(
        [s, self.option, a, r, r_i, self.primitive_action, s1])
    else:
      r_i = r
      self.episode_buffer_option.append(
        [s, self.option, a, r, r_i,
         self.primitive_action, s1])

  def option_prediction(self, s, s1, r):
    self.option_counter += 1
    self.store_option_info(s, s1, self.action, r)

    if self.option_counter == self.config.max_update_freq or self.done or (
          self.o_term and self.option_counter >= self.config.min_update_freq):
      if self.done:
        R = 0
        R_mix = 0
      else:
        feed_dict = {self.local_network.observation: np.stack([s1])}
        to_run = [self.local_network.v, self.local_network.q_val]
        if not self.primitive_action:
          feed_dict[self.local_network.option_direction_placeholder] = [self.global_network.directions[self.option]]
          to_run.append(self.local_network.eigen_q_val)

        results = self.sess.run(to_run, feed_dict=feed_dict)

        if self.primitive_action:
          value, q_value = results
          q_value = q_value[0, self.option]
          value = value[0]
          R_mix = value if self.o_term else q_value
        else:
          value, q_value, q_eigen = results
          q_value = q_value[0, self.option]
          value = value[0]
          q_eigen = q_eigen[0]
          if self.o_term:
            feed_dict = {self.local_network.observation: np.repeat([s1], self.nb_options, 0),
                         self.local_network.option_direction_placeholder: self.directions,
                         }
            eigen_qs = self.sess.run(self.local_network.eigen_q_val, feed_dict=feed_dict)
            evalue = np.mean(eigen_qs)
            R_mix = evalue
          else:
            R_mix = q_eigen

        R = value if self.o_term else q_value

      results = self.train_option(R, R_mix)
      if results is not None:
        self.ms_option, self.ms_term, self.ms_critic, self.ms_eigen_critic, \
        option_loss, policy_loss, entropy_loss, critic_loss, eigen_critic_loss, term_loss, self.R, self.eigen_R = results

      self.episode_buffer_option = []
      self.option_counter = 0

  def SF_prediction(self, s1):
    self.sf_counter += 1
    if self.config.eigen and (self.sf_counter == self.config.max_update_freq or self.done):
      feed_dict = {self.local_network.observation: np.stack([s1])}
      sf = self.sess.run(self.local_network.sf,
                         feed_dict=feed_dict)[0]
      bootstrap_sf = np.zeros_like(sf) if self.done else sf
      self.ms_sf, self.sf_loss, self.ms_aux, self.aux_loss = self.train_sf(bootstrap_sf)
      self.episode_buffer_sf = []
      self.sf_counter = 0

  def train_sf(self, bootstrap_sf):
    rollout = np.array(self.episode_buffer_sf)

    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    fi = rollout[:, 4]

    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0)}

    _, ms_sf, sf_loss, _, ms_aux, aux_loss = \
      self.sess.run([self.local_network.apply_grads_sf,
                     self.local_network.merged_summary_sf,
                     self.local_network.sf_loss,
                     self.local_network.apply_grads_aux,
                     self.local_network.merged_summary_aux,
                     self.local_network.aux_loss
                     ],
                    feed_dict=feed_dict)

    return ms_sf, sf_loss, ms_aux, aux_loss

  def recompute_eigenvectors_dynamic_SVD(self):
    if self.config.eigen:
      feed_dict = {self.local_network.matrix_sf: [self.global_network.sf_matrix_buffer]}
      eigenvect = self.sess.run(self.local_network.eigenvectors,
                                feed_dict=feed_dict)
      eigenvect = eigenvect[0]

      new_eigenvectors = self.associate_closest_vectors(self.global_network.directions, eigenvect)

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
      self.global_network.directions = new_eigenvectors
      self.directions = self.global_network.directions

  def associate_closest_vectors(self, old, new):
    to_return = copy.deepcopy(old)
    skip_list = []
    featured = new[self.config.first_eigenoption: self.config.nb_options + self.config.first_eigenoption]

    for v in featured:
      sign = np.argmax(
        [np.sum([np.sign(np.dot(v, x)) * (np.dot(v, x) ** 2) for x in self.global_network.sf_matrix_buffer]),
         np.sum([np.sign(np.dot((-1) * v, x)) * (np.dot(v, x) ** 2) for x in self.global_network.sf_matrix_buffer])])
      if sign == 1:
        v = (-1) * v
      distances = [
        -np.inf if b_idx in skip_list else (np.inf if np.all(b == np.zeros(b.shape)) else self.cosine_similarity(v, b))
        for b_idx, b in enumerate(self.global_network.directions)]
      closest_distance_idx = np.argmax(distances)

      old_v_idx = closest_distance_idx
      skip_list.append(old_v_idx)

      to_return[old_v_idx] = v

    return to_return

  def save_SF_matrix(self):
    np.save(self.global_network.sf_matrix_path, self.global_network.sf_matrix_buffer)

  def save_eigen_directions(self):
    np.save(self.global_network.directions_path, self.global_network.directions)

  def train_option(self, bootstrap_value, bootstrap_value_mix):
    rollout = np.array(self.episode_buffer_option)  # s, self.option, self.action, r, r_i
    observations = rollout[:, 0]
    options = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    eigen_rewards = rollout[:, 4]
    primitive_actions = rollout[:, 5]
    next_observations = rollout[:, 6]

    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_returns = reward_discount(rewards_plus, self.config.discount)[:-1]

    eigen_rewards_plus = np.asarray(eigen_rewards.tolist() + [bootstrap_value_mix])
    discounted_eigen_returns = reward_discount(eigen_rewards_plus, self.config.discount)[:-1]



    feed_dict = {self.local_network.observation: np.stack(observations + next_observations, axis=0)}
    fi = self.sess.run(self.local_network.fi,
                       feed_dict=feed_dict)
    fi_next = fi[len(observations):]
    fi = fi[:len(observations)]
    directions = fi_next - fi
    real_approx_options = []
    for i, d in enumerate(directions):
      if primitive_actions[i]:
        real_approx_options[i] = options[i]
      else:
        real_approx_options.append(
          np.argmax([self.cosine_similarity(d, self.directions[o]) for o in range(self.nb_options)]))

    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.options_placeholder: options,
                 self.local_network.option_direction_placeholder: directions,
                 self.local_network.target_eigen_return: discounted_eigen_returns,
                 self.local_network.primitive_actions_placeholder: primitive_actions
                 }

    _, self.ms_option = self.sess.run([self.local_network.apply_grads_option,
                                       self.local_network.merged_summary_option,
                                       ], feed_dict=feed_dict)

    feed_dict = {
      self.local_network.observation: np.stack(next_observations, axis=0),
      self.local_network.options_placeholder: options,
      self.local_network.option_direction_placeholder: directions,
      self.local_network.primitive_actions_placeholder: primitive_actions
    }

    _, self.ms_term = self.sess.run([
      self.local_network.apply_grads_term,
      self.local_network.merged_summary_term,
    ], feed_dict=feed_dict)

    self.R = discounted_returns[-1]
    self.eigen_R = discounted_eigen_returns[-1]

  def write_step_summary(self, r):
    self.summary = tf.Summary()
    if self.ms_sf is not None:
      self.summary_writer.add_summary(self.ms_sf, self.total_steps)
    if self.ms_aux is not None:
      self.summary_writer.add_summary(self.ms_aux, self.total_steps)
    if self.ms_option is not None:
      self.summary_writer.add_summary(self.ms_option, self.total_steps)
    if self.ms_term is not None:
      self.summary_writer.add_summary(self.ms_term, self.total_steps)

    # if self.total_steps > self.config.eigen_exploration_steps:
    self.summary.value.add(tag='Step/Reward', simple_value=r)
    self.summary.value.add(tag='Step/Action', simple_value=self.action)
    self.summary.value.add(tag='Step/Option', simple_value=self.option)
    self.summary.value.add(tag='Step/Q', simple_value=self.q_value)
    if self.config.eigen and not self.primitive_action and self.eigen_q_value is not None:
      self.summary.value.add(tag='Step/EigenQ', simple_value=self.eigen_q_value)
      # self.summary.value.add(tag='Step/EigenV', simple_value=self.evalue)
    self.summary.value.add(tag='Step/V', simple_value=self.value)
    self.summary.value.add(tag='Step/Term', simple_value=int(self.o_term))
    self.summary.value.add(tag='Step/R', simple_value=self.R)
    if self.config.eigen:
      self.summary.value.add(tag='Step/EigenR', simple_value=self.eigen_R)

    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()
    # tf.logging.warning("Writing step summary....")

  def write_episode_summary(self, r):
    self.tracker()
    # self.write_option_map()
    # self.write_episode_summary_stats()
    # self.viz_options()
    self.summary = tf.Summary()
    if len(self.episode_rewards) != 0:
      last_reward = self.episode_rewards[-1]
      self.summary.value.add(tag='Perf/Reward', simple_value=float(last_reward))
    if len(self.episode_lengths) != 0:
      mean_length = np.mean(self.episode_lengths[-self.config.episode_summary_interval:])
      self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
    if len(self.episode_term_prob) != 0:
      mean_term_prob = np.mean(self.episode_term_prob[-self.config.episode_summary_interval:])
      self.summary.value.add(tag='Perf/Term_prob', simple_value=float(mean_term_prob))
    if len(self.episode_primtive_action_prob) != 0:
      mean_primitive_prob = np.mean(self.episode_primtive_action_prob[-self.config.episode_summary_interval:])
      self.summary.value.add(tag='Perf/Primitive_prob', simple_value=float(mean_primitive_prob))
    if len(self.episode_mean_values) != 0:
      last_mean_value = np.mean(self.episode_mean_values[-self.config.episode_summary_interval:])
      self.summary.value.add(tag='Perf/Value', simple_value=float(last_mean_value))
    if len(self.episode_mean_q_values) != 0:
      last_mean_q_value = np.mean(self.episode_mean_q_values[-self.config.episode_summary_interval:])
      self.summary.value.add(tag='Perf/QValue', simple_value=float(last_mean_q_value))
    if self.config.eigen and len(self.episode_mean_eigen_q_values) != 0:
      last_mean_eigen_q_value = np.mean(self.episode_mean_eigen_q_values[-self.config.episode_summary_interval:])
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
    # for op in range(self.config.nb_options):
    #   self.summary.value.add(tag='Perf/Option_length_{}'.format(op), simple_value=self.episode_mean_options_lengths[op])
    self.summary.value.add(tag='Perf/Goal_position', simple_value=self.goal_position)

    self.summary_writer.add_summary(self.summary, self.episode_count)
    self.summary_writer.flush()
    self.write_step_summary(r)
