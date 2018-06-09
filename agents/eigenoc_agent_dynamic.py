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
from agents.eigenoc_agent import EigenOCAgent
import copy
from threading import Barrier, Thread

FLAGS = tf.app.flags.FLAGS


class EigenOCAgentDyn(EigenOCAgent):
  def __init__(self, game, thread_id, global_step, config, lr, network_optimizer, global_network, barrier):
    super(EigenOCAgentDyn, self).__init__(game, thread_id, global_step, config, lr, network_optimizer, global_network,
                                          barrier)
    self.barrier = barrier

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
          r_i = 0

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

            if self.config.behaviour_agent is None and self.config.eigen:
              self.SF_prediction(s1)
            self.next_frame_prediction()

            # if self.episode_count > 0:
            if self.episode_count > 0 and (not self.config.eigen or (self.config.eigen and
                                                             len(self.directions) == self.nb_options)):
              r_i = self.option_prediction(s, s1)

            if not self.done and (self.o_term or self.primitive_action):
              self.option_evaluation(s1)

            if not self.done:
              self.o_tracker_steps[self.option] += 1

            if self.total_steps % self.config.steps_checkpoint_interval == 0 and self.name == 'worker_0':
              self.save_model()

            if self.total_steps % self.config.steps_summary_interval == 0 and self.name == 'worker_0':
              self.write_step_summary(r, r_i)

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

  def add_SF(self, sf):
    if self.config.eigen_approach == "SVD":
      self.global_network.sf_matrix_buffer[0] = sf.copy()
      self.global_network.sf_matrix_buffer = np.roll(self.global_network.sf_matrix_buffer, 1, 0)
    else:
      old_directions = copy.deepcopy(self.directions)
      self.global_network.eigencluster.cluster(sf)
      self.load_directions()
      # ci = np.argmax(
      #   [self.cosine_similarity(sf, d) for d in self.global_network.directions])
      #
      # sf_norm = np.linalg.norm(np.asarray(sf, np.float64))
      # sf_normalized = sf / (sf_norm + 1e-8)
      # new_center = self.config.tau * sf_normalized + (1 - self.config.tau) * self.global_network.directions[ci]
      # new_center_norm = np.linalg.norm(np.asarray(new_center, np.float64))
      # self.global_network.directions[ci] = new_center / (new_center_norm + 1e-8)
      # self.directions = self.global_network.directions

      if len(old_directions) == self.nb_options and len(
          self.directions) == self.nb_options and self.name == "worker_0" and self.total_steps % self.config.steps_summary_interval == 0:
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
        self.summary_writer.add_summary(self.summary, self.episode_count)
        self.summary_writer.flush()

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
      tensor_list = [self.local_network.sf, self.local_network.options, self.local_network.v,
                     self.local_network.q_val,
                     self.local_network.eigen_q_val, self.local_network.eigenv]
      sf, options, value, q_value, eigen_q_value, evalue = self.sess.run(tensor_list, feed_dict=feed_dict)
      if not self.primitive_action:
        self.eigen_q_value = eigen_q_value[0, self.option]
        pi = options[0, self.option]
        self.action = np.random.choice(pi, p=pi)
        self.action = np.argmax(pi == self.action)
        self.evalue = evalue[0]
      else:
        self.action = self.option - self.nb_options
      self.q_value = q_value[0, self.option]
      self.q_values = q_value[0]
      self.value = value[0]

      if self.config.behaviour_agent is None:
        sf = sf[0]
        self.add_SF(sf)
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
    if self.config.history_size <= 3:
      self.aux_episode_buffer.append([s, s1, a])
    else:
      self.aux_episode_buffer.append([s, s1[:, :, -2:-1], a])

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps),
                    global_step=self.global_step)
    tf.logging.info(
      "Saved Model at {}".format(self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps)))

    if self.config.sr_matrix is not None:
      self.save_SF_matrix()
    if self.config.eigen:
      self.save_eigen_directions()

  def recompute_eigenvectors_dynamic_SVD(self):
    if self.config.eigen:
      feed_dict = {self.local_network.matrix_sf: [self.global_network.sf_matrix_buffer]}
      eigenvect = self.sess.run(self.local_network.eigenvectors,
                                feed_dict=feed_dict)
      eigenvect = eigenvect[0]

      if self.global_network.directions_init:
        new_eigenvectors = self.associate_closest_vectors(self.global_network.directions, eigenvect)
      else:
        new_eigenvectors = eigenvect[
                           self.config.first_eigenoption:(self.config.nb_options // 2) + self.config.first_eigenoption]
        new_eigenvectors = np.concatenate((new_eigenvectors, (-1) * new_eigenvectors))
        self.global_network.directions_init = True

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
    # featured = new[self.config.first_eigenoption: self.config.nb_options + self.config.first_eigenoption]
    featured = new[self.config.first_eigenoption: (self.config.nb_options // 2) + self.config.first_eigenoption]
    featured = np.concatenate((featured, (-1) * featured))


    for d in featured:
      # sign = np.argmax(
      #   [np.sum([np.sign(np.dot(v, x)) * (np.dot(v, x) ** 2) for x in self.global_network.sf_matrix_buffer]),
      #    np.sum([np.sign(np.dot((-1) * v, x)) * (np.dot(v, x) ** 2) for x in self.global_network.sf_matrix_buffer])])
      # if sign == 1:
      #   v = (-1) * v
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

  def save_SF_matrix(self):
    np.save(self.global_network.sf_matrix_path, self.global_network.sf_matrix_buffer)

  def save_eigen_directions(self):
    np.save(self.global_network.directions_path, self.global_network.directions)
