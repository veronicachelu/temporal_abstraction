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
  def __init__(self, game, thread_id, global_step, config, global_network, barrier):
    super(EigenOCAgentDyn, self).__init__(game, thread_id, global_step, config, global_network, barrier)
    # self.sf_matrix_path = os.path.join(config.logdir, "sf_matrix.npy")
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
            # else:
            #   self.recompute_eigenvectors_dynamic_NN()

          if self.config.sr_matrix is not None:
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
            self.log_timestep()

            if self.total_steps > self.config.observation_steps:
              if self.config.behaviour_agent is None and self.config.eigen:
                self.SF_prediction(s1)
              self.next_frame_prediction()

              if self.total_steps > self.config.eigen_exploration_steps:
                self.option_prediction(s, s1, r)

                if not self.done and (self.o_term or self.primitive_action):
                  # if not self.primitive_action:
                  #   self.episode_options_lengths[self.option][-1] = self.episode_len - \
                  #                                                   self.episode_options_lengths[self.option][-1]
                  self.option_evaluation(s1)

              if self.total_steps % self.config.steps_checkpoint_interval == 0 and self.name == 'worker_0':
                self.save_model()

              if self.total_steps % self.config.steps_summary_interval == 0 and self.name == 'worker_0':
                self.write_step_summary(self.ms_sf, self.ms_aux, self.ms_option, r)

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
                  self.episode_count != 0 and self.config.multi_task:
            tf.logging.info("Moving GOAL....")
            self.barrier.wait()
            self.goal_position = self.env.set_goal(self.episode_count, self.config.move_goal_nb_of_ep)

          if self.episode_count % self.config.episode_checkpoint_interval == 0 and self.name == 'worker_0' and \
                  self.episode_count != 0:
            self.save_model()

          if self.episode_count % self.config.episode_summary_interval == 0 and self.total_steps != 0 and \
                  self.name == 'worker_0' and self.episode_count != 0:
            self.write_episode_summary(self.ms_sf, self.ms_aux, self.ms_option, r)

          if self.name == 'worker_0':
            sess.run(self.increment_global_step)
          self.episode_count += 1

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
    if self.total_steps > self.config.eigen_exploration_steps:
      feed_dict = {self.local_network.observation: np.stack([s])}

      if self.config.eigen:
        tensor_list = [self.local_network.sf, self.local_network.options, self.local_network.v, self.local_network.q_val,
                       self.local_network.termination, self.local_network.eigen_q_val, self.local_network.eigenv]
        sf, options, value, q_value, o_term, eigen_q_value, evalue = self.sess.run(tensor_list, feed_dict=feed_dict)
        if not self.primitive_action:
          self.eigen_q_value = eigen_q_value[0, self.option]
          pi = options[0, self.option]
          self.action = np.random.choice(pi, p=pi)
          self.action = np.argmax(pi == self.action)
          self.o_term = o_term[0, self.option] > np.random.uniform()
          self.evalue = evalue[0]
        else:
          self.action = self.option - self.nb_options
          self.o_term = True
        self.q_value = q_value[0, self.option]
        self.value = value[0]

        if self.config.behaviour_agent is None:
          sf = sf[0]
          self.add_SF(sf)
      else:
        tensor_list = [self.local_network.options, self.local_network.v, self.local_network.q_val,
                       self.local_network.termination]
        options, value, q_value, o_term = self.sess.run(tensor_list, feed_dict=feed_dict)

        if self.config.include_primitive_options and self.primitive_action:
          self.action = self.option - self.nb_options
          self.o_term = True
        else:
          pi = options[0, self.option]
          self.action = np.random.choice(pi, p=pi)
          self.action = np.argmax(pi == self.action)
          self.o_term = o_term[0, self.option] > np.random.uniform()
        self.q_value = q_value[0, self.option]
        self.value = value[0]

    else:
      self.action = np.random.choice(range(self.action_size))
    self.episode_actions.append(self.action)

  def store_general_info(self, s, s1, a, r):
    if self.config.eigen:
      self.episode_buffer_sf.append([s, s1, a])
    if len(self.aux_episode_buffer) == self.config.memory_size:
      self.aux_episode_buffer.popleft()
    if self.config.history_size == 3:
      self.aux_episode_buffer.append([s, s1, a])
    else:
      self.aux_episode_buffer.append([s, s1[:, :, -2:-1], a])
    self.episode_reward += r

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
      sign = np.argmax([np.sum([np.sign(np.dot(v, x)) * (np.dot(v, x)**2) for x in self.global_network.sf_matrix_buffer]),
       np.sum([np.sign(np.dot((-1) * v, x)) * (np.dot(v, x)**2) for x in self.global_network.sf_matrix_buffer])])
      if sign == 1:
        v = (-1) * v
      distances = [-np.inf if b_idx in skip_list else (np.inf if np.all(b == np.zeros(b.shape)) else self.cosine_similarity(v, b)) for b_idx, b in enumerate(self.global_network.directions)]
      closest_distance_idx = np.argmax(distances)

      old_v_idx = closest_distance_idx
      skip_list.append(old_v_idx)

      to_return[old_v_idx] = v

    return to_return


  # def recompute_eigenvectors_dynamic_NN(self):
  #   if self.config.eigen:
  #     feed_dict = {self.local_network.matrix_sf: [self.global_network.sf_matrix_buffer]}
  #     eigenvect = self.sess.run(self.local_network.eigenvectors,
  #                                         feed_dict=feed_dict)
  #     eigenvect = eigenvect[0]
  #
  #     # eigenvalues = eigenval[self.config.first_eigenoption:self.config.nb_options + self.config.first_eigenoption]
  #     new_eigenvectors = eigenvect[self.config.first_eigenoption:self.config.nb_options + self.config.first_eigenoption]
  #     min_similarity = np.min(
  #       [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
  #     max_similarity = np.max(
  #       [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
  #     mean_similarity = np.mean(
  #       [self.cosine_similarity(a, b) for a, b in zip(self.global_network.directions, new_eigenvectors)])
  #     self.summary = tf.Summary()
  #     self.summary.value.add(tag='Eigenvectors/Min similarity', simple_value=float(min_similarity))
  #     self.summary.value.add(tag='Eigenvectors/Max similarity', simple_value=float(max_similarity))
  #     self.summary.value.add(tag='Eigenvectors/Mean similarity', simple_value=float(mean_similarity))
  #     self.summary_writer.add_summary(self.summary, self.episode_count)
  #     self.summary_writer.flush()
  #     self.global_network.directions = new_eigenvectors
  #     self.directions = self.global_network.directions


  def save_SF_matrix(self):
    np.save(self.global_network.sf_matrix_path, self.global_network.sf_matrix_buffer)

  def save_eigen_directions(self):
    np.save(self.global_network.directions_path, self.global_network.directions)
