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
from agents.eigenoc_agent import EigenOCAgent
import copy
from threading import Barrier, Thread

FLAGS = tf.app.flags.FLAGS

"""This Agent is a specialization of the successor representation direction based agent, but one which builds the SR matrix as a buffer adding transitions to it while it explores"""
class EigenOCAgentDyn(EigenOCAgent):
  def __init__(self, sess, game, thread_id, global_step, global_episode, config, global_network, barrier):
    super(EigenOCAgentDyn, self).__init__(sess, game, thread_id, global_step, global_episode, config, global_network, barrier)

  def add_SF(self, sf):
    """Add the successor representation of the current state tot the SR buffer matrix"""
    self.global_network.sf_matrix_buffer[0] = sf.copy()
    self.global_network.sf_matrix_buffer = np.roll(self.global_network.sf_matrix_buffer, 1, 0)

  """Sample an action from the current option's policy"""
  def policy_evaluation(self, s):
    feed_dict = {self.local_network.observation: [s]}
    """If we use eigendirections as basis for the options"""
    if self.config.eigen:
      tensor_list = [self.local_network.sf,
                     self.local_network.options,
                     self.local_network.v,
                     self.local_network.q_val,
                     self.local_network.eigen_q_val,
                     self.local_network.eigenv]
      sf,\
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

      sf = sf[0]
      self.add_SF(sf)
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

  def save_model(self):
    super(EigenOCAgentDyn, self).save_model()

    if self.config.sr_matrix is not None:
      self.save_SF_matrix()
    if self.config.use_eigendirections:
      self.save_eigen_directions()

  """Redo the SVD decomposition of the successor representation buffer matrix"""
  def recompute_eigendirections(self):
    if self.name == "worker_0" and self.global_episode_np > 0 and \
        self.config.use_eigendirections:
      self.recompute_eigenvectors_svd()

  "Do SVD decomposition on the SR matrix buffer"
  def recompute_eigenvectors_svd(self):
    """Keep track of the eigendirection before the update"""
    old_directions = self.global_network.directions

    """Do SVD decomposition"""
    feed_dict = {self.local_network.matrix_sf: [self.global_network.sf_matrix_buffer]}
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
    self.summary_writer.add_summary(self.summary, self.global_episode_np)
    self.summary_writer.flush()

  def save_SF_matrix(self):
    np.save(self.global_network.sf_matrix_path, self.global_network.sf_matrix_buffer)

  def save_eigen_directions(self):
    np.save(self.global_network.directions_path, self.global_network.directions)
