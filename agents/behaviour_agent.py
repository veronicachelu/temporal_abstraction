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


class BehaviourAgent(BaseAgent):
  def __init__(self, game, thread_id, global_step, config, global_network, barrier):
    super(BehaviourAgent, self).__init__(game, thread_id, global_step, config, global_network)
    self.barrier = barrier

  def init_play(self, sess, saver):
    self.sess = sess
    self.saver = saver
    self.episode_count = sess.run(self.global_step)

    if self.config.move_goal_nb_of_ep and self.config.multi_task:
      self.goal_position = self.env.set_goal(self.episode_count, self.config.move_goal_nb_of_ep)

    self.total_steps = sess.run(self.total_steps_tensor)
    tf.logging.info("Starting worker " + str(self.thread_id))
    self.behaviour_episode_buffer = deque()
    self.ms_aux = self.ms_sf = None

  def init_episode(self):
    self.done = False
    self.episode_len = 0

  def sync_threads(self, force=False):
    if force:
      self.sess.run(self.update_local_vars_aux)
      self.sess.run(self.update_local_vars_sf)
    else:
      if self.total_steps % self.config.target_update_iter_aux_behaviour == 0:
        self.sess.run(self.update_local_vars_aux)
      if self.total_steps % self.config.target_update_iter_sf_behaviour == 0:
        self.sess.run(self.update_local_vars_sf)

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.init_play(sess, saver)

      while not coord.should_stop():
        self.sync_threads()

        if self.episode_count > 0:
          self.recompute_eigenvectors_SVD()

        self.init_episode()

        s, s_idx = self.env.reset()
        self.option_evaluation(s, s_idx)
        while not self.done:
          self.sync_threads()

          self.policy_evaluation(s)

          s1, r, self.done, s1_idx = self.env.step(self.action)

          if self.done:
            s1 = s

          if self.total_steps > self.config.observation_steps:
            self.old_option = self.option

            self.o_term = np.random.uniform() > 0.5

            if not self.done and self.o_term:
              self.option_evaluation(s1, s1_idx)

            self.store_general_info(s, self.old_option, s1, self.option, self.action, r, self.done)
            if len(self.behaviour_episode_buffer) > self.config.observation_steps and \
                        self.total_steps % self.config.behaviour_update_freq == 0:
              self.ms_aux, self.aux_loss, self.ms_sf, self.sf_loss = self.train()

            if self.total_steps % self.config.steps_summary_interval == 0:
              self.write_step_summary(self.ms_sf, self.ms_aux)

          s = s1
          self.episode_len += 1
          self.total_steps += 1
          sess.run(self.increment_total_steps_tensor)

        if self.episode_count % self.config.move_goal_nb_of_ep == 0 and self.episode_count != 0:
          tf.logging.info("Moving GOAL....")
          self.barrier.wait()
          self.goal_position = self.env.set_goal(self.episode_count, self.config.move_goal_nb_of_ep)

        if self.episode_count % self.config.episode_summary_interval == 0 and self.total_steps != 0 and self.episode_count != 0:
          self.write_episode_summary(self.ms_sf, self.ms_aux)

        self.episode_count += 1

  def option_evaluation(self, s, s_idx):
    self.option = np.random.choice(range(self.nb_options))

  def policy_evaluation(self, s):
    self.action = np.random.choice(range(self.action_size))

  def store_general_info(self, s, o, s1, o1, a, r, d):
    if len(self.behaviour_episode_buffer) == self.config.memory_size:
      self.behaviour_episode_buffer.popleft()

    self.behaviour_episode_buffer.append([s, o, s1, o1, a, r, d])

  def write_step_summary(self, ms_sf, ms_aux):
    self.summary = tf.Summary()
    if ms_sf is not None:
      self.summary_writer.add_summary(ms_sf, self.total_steps)
    if ms_aux is not None:
      self.summary_writer.add_summary(ms_aux, self.total_steps)

    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()

  def write_episode_summary(self, ms_sf, ms_aux):
    self.summary = tf.Summary()
    self.summary.value.add(tag='Perf/Goal_position', simple_value=self.goal_position)

    self.summary_writer.add_summary(self.summary, self.episode_count)
    self.summary_writer.flush()
    self.write_step_summary(ms_sf, ms_aux)

  # def recompute_eigenvectors_SVD(self, plotting=False):
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

  def recompute_eigenvectors_SVD(self):
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

  def train(self):
    minibatch = random.sample(self.behaviour_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    options = rollout[:, 1]
    next_observations = rollout[:, 2]
    next_options = rollout[:, 3]
    actions = rollout[:, 4]
    rewards = rollout[:, 5]
    done = rollout[:, 6]

    feed_dict = {self.global_network.observation: np.stack(observations, axis=0)}
    fi = self.sess.run(self.global_network.fi, feed_dict=feed_dict)

    feed_dict2 = {self.global_network.observation: np.stack(next_observations, axis=0),
                  self.global_network.options_placeholder: np.stack(next_options, axis=0)}
    # next_sf = self.sess.run(self.global_network.sf_o, feed_dict=feed_dict2)
    next_sf = self.sess.run(self.global_network.sf, feed_dict=feed_dict2)

    bootstrap_sf = [np.zeros_like(next_sf[0]) if d else n_sf for d, n_sf in list(zip(done, next_sf))]
    target_sf = fi + self.config.discount * np.asarray(bootstrap_sf)

    feed_dict = {
                self.local_network.options_placeholder: np.stack(options, axis=0),
                self.local_network.target_sf: np.stack(target_sf, axis=0),
                self.local_network.observation: np.stack(observations, axis=0),
                self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                self.local_network.actions_placeholder: actions}

    ms_aux, aux_loss, _, ms_sf, sf_loss, _ = \
      self.sess.run([self.local_network.merged_summary_aux,
                     self.local_network.aux_loss,
                     self.local_network.apply_grads_aux,
                     self.local_network.merged_summary_sf,
                     self.local_network.sf_loss,
                     self.local_network.apply_grads_sf],
                    feed_dict=feed_dict)
    return ms_aux, aux_loss, ms_sf, sf_loss
