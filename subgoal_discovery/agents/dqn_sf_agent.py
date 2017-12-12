import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount
import os
import matplotlib.patches as patches
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
# import plotly.plotly as py
# import plotly.tools as tls
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import io
import numpy as np
from collections import deque
from PIL import Image
import scipy.stats
import seaborn as sns
from .base_vis_agent import BaseVisAgent
from .dqn_sf_base_agent import DQNSFBaseAgent
sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from auxilary.policy_iteration import PolicyIteration
from tools.schedules import LinearSchedule
FLAGS = tf.app.flags.FLAGS
import random


class DQNSFAgent(DQNSFBaseAgent):
  def __init__(self, game, _, global_step, config, type_of_task):
    DQNSFBaseAgent.__init__(self, game, _, global_step, config, type_of_task)
    self.name = "DQN_agent"
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.episode_max_values = []
    self.episode_min_values = []
    self.episode_mean_returns = []
    self.episode_max_returns = []
    self.episode_min_returns = []
    self.option_episode_buffer = deque()
    self.orig_net = config.network('orig', config, self.action_size, self.nb_states)
    self.target_net = config.network('target', config, self.action_size, self.nb_states)

    self.targetOps = self.update_target_graph('orig', 'target')
    self.batch_generator = self.get_next_batch()
    self.exploration = LinearSchedule(self.config.option_explore_steps, self.config.final_random_action_prob,
                                          self.config.initial_random_action_prob)
    # self.probability_of_random_action = self.exploration.value(0)


  def get_next_batch(self):
    while True:
      index_array = np.arange(self.config.observation_steps)
      np.random.shuffle(index_array)
      self.episode_buffer["observations"] = self.episode_buffer["observations"][index_array]
      # self.episode_buffer["fi"] = self.episode_buffer["fi"][index_array]
      self.episode_buffer["next_observations"] = self.episode_buffer["next_observations"][index_array]
      self.episode_buffer["actions"] = self.episode_buffer["actions"][index_array]
      self.episode_buffer["done"] = self.episode_buffer["done"][index_array]

      for i in range(0, self.config.observation_steps, self.config.batch_size):
          yield [self.episode_buffer["observations"][i:i + self.config.batch_size],
                 # self.episode_buffer["fi"][i:i + self.config.batch_size],
                 self.episode_buffer["next_observations"][i:i + self.config.batch_size],
                 self.episode_buffer["actions"][i:i + self.config.batch_size],
                 self.episode_buffer["done"][i:i + self.config.batch_size]]

  # def show_statistics(self, sess):
  #   matrix_sf = np.zeros((self.nb_states, self.config.sf_layers[-1]))
  #   matrix_fi = np.zeros((self.nb_states, self.config.sf_layers[-1]))
  #   real_states = []
  #   states = []
  #   for idx in range(self.nb_states):
  #     s, ii, jj = self.env.get_state(idx)
  #     if self.env.not_wall(ii, jj):
  #       states.append(s)
  #       real_states.append(True)
  #     else:
  #       real_states.append(False)
  #   feed_dict = {self.orig_net.observation:  np.stack(states, axis=0)}
  #   fi, sf = sess.run([self.orig_net.fi, self.orig_net.sf], feed_dict=feed_dict)
  #   i = 0
  #   for idx in range(self.nb_states):
  #     if real_states[idx]:
  #       matrix_fi[idx], matrix_sf[idx] = fi[i], sf[i]
  #       i += 1
  #
  #   sns.plt.clf()
  #   buf = io.BytesIO()
  #   ax = sns.heatmap(self.matrix_sf, cmap="Blues")
  #   ax.set(xlabel='SR_vect_size=128', ylabel='Grid states/positions')
  #   sns.plt.savefig(buf, format='png')
  #   buf.seek(0)
  #   sns.plt.close()
  #
  #   image = tf.image.decode_png(buf.getvalue(), channels=4)
  #
  #   return matrix_fi, matrix_sf

  def train(self, sess):
    minibatch = self.batch_generator.__next__()
    observations = minibatch[0]
    # fi = minibatch[1]
    next_observations = minibatch[1]
    actions = minibatch[2]
    done = minibatch[3]

    feed_dict = {self.target_net.observation: np.stack(observations, axis=0)}
    feed_dict2 = {self.target_net.observation: np.stack(next_observations, axis=0)}
    fi = sess.run(self.target_net.fi, feed_dict=feed_dict)
    next_sf = sess.run(self.target_net.sf, feed_dict=feed_dict2)
    bootstrap_sf = [np.zeros_like(next_sf[0]) if d else n_sf for d, n_sf in list(zip(done, next_sf))]
    target_sf = fi + self.config.discount * np.asarray(bootstrap_sf)

    feed_dict = {self.orig_net.target_sf: np.stack(target_sf, axis=0),
                 self.orig_net.observation: np.stack(observations, axis=0),
                 self.orig_net.target_next_obs: np.stack(next_observations, axis=0),
                 self.orig_net.actions_placeholder: actions}

    loss, _, ms, sf_loss, aux_loss = sess.run(
      [self.orig_net.loss,
       self.orig_net.apply_grads,
       self.orig_net.merged_summary,
       self.orig_net.sf_loss,
       self.orig_net.aux_loss],
      feed_dict=feed_dict)

    # self.updateTarget()

    return ms, loss, sf_loss, aux_loss

  def train_option(self, sess):
    minibatch = random.sample(self.option_episode_buffer, self.config.option_batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    actions = rollout[:, 1]
    rewards = rollout[:, 2]
    next_observations = rollout[:, 3]
    done = rollout[:, 4]

    feed_dict = {self.target_net.observation: np.stack(next_observations, axis=0)}
    target_q = sess.run(self.target_net.q,
                       feed_dict=feed_dict)
    target_q_max = np.max(target_q, axis=1)
    target_q_a = []

    for i in range(len(done)):
      if done[i]:
        target_q_a.append(rewards[i])
      else:
        target_q_a.append(
          rewards[i] + self.config.discount * target_q_max[i])

    feed_dict = {self.orig_net.target_q_a: target_q_a,
                 self.orig_net.observation: np.stack(observations, axis=0),
                 self.orig_net.actions_placeholder: actions}

    q_loss, _, q_ms = sess.run(
      [self.orig_net.q_loss,
       self.orig_net.q_apply_grads,
       self.orig_net.q_merged_summary],
      feed_dict=feed_dict)

    # self.updateTarget()

    return q_ms, q_loss

  def updateTarget(self, sess):
    for op in self.targetOps:
      sess.run(op)

  def set_evect(self, option, sign):
    eigenvectors_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvectors.npy")
    eigenvalues_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "eigenvalues.npy")
    eigenvectors = np.load(eigenvectors_path)
    eigenvalues = np.load(eigenvalues_path)
    self.evect = eigenvectors[option]
    if sign == "neg":
      self.evect = -self.evect

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      # sess.run(self.global_step.assign(self.config.observation_steps))
      self.total_steps = sess.run(self.global_step)
      if self.total_steps == 0:
        self.updateTarget(sess)

      print("Starting agent")
      s = self.env.reset()

      while self.total_steps < self.config.observation_steps:
        a = self.policy_evaluation(s)

        # feed_dict = {self.orig_net.observation: np.stack([s])}
        # sf, fi = sess.run([self.orig_net.sf, self.orig_net.fi],
        #                   feed_dict=feed_dict)
        # sf, fi = sf[0], fi[0]

        s1, r, d, info = self.env.step(a)

        print(self.total_steps)
        self.episode_buffer["observations"][self.total_steps] = s

        if d:
          s = self.env.reset()
        else:
          s = s1

        self.episode_buffer["next_observations"][self.total_steps] = s
        self.episode_buffer["actions"][self.total_steps] = a
        self.episode_buffer["done"][self.total_steps] = d

        if self.total_steps % self.config.checkpoint_interval == 0 and self.total_steps != 0:
          self.save_buffer()
          self.save_model(sess, saver, self.total_steps)

        self.total_steps += 1
        sess.run(self.increment_global_step)

      while self.total_steps <= self.config.observation_steps + self.config.training_steps:

        if self.total_steps % self.config.target_update_freq == 0:
          self.updateTarget(sess)

        ms, loss, sf_loss, aux_loss = self.train(sess)
        print("Step {} >>> SF_loss {} >>> AUX_loss {} ".format(self.total_steps, sf_loss, aux_loss))

        if self.total_steps % self.config.summary_interval == 0 and self.total_steps > self.config.observation_steps and ms is not None:
          # matrix_fi, matrix_sf = self.show_statistics(sess)
          self.summary_writer.add_summary(ms, self.total_steps)

          self.summary_writer.add_summary(self.summary, self.total_steps)
          self.summary_writer.flush()

        if self.total_steps % self.config.checkpoint_interval == 0 and self.total_steps > self.config.observation_steps:
          self.save_model(sess, saver, self.total_steps)

        self.total_steps += self.config.batch_size
        sess.run(self.increment_batch_global_step)

  def play_option(self, sess, coord, saver, option, sign):
    with sess.as_default(), sess.graph.as_default():
      self.set_evect(option, sign)
      self.total_steps = sess.run(self.global_step)
      if self.total_steps == 0:
        self.updateTarget(sess)

      ms = None

      print("Starting agent")
      while self.total_steps < self.config.observation_steps + \
          self.config.training_steps + \
          self.config.option_steps:
        if self.total_steps % self.config.target_update_freq == 0:
          self.updateTarget(sess)

        q_values = []
        option_episode_reward = 0
        t = 0
        q_values = []
        d = False

        s = self.env.reset()

        while not d:
          a, max_action_values_evaled = self.option_policy_evaluation(s, sess)

          if max_action_values_evaled is not None:
            q_values.append(max_action_values_evaled)

          feed_dict = {self.orig_net.observation: np.stack([s])}
          fi = sess.run(self.orig_net.fi,
                            feed_dict=feed_dict)
          fi = fi[0]

          if a == self.action_size:
            r = 0
            s1 = s
            d = True
          else:
            s1, _, d, _ = self.env.step(a)
            feed_dict = {self.orig_net.observation: np.stack([s1])}
            fi1 = sess.run(self.orig_net.fi,
                          feed_dict=feed_dict)
            fi1 = fi1[0]
            r = self.get_reward(fi, fi1)

          print(self.total_steps)
          self.option_episode_buffer.append([s, a, r, s1, d])
          option_episode_reward += r
          t += 1
          s = s1
          if len(self.option_episode_buffer) == self.config.option_memory_size:
            self.option_episode_buffer.popleft()

          if self.total_steps > self.config.observation_steps +\
              self.config.training_steps +\
              self.config.option_observation_steps and len(
              self.option_episode_buffer) > self.config.option_observation_steps and \
                      self.total_steps % self.config.option_update_freq == 0:
            ms, q_loss = self.train_option(sess)
            print("Step {} >>> Q_loss {}".format(self.total_steps, q_loss))

          if self.total_steps % self.config.option_summary_interval == 0 and\
                  self.total_steps > self.config.observation_steps +\
              self.config.training_steps +\
              self.config.option_observation_steps and ms is not None:
            self.summary_writer.add_summary(ms, self.total_steps)
            self.summary_writer.add_summary(self.summary, self.total_steps)
            self.summary_writer.flush()

          if self.total_steps % self.config.option_checkpoint_interval == 0 and \
                  self.total_steps > self.config.observation_steps +\
              self.config.training_steps +\
              self.config.option_observation_steps:
            self.save_model(sess, saver, self.total_steps)

          self.total_steps += 1
          sess.run(self.increment_global_step)


  def policy_evaluation(self, s):
    a = np.random.choice(range(self.action_size))
    return a

  def get_reward(self, fi, fi1):
      state_dif = fi1 - fi
      state_dif_norm = np.linalg.norm(fi1 - fi)
      state_dif_normalized = state_dif / (state_dif_norm + 1e-8)

      return np.dot(state_dif_normalized, self.evect)

  def option_policy_evaluation(self, s, sess):
    q = None
    self.probability_of_random_action = self.exploration.value(self.total_steps -
                                                               self.config.observation_steps -
                                                               self.config.training_steps)
    if random.random() <= self.probability_of_random_action:
      a = np.random.choice(range(self.action_size + 1))
    else:
      feed_dict = {self.orig_net.observation: [s]}
      q = sess.run(self.orig_net.q, feed_dict=feed_dict)[0]

      a = np.argmax(q)

    return a, np.max(q)

  def option_policy_evaluation_eval(self, s, sess):
    feed_dict = {self.orig_net.observation: [s]}
    q = sess.run(self.orig_net.q, feed_dict=feed_dict)[0]

    a = np.argmax(q)

    return a
