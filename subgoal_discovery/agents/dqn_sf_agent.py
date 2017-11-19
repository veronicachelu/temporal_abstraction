import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
import matplotlib.patches as patches
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
# import plotly.plotly as py
# import plotly.tools as tls
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
FLAGS = tf.app.flags.FLAGS
FLAGS = tf.app.flags.FLAGS
import random



class DQNSFAgent(DQNSFBaseAgent):
  def __init__(self, game, _, global_step, config):
    DQNSFBaseAgent.__init__(self, game, _, global_step, config)
    self.name = "DQN_agent"
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.episode_max_values = []
    self.episode_min_values = []
    self.episode_mean_returns = []
    self.episode_max_returns = []
    self.episode_min_returns = []

    self.orig_net = config.network('orig', config, self.action_size, self.nb_states)
    self.target_net = config.network('target', config, self.action_size, self.nb_states)

    self.targetOps = self.update_target_graph('orig', 'target')


  def get_next_batch(self):
    while True:
      for i in range(0, len(self.episode_buffer), self.config.batch_size):
          yield self.episode_buffer[i:i + self.config.batch_size]

  def train(self, sess):
    minibatch = self.get_next_batch()
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    fi = rollout[:, 1]
    next_observations = rollout[:, 2]
    actions = rollout[:, 3]
    done = rollout[:, 4]

    feed_dict = {self.target_net.observation: np.stack(next_observations, axis=0)}
    next_sf = sess.run(self.target_net.sf,
                       feed_dict=feed_dict)[0]
    target_sf = []

    for i in range(len(done)):
      if done[i]:
        target_sf.append(fi[i])
      else:
        target_sf.append(
          fi[i] + self.config.gamma * next_sf[i])

    feed_dict = {self.orig_net.target_sf: target_sf,
                 self.orig_net.observation: np.stack(observations, axis=0),
                 self.orig_net.target_next_obs: np.stack(next_observations, axis=0),
                 self.orig_net.actions_placeholder: actions}

    _, ms, loss, sf_loss, aux_loss = sess.run(
      [self.orig_net.loss,
       self.orig_net.apply_grads,
       self.orig_net.merged_summary,
       self.orig_net.sf_loss,
       self.orig_net.aux_loss],
      feed_dict=feed_dict)

    # self.updateTarget()

    return ms, loss, sf_loss, aux_loss

  def updateTarget(self, sess):
    for op in self.targetOps:
      sess.run(op)

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.total_steps = sess.run(self.global_step)
      if self.total_steps == 0:
        self.updateTarget(sess)

      print("Starting agent")
      s = self.env.reset()

      while self.total_steps < self.config.observation_steps:
        a = self.policy_evaluation(s)

        feed_dict = {self.orig_net.observation: np.stack([s])}
        sf, fi = sess.run([self.orig_net.sf, self.orig_net.fi],
                          feed_dict=feed_dict)
        sf, fi = sf[0], fi[0]

        s1, r, d, info = self.env.step(a)

        print(self.total_steps)
        self.total_steps += 1

        self.episode_buffer["observations"][self.buf_counter] = s
        self.episode_buffer["fi"][self.buf_counter] = fi
        self.episode_buffer["next_observations"][self.buf_counter] = s1
        self.episode_buffer["actions"][self.buf_counter] = a
        self.episode_buffer["done"][self.buf_counter] = d
        self.episode_buffer["counter"] += 1
        self.buf_counter += 1

        if self.total_steps % self.config.checkpoint_interval == 0 and self.total_steps != 0:
          self.save_buffer()

        s = s1

      while self.total_steps < self.config.observation_steps + self.config.training_steps:

        if self.total_steps % self.config.target_update_freq == 0:
          self.updateTarget(sess)

        ms, loss, sf_loss, aux_loss = self.train(sess)
        print("Step {} >>> SF_loss {} >>> AUX_loss {} ".format(self.total_steps, sf_loss, aux_loss))

        if self.total_steps % FLAGS.summary_interval == 0 and self.total_steps != 0 and ms is not None:
          self.summary_writer.add_summary(ms, self.total_steps)

          self.summary_writer.add_summary(self.summary, self.total_steps)
          self.summary_writer.flush()

        if self.total_steps % FLAGS.checkpoint_interval == 0 and self.total_steps != 0:
          self.save_model(sess, saver, self.total_steps)
        self.total_steps += 100
        self.sess.run(self.increment_batch_global_step)


  def policy_evaluation(self, s):
    a = np.random.choice(range(self.action_size))
    return a

