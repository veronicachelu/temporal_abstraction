import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from collections import deque
from agents.schedules import LinearSchedule, TFLinearSchedule
from PIL import Image
import scipy.stats
import random
import math

FLAGS = tf.app.flags.FLAGS


class TabSFAgent():
  def __init__(self, game, thread_id, config):
    self.name = "worker_" + str(thread_id)
    self.model_path = os.path.join(config.stage_logdir, "models")
    tf.gfile.MakeDirs(self.model_path)
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.episode_mean_q_values = []
    self.episode_mean_returns = []
    self.episode_mean_oterms = []
    self.episode_mean_options = []
    self.episode_options = []
    self.config = config
    self.total_steps = 0
    self.theta = 0.0001
    self.nb_states = game.nb_states
    self.action_size = game.action_space.n
    self.sf = np.eye((game.nb_states))

    self.env = game

  def train(self, rollout, sess, bootstrap_sf, summaries=False):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    actions = rollout[:, 1]
    sf = rollout[:, 2]
    fi = rollout[:, 3]

    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions}

    _, ms, img_summ, loss, sf_loss = \
      sess.run([self.local_network.apply_grads,
                self.local_network.merged_summary,
                self.local_network.image_summaries,
                self.local_network.loss,
                self.local_network.sf_loss],
               feed_dict=feed_dict)
    return ms, img_summ, loss, sf_loss

  def sf_eval(self):
    ''' Policy evaluation step.'''
    delta = 0.0
    for s in range(self.nb_states):
      sf = self.sf[s]
      a = np.random.choice(range(self.action_size))
      nextS, nextR = self.env.get_next_state_and_reward(
        s, a)
      state_features = np.identity(self.nb_states)
      self.sf[s] = state_features[s] + self.config.gamma * self.sf[nextS]
      delta = max(delta, math.fabs(sf - self.sf[s]))

    return delta

  def play(self, sess, coord, saver):
    # with sess.as_default(), sess.graph.as_default():
    #   episode_count = 0
    #   self.total_steps = 0
    # 
    #   while not coord.should_stop():
    #     if episode_count > self.config.steps:
    #       return 0
    # 
    #     episode_reward = 0
    #     d = False
    #     t = 0
    #     t_counter = 0
    #     R = 0
    #     old_sf = None
    delta = np.inf
    while (self.theta < delta):
      delta = self.sf_eval()

    u, s, v = np.linalg.svd(self.sf)
    print(s)



        #
        # while not sf_stable:
        #   a = self.policy_evaluation(s)
        #   s1, r, d, _ = self.env.step(a)
        #
        #   delta = self._evalSF(s, a, r, s1)
        #   while (theta < delta):
        #     delta = self._evalPolicy()
        #
        #   # Policy improvement
        #   policy_stable = self._improvePolicy()
        #
        # while not d:
        #   a, sf = self.policy_evaluation(s)
        #   s1, r, d, _ = self.env.step(a)
        #   r = np.clip(r, -1, 1)
        #
        #   self.total_steps += 1
        #
        #   for idx in range(self.nb_states):
        #     s, i, j = self.env.get_state(idx)
        #     a, sf = self.policy_evaluation(s)
        #     s1, r, d, _ = self.env.step(a)
        #
        #     delta = max(delta, np.abs(s + self.config.gamma * sf[s1] - sf[s]))
        #     # Update the value function
        #     V[s] = best_action_value
        #     # Check if we can stop
        #   if delta < self.theta:
        #     break
        #
        #
        #   episode_reward += r
        #   t += 1
        #   s = s1
        #
        #   if self.name == "worker_0":
        #     print("Episode {} >>> Step {} >>> Length: {} >>> Reward: {}".format(episode_count, self.total_steps, t, episode_reward))
        # self.episode_rewards.append(episode_reward)
        # self.episode_lengths.append(t)
        #
        # episode_count += 1

  # def policy_evaluation(self, s):
  #   a = np.random.choice(range(self.nb_actions))


  def build_sf_matrix(self, sf_old, sf_new):
    if len(self.sf_transition_matrix) == self.config.sf_transition_matrix_size:
      print("Matrix is ready")
      self.task = 3
      self.sf_transition_matrix.popleft()

    self.sf_transition_matrix.append(sf_new - sf_old)

