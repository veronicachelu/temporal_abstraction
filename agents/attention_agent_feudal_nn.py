import numpy as np
import tensorflow as tf
from tools.agent_utils import get_mode, update_target_graph_aux, update_target_graph_sf, \
  update_target_graph_option, discount, reward_discount, set_image, make_gif, set_image_plain
import os
from auxilary.policy_iteration import PolicyIteration
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
sns.set()
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from agents.eigenoc_agent_dynamic import EigenOCAgentDyn
import pickle
import copy
from threading import Barrier, Thread

FLAGS = tf.app.flags.FLAGS

class AttentionFeudalNNAgent(EigenOCAgentDyn):
  def __init__(self, sess, game, thread_id, global_step, global_episode, config, global_network, barrier):
    super(AttentionFeudalNNAgent, self).__init__(sess, game, thread_id, global_step, global_episode, config, global_network, barrier)
    self.episode_mean_values_mix = []

  def init_episode(self):
    super(AttentionFeudalNNAgent, self).init_episode()
    self.episode_values_mix = []
    self.episode_buffer_option = []
    self.episode_goals = []
    self.episode_g_sums = []
    self.episode_clusters = []
    self.states = []
    self.episode_length = 0
    self.reward = 0
    self.action = 1
    self.episode_state_occupancy = np.zeros((self.nb_states))
    self.episode_goal_occupancy = np.zeros((self.nb_states))
    self.summaries_critic = self.summaries_option = self.summaries_aux = self.summaries_sf = self.summaries_term = self.summaries_goal = None
    self.R = self.R_mix = None
    self.last_c_g = np.zeros((1, self.config.c, self.config.sf_layers[-1]), np.float32)
    self.last_batch_done = True
    # self.state = self.local_network.worker_lstm.state_init + self.local_network.manager_lstm.state_init

  def init_agent(self):
    super(AttentionFeudalNNAgent, self).init_agent()

    self.clusters_folder = os.path.join(self.summary_path, "clusters")
    tf.gfile.MakeDirs(self.clusters_folder)

    self.policy_folder = os.path.join(self.summary_path, "policies_clusters")
    tf.gfile.MakeDirs(self.policy_folder)

    self.learning_progress_folder = os.path.join(self.summary_path, "learning_progress")
    tf.gfile.MakeDirs(self.learning_progress_folder)

    self.cluster_model_path = os.path.join(self.config.logdir, "cluster_models")
    tf.gfile.MakeDirs(self.cluster_model_path)

    self.total_episodes = self.global_episode.eval()

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
          self.s = self.env.reset()
          """While the episode does not terminate"""
          while not self.done:
            """update local network parameters from global network"""
            self.sync_threads()

            """Choose an action from the current intra-option policy"""
            self.policy_evaluation(self.s)
            self.s1, self.reward, self.done, self.s1_idx = self.env.step(self.action)
            self.episode_state_occupancy[self.s1_idx] += 1
            if self.s_idx is not None:
              self.episode_goal_occupancy[self.s_idx] = self.which_goal + 1
            self.episode_reward += self.reward

            if self.done:
              self.s1 = self.s
              self.s1_idx = self.s_idx

            if len(self.aux_episode_buffer) == self.config.memory_size:
              self.aux_episode_buffer.popleft()
            self.aux_episode_buffer.append([self.s, self.s1, self.action])

            self.next_frame_prediction()
            if len(self.aux_episode_buffer) > self.config.observation_steps:
              self.episode_buffer_sf.append([self.s, self.fi, self.s1])
              self.sf_prediction(self.s1)

              if self.global_episode_np >= self.config.cold_start_episodes:
                self.option_prediction(self.s, self.s1)

            if self.total_steps % self.config.step_summary_interval == 0 and self.name == 'worker_0':
              self.write_step_summary()

            self.s = self.s1
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

            # if self.global_episode_np % self.config.cluster_interval == 0:

          """If it's time to change the task - move the goal, wait for all other threads to finish the current task"""
          if self.total_episodes % self.config.move_goal_nb_of_ep == 0 and \
                  self.total_episodes != 0:
            tf.logging.info(f"Moving GOAL....{self.total_episodes}")

            if self.name == "worker_0":
              self.print_g()

            self.barrier.wait()
            self.goal_position = self.env.set_goal(self.total_episodes, self.config.move_goal_nb_of_ep)

            # goalstateIdx = self.env.get_state_index(self.env.goalX, self.env.goalY)
            # self.goal_sf = self.sess.run(self.local_network.sf, {
            #   self.local_network.observation: np.identity(self.nb_states)[goalstateIdx:goalstateIdx + 1]})[0]
          self.total_episodes += 1


  """Sample an action from the current option's policy"""
  def policy_evaluation(self, s):
    self.current_clusters = self.global_network.goal_clusters.get_clusters()
    feed_dict = {self.local_network.observation: [s],
                 self.local_network.goal_sr_clusters: [self.current_clusters],
                 self.local_network.prev_goals: self.last_c_g,
                 }

    tensor_results = {
      "g": self.local_network.g,
      "last_c_goals": self.local_network.last_c_g,
      "query_goal": self.local_network.query_goal,
      "attention_weights": self.local_network.attention_weights,
      "query_content_match": self.local_network.query_content_match,
      "which_goal": self.local_network.which_goal,
      "v": self.local_network.v_ext,
      "v_mix": self.local_network.v_mix,
      "sf": self.local_network.sf,
      "g_policy": self.local_network.g_policy,
      "g_sum": self.local_network.g_sum,
      "fi": self.local_network.fi,
      "random_goal_cond": self.local_network.random_goal_cond,
      "global_episode": self.global_episode}

    results = self.sess.run(tensor_results, feed_dict=feed_dict)

    self.g = results["g"][0]
    self.which_goal = results["which_goal"][0]
    self.which_random_goal = results["which_random_goal"][0]
    self.fi = results["fi"][0]
    self.g_sum = results["g_sum"][0]
    self.last_c_g = results["last_c_goals"]
    self.query_goal = results["query_goal"][0]
    self.attention_weights = results["attention_weights"][0]
    self.query_content_match = results["query_content_match"][0]
    self.v = results["v"][0]
    self.v_mix = results["v_mix"][0]
    self.sf = results["sf"][0]
    self.add_SF(self.sf)
    self.global_episode_np = results["global_episode"]
    self.random_goal_cond = results["random_goal_cond"][0]
    pi = results["g_policy"][0]

    """Sample an action"""
    self.action = np.random.choice(pi, p=pi)
    self.action = np.argmax(pi == self.action)
    if self.global_episode_np < self.config.cold_start_episodes:
      self.action = np.random.choice(range(self.action_size))

    which_goal = self.which_goal if self.random_goal_cond else self.which_random_goal
    if self.name == "worker_0" and self.global_step_np % 10:
      print(f"Deterministic goal: {self.random_goal_cond} >>"
          f" Chosen goal: {which_goal} >> Random action {self.global_episode_np < self.config.cold_start_episodes} >> Chosen action {self.action}")
    """Store information in buffers for stats in tensorboard"""
    self.episode_actions.append(self.action)

  """Do n-step prediction for the returns and update the option policies and critics"""
  def option_prediction(self, s, s1):
    """Adding to the transition buffer for doing n-step prediction on critics and policies"""
    self.episode_buffer_option.append(
      [s, self.action, self.reward, s1, self.fi, self.random_goal_cond])
    self.episode_goals.append(self.g)
    self.episode_g_sums.append(self.g_sum)
    self.episode_clusters.append(self.current_clusters)

    if len(self.episode_buffer_option) >= self.config.max_update_freq or self.done:
      """Get the bootstrap option-value functions for the next time step"""
      if self.done:
        bootstrap_V_mix = 0
        bootstrap_V_ext = 0
      else:
        feed_dict = {self.local_network.observation: [s1],
                     self.local_network.goal_sr_clusters: [self.current_clusters], #self.global_network.goal_clusters.get_clusters(),
                     self.local_network.prev_goals: self.last_c_g,
                     }
        to_run = {"v_mix": self.local_network.v_mix,
                  "v_ext": self.local_network.v_ext}
        results = self.sess.run(to_run, feed_dict=feed_dict)
        v_mix, v = results["v_mix"][0], results["v_ext"][0]
        bootstrap_V_mix = v_mix
        bootstrap_V_ext = v

      self.train_goal(bootstrap_V_mix, bootstrap_V_ext, s1)
      if self.done:
        self.last_batch_done = True
      else:
        twoc = 2 * self.config.c
        self.episode_buffer_option = self.episode_buffer_option[-twoc:]
        self.episode_goals = self.episode_goals[-twoc:]
        self.episode_g_sums = self.episode_g_sums[-twoc:]
        self.episode_clusters = self.episode_clusters[-twoc:]


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
    fi = rollout[:, 1]

    """Get the latent representations for each state"""
    feed_dict = {self.local_network.observation: np.stack(observations, axis=0)}
    fi = self.sess.run(self.local_network.fi,
                       feed_dict=feed_dict)

    """Construct list of latent representations for the entire trajectory"""
    sf_plus = np.asarray(fi.tolist() + [bootstrap_sf])
    """Construct the targets for the next step successor representations for the entire trajectory"""
    discounted_sf = discount(sf_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_sf: np.stack(discounted_sf, axis=0),
                 self.local_network.observation: np.stack(observations, axis=0)}

    to_run = {"summary_sf": self.local_network.merged_summary_sf,
              "sf_loss": self.local_network.sf_loss
    }
    if self.name != "worker_0":
      to_run["apply_grads_sf"] = self.local_network.apply_grads_sf
    results = self.sess.run(to_run, feed_dict=feed_dict)
    self.summaries_sf = results["summary_sf"]

  def train_aux(self):
    minibatch = random.sample(self.aux_episode_buffer, self.config.batch_size)
    rollout = np.array(minibatch)
    observations = rollout[:, 0]
    next_observations = rollout[:, 1]
    actions = rollout[:, 2]

    feed_dict = {self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_next_obs: np.stack(next_observations, axis=0),
                 self.local_network.actions_placeholder: actions}
    to_run = {"summary_aux": self.local_network.merged_summary_aux,
              "aux_loss": self.local_network.aux_loss
              }
    if self.name != "worker_0":
      to_run["apply_grads_aux"] = self.local_network.apply_grads_aux
    results = self.sess.run(to_run, feed_dict=feed_dict)
    self.summaries_aux = results["summary_aux"]

  def next_frame_prediction(self):
    if len(self.aux_episode_buffer) > self.config.observation_steps and \
                self.total_steps % self.config.aux_update_freq == 0:
      self.train_aux()

  def add_SF(self, sf):
    self.global_network.goal_clusters.cluster(sf)

  def extend(self, batch):
    (observations, fi, actions, rewards, discounted_returns, goals, g_sums, random_goal_cond, current_clusters) = batch
    new_observations, new_fi, new_actions, new_rewards, new_discounted_returns, new_goals, new_g_sums, new_random_goal_cond, new_current_clusters = \
      [], [], [], [], [], [], [], [], []
    if self.last_batch_done:
      new_fi = [fi[0] for _ in range(self.config.c)]
      new_goals = [goals[0] for _ in range(self.config.c)]
      new_actions = [None for _ in range(self.config.c)]
      new_discounted_returns = [None for _ in range(self.config.c)]
      new_rewards = [None for _ in range(self.config.c)]
      new_g_sums = [None for _ in range(self.config.c)]
      new_observations = [None for _ in range(self.config.c)]
      new_random_goal_cond = [None for _ in range(self.config.c)]
      new_current_clusters = [None for _ in range(self.config.c)]

    # extend with the actual values
    new_observations.extend(observations)
    new_fi.extend(fi)
    new_goals.extend(goals)
    new_rewards.extend(rewards)
    new_discounted_returns.extend(discounted_returns)
    new_actions.extend(actions)
    new_g_sums.extend(g_sums)
    new_random_goal_cond.extend(random_goal_cond)
    new_current_clusters.extend(current_clusters)

    # if this is a terminal batch, then append the final s and g c times
    # note that both this and the above case can occur at the same time
    if self.done:
      new_fi.extend([fi[-1] for _ in range(self.config.c)])
      new_goals.extend([goals[-1] for _ in range(self.config.c)])

    return new_observations, new_fi, new_actions, new_rewards, new_discounted_returns, new_goals, new_g_sums, \
           new_random_goal_cond, new_current_clusters

  """Do n-step prediction on the critics and policies"""
  def train_goal(self, bootstrap_value_mix, bootstrap_value_ext, s1):
    rollout = np.array(self.episode_buffer_option)
    observations = rollout[:, 0]
    actions = rollout[:, 1]
    rewards = rollout[:, 2]
    fi = rollout[:, 4]
    random_goal_cond = rollout[:, 5]
    goals = self.episode_goals
    g_sums = self.episode_g_sums
    current_clusters = self.episode_clusters
    """Construct list of discounted returns using mixed reward signals for the entire n-step trajectory"""
    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value_ext])
    discounted_returns = reward_discount(rewards_plus, self.config.discount)[:-1]

    batch = (observations, fi, actions, rewards, discounted_returns, goals, g_sums, random_goal_cond, current_clusters)

    new_batch = self.extend(batch)
    new_observations, new_fi, new_actions, new_rewards, new_discounted_returns, new_goals, new_g_sums,\
    new_random_goal_cond, new_current_clusters = new_batch

    c = self.config.c
    batch_len = len(new_actions)
    end = batch_len if self.done else batch_len - c

    observations, fi, actions, rewards, discounted_returns, goals, g_sums, random_goal_cond, current_clusters = \
      [], [], [], [], [], [], [], [], []
    target_goals = []
    ris = []

    for t in range(c, end):
      target_goal = new_fi[t + c] - new_fi[t]
      target_goals.append(target_goal)

      ri = 0
      for i in range(1, c + 1):
        ri_s_diff = new_fi[t] - new_fi[t - i]
        ri += self.cosine_similarity(ri_s_diff, new_goals[t - i])
      ri /= c

      ris.append(ri)
      rewards.append(new_rewards[t])
      discounted_returns.append(new_discounted_returns[t])
      goals.append(new_goals[t])
      g_sums.append(new_g_sums[t])
      actions.append(new_actions[t])
      observations.append(new_observations[t])
      fi.append(new_fi[t])
      random_goal_cond.append(new_random_goal_cond[t])
      current_clusters.append(new_current_clusters[t])

    rewards_mix = [r_i * self.config.alpha_r + (1 - self.config.alpha_r) * r_e for (r_i, r_e) in zip(ris, rewards)]
    rewards_mix_plus = np.asarray(rewards_mix + [bootstrap_value_mix])
    discounted_returns_mix = reward_discount(rewards_mix_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_return: discounted_returns,
                 self.local_network.target_mix_return: discounted_returns_mix,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.target_goal: np.stack(target_goals, 0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.goal_sr_clusters: current_clusters,
                 self.local_network.g_sum: np.stack(g_sums, 0),
                 self.local_network.random_goal_cond: random_goal_cond
                 }

    to_run = {
             "summary_goal": self.local_network.merged_summary_goal
            }
    if self.name != "worker_0":
      to_run["apply_grad_goal"] = self.local_network.apply_grads_goal

    """Do an update on the intra-option policies"""
    results = self.sess.run(to_run, feed_dict=feed_dict)
    self.summaries_goal = results["summary_goal"]

    to_run = {
      "summary_option": self.local_network.merged_summary_option,
      "summary_critic": self.local_network.merged_summary_critic,

    }
    if self.name != "worker_0":
      to_run["apply_grads_option"] = self.local_network.apply_grads_option
      to_run["apply_grads_critic"] = self.local_network.apply_grads_critic

    results = self.sess.run(to_run, feed_dict=feed_dict)
    self.summaries_critic = results["summary_critic"]
    self.summaries_option = results["summary_option"]

    """Store the bootstrap target returns at the end of the trajectory"""
    self.R_mix = discounted_returns_mix[-1]
    self.R = discounted_returns[-1]

    if self.last_batch_done:
      self.last_batch_done = False

  def write_step_summary(self):
    self.summary = tf.Summary()
    self.summary.value.add(tag='Step/Action', simple_value=self.action)
    self.summary.value.add(tag='Step/Reward', simple_value=self.reward)
    self.summary.value.add(tag='Step/V', simple_value=self.v)
    self.summary.value.add(tag='Step/V_mix', simple_value=self.v_mix)
    self.summary.value.add(tag='Step/Target_Return_Mix', simple_value=self.R_mix)
    self.summary.value.add(tag='Step/Target_Return', simple_value=self.R)

    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()

  def update_episode_stats(self):
    if len(self.episode_values_mix) != 0:
      self.episode_mean_values_mix.append(np.mean(self.episode_values_mix))
    if len(self.episode_actions) != 0:
      self.episode_mean_actions.append(get_mode(self.episode_actions))

  def write_summaries(self):
    self.summary = tf.Summary()
    self.summary.value.add(tag='Perf/UndiscReturn', simple_value=float(self.episode_reward))
    # self.summary.value.add(tag='Perf/UndiscMixedReturn', simple_value=float(self.episode_mixed_reward))
    # self.summary.value.add(tag='Perf/UndiscIntrinsicReturn', simple_value=float(self.episode_intrinsic_reward))
    self.summary.value.add(tag='Perf/Length', simple_value=float(self.episode_length))

    for sum in [self.summaries_sf, self.summaries_aux, self.summaries_term, self.summaries_critic, self.summaries_option, self.summaries_goal]:
      if sum is not None:
        self.summary_writer.add_summary(sum, self.global_episode_np)

    if len(self.episode_mean_values_mix) != 0:
      last_mean_value_mix = np.mean(self.episode_mean_values_mix[-self.config.step_summary_interval:])
      self.summary.value.add(tag='Perf/MixValue', simple_value=float(last_mean_value_mix))
    if len(self.episode_mean_actions) != 0:
      last_frequent_action = self.episode_mean_actions[-1]
      self.summary.value.add(tag='Perf/FreqActions', simple_value=last_frequent_action)

    self.summary_writer.add_summary(self.summary, self.global_episode_np)
    self.summary_writer.flush()

  def print_g(self):
    plt.clf()
    reproj_obs = np.squeeze(self.s, -1)
    clusters = self.global_network.goal_clusters.get_clusters()
    reproj_state_occupancy = self.episode_state_occupancy.reshape(
      self.config.input_size[0],
      self.config.input_size[1])
    reproj_goal_occupancy = self.episode_goal_occupancy.reshape(
      self.config.input_size[0],
      self.config.input_size[1])

    params = {'figure.figsize': (160, 40),
              'axes.titlesize': 'x-large',
              }
    plt.rcParams.update(params)

    f = plt.figure(figsize=(160, 40), frameon=False)
    plt.axis('off')
    # f.patch.set_visible(False)

    gs0 = gridspec.GridSpec(1, 6)

    gs00 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[0, 0])
    gs01 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0, 1])
    gs02 = gridspec.GridSpecFromSubplotSpec(4, 12, subplot_spec=gs0[0, 2:6])

    ax1 = plt.Subplot(f, gs00[:, :])
    ax1.set_aspect(1.0)
    ax1.axis('off')
    ax1.set_title(f'Goal {self.random_goal_cond}', fontsize=80)
    # sns.heatmap(np.zeros([self.config.input_size[0],
    # self.config.input_size[1]]), cmap="Blues", ax=ax1)

    """Adding borders"""
    for idx in range(self.nb_states):
      i, j = self.env.get_state_xy(idx)
      if self.env.not_wall(i, j):
        if reproj_goal_occupancy[i][j] == 0:
          ax1.add_patch(
            patches.Rectangle(
              (j, self.config.input_size[0] - i - 1),  # (x,y)
              1.0,  # width
              1.0,  # height
              facecolor="white"
            )
          )
        else:
          ax1.text(j, self.config.input_size[0] - i - 1,
                            int(reproj_goal_occupancy[i][j]), fontsize=80, color='k')
        # ax1.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, 0.35, 0.35,
        #           head_width=0.2, head_length=0.2, fc='k', ec='k', linewidth=1)
      else:
        ax1.add_patch(
          patches.Rectangle(
            (j, self.config.input_size[0] - i - 1),  # (x,y)
            1.0,  # width
            1.0,  # height
            facecolor="gray"
          )
        )

      ax1.set_xlim([0, self.config.input_size[1]])
      ax1.set_xlim([0, self.config.input_size[0]])

      for i in range(self.config.input_size[1]):
        ax1.axvline(i, color='k', linestyle=':')
      ax1.axvline(self.config.input_size[1], color='k', linestyle=':')

      for j in range(self.config.input_size[0]):
        ax1.axhline(j, color='k', linestyle=':')
      ax1.axhline(self.config.input_size[0], color='k', linestyle=':')
			#
      # if self.env.not_wall(ii, jj):
      #   r = patches.Rectangle(
      #       (jj, self.config.input_size[0] - ii - 1),  # (x,y)
      #       1.0,  # width
      #       1.0,  # height
      #       color="blue"
      #     )
      #   ax1.add_patch(r)
      #   ax1.annotate(r, (jj + 0.5, self.config.input_size[0] - ii - 1 + 0.5), color='k', weight='bold', fontsize=80, ha='center', va='center')
        # ax1.text(jj, self.config.input_size[0] - ii - 1,
        #          reproj_goal_occupancy[ii][jj], fontsize=80)
      # else:
      #   ax1.add_patch(
      #     patches.Rectangle(
      #       (jj, self.config.input_size[0] - ii - 1),  # (x,y)
      #       1.0,  # width
      #       1.0,  # height
      #       color="white"
      #     )
      #   )
    f.add_subplot(ax1)

    # sns.heatmap(reproj_goal_occupancy, cmap="Blues", ax=ax1, annot=True)
    # for i in range(self.config.input_size[0]):
    #   for j in range(self.config.input_size[1]):
    #     text = ax1.text(j, i, reproj_goal_occupancy[i][j],
    #                    ha="center", va="center", color="k")

    # self.plot_policy_embedding(self.g, ax1)
    # """Adding borders"""
    # for idx in range(self.nb_states):
    #   ii, jj = self.env.get_state_xy(idx)
    #   if self.env.not_wall(ii, jj):
    #     # ax1.text(jj, self.config.input_size[0] - ii - 1, reproj_goal_occupancy[ii][jj], fontsize=80)
    #     continue
    #   else:
    #     ax1.add_patch(
    #       patches.Rectangle(
    #         (jj, self.config.input_size[0] - ii - 1),  # (x,y)
    #         1.0,  # width
    #         1.0,  # height
    #         facecolor="gray"
    #       )
    #     )
    # f.add_subplot(ax1)

    ax2 = plt.Subplot(f, gs01[0:2, 0:2])
    ax2.set_aspect(1.0)
    ax2.axis('off')
    ax2.set_title('Last observation', fontsize=80)
    sns.heatmap(reproj_obs, cmap="Blues", ax=ax2)

    """Adding borders"""
    for idx in range(self.nb_states):
      ii, jj = self.env.get_state_xy(idx)
      if self.env.not_wall(ii, jj):
        continue
      else:
        ax2.add_patch(
          patches.Rectangle(
            (jj, self.config.input_size[0] - ii - 1),  # (x,y)
            1.0,  # width
            1.0,  # height
            facecolor="gray"
          )
        )
    f.add_subplot(ax2)

    ax3 = plt.Subplot(f, gs01[2:4, 0:2])
    ax3.set_aspect(1.0)
    ax3.axis('off')
    ax3.set_title('State occupancy', fontsize=80)
    sns.heatmap(reproj_state_occupancy, cmap="Blues", ax=ax3)

    """Adding borders"""
    for idx in range(self.nb_states):
      ii, jj = self.env.get_state_xy(idx)
      if self.env.not_wall(ii, jj):
        continue
      else:
        ax3.add_patch(
          patches.Rectangle(
            (jj, self.config.input_size[0] - ii - 1),  # (x,y)
            1.0,  # width
            1.0,  # height
            facecolor="gray"
          )
        )
    f.add_subplot(ax3)

    indx = [[0, 0], [0, 3], [0, 6], [0, 9],
            [2, 0], [2, 3], [2, 6], [2, 9]]

    for k in range(len(clusters)):
      # reproj_cluster = clusters[k].reshape(
      #   self.config.input_size[0],
      #   self.config.input_size[1])

      """Plot of the eigenvector"""
      axn = plt.Subplot(f, gs02[indx[k][0]:(indx[k][0]+2), indx[k][1]:(indx[k][1]+3)])
      axn.set_aspect(1.0)
      axn.axis('off')
      axn.set_title("%.3f/%.3f" % (self.attention_weights[k], self.query_content_match[k]), fontsize=80)
      self.plot_policy_embedding(clusters[k], axn)

      """Adding borders"""
      for idx in range(self.nb_states):
        ii, jj = self.env.get_state_xy(idx)
        if self.env.not_wall(ii, jj):
          continue
        else:
          axn.add_patch(
            patches.Rectangle(
              (jj, self.config.input_size[0] - ii - 1),  # (x,y)
              1.0,  # width
              1.0,  # height
              facecolor="gray"
            )
          )
      f.add_subplot(axn)

    """Saving plots"""
    plt.savefig(os.path.join(self.policy_folder, f'g_{self.global_step_np}_{self.global_episode_np}.png'))
    plt.close()

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-{}.cptk'.format(self.global_episode_np),
                    global_step=self.global_episode)
    tf.logging.info(
      "Saved Model at {}".format(self.model_path + '/model-{}.cptk'.format(self.global_episode_np)))

    goal_clusters_path = os.path.join(self.cluster_model_path, "goal_clusters_{}.pkl".format(self.global_episode_np))
    f = open(goal_clusters_path, 'wb')
    pickle.dump(self.global_network.goal_clusters, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

  def plot_policy_embedding(self, embedding, ax):
    epsilon = 0.0001
    with self.sess.as_default(), self.sess.graph.as_default():
      self.env.define_network(self.local_network)
      self.env.define_session(self.sess)
      """Do policy iteration"""
      discount = 0.9
      polIter = PolicyIteration(discount, self.env, augmentActionSet=True)
      """Use the direction of the eigenvector as intrinsic reward for the policy iteration algorithm"""
      self.env.define_reward_function(embedding)
      """Get the optimal value function and policy"""
      V, pi = polIter.solvePolicyIteration()

      for j in range(len(V)):
        if V[j] < epsilon:
          pi[j] = len(self.env.get_action_set())

      policy = pi[0:self.nb_states]
      for idx in range(len(policy)):
        i, j = self.env.get_state_xy(idx)

        dx = 0
        dy = 0
        if policy[idx] == 0:  # up
          dy = 0.35
        elif policy[idx] == 1:  # right
          dx = 0.35
        elif policy[idx] == 2:  # down
          dy = -0.35
        elif policy[idx] == 3:  # left
          dx = -0.35
        elif self.env.not_wall(i, j) and policy[idx] == 4:  # termination
          ax.add_patch(
            patches.Rectangle(
              (j, self.config.input_size[0] - i - 1),  # (x,y)
              1.0,  # width
              1.0,  # height
              facecolor="lime"
            )
          )
          circle = plt.Circle(
            (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.1, color='k', linewidth=1)
          ax.add_artist(circle)

        if self.env.not_wall(i, j):
          ax.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, dx, dy,
                    head_width=0.2, head_length=0.2, fc='k', ec='k', linewidth=1)
        else:
          ax.add_patch(
            patches.Rectangle(
              (j, self.config.input_size[0] - i - 1),  # (x,y)
              1.0,  # width
              1.0,  # height
              facecolor="gray"
            )
          )

      ax.set_xlim([0, self.config.input_size[1]])
      ax.set_xlim([0, self.config.input_size[0]])

      for i in range(self.config.input_size[1]):
        ax.axvline(i, color='k', linestyle=':')
      ax.axvline(self.config.input_size[1], color='k', linestyle=':')

      for j in range(self.config.input_size[0]):
        ax.axhline(j, color='k', linestyle=':')
      ax.axhline(self.config.input_size[0], color='k', linestyle=':')


  """Plot plicies and value functions"""
  def plot_clusters(self):
    clusters = self.global_network.goal_clusters.get_clusters()
    policy_folder = os.path.join(self.summary_path, "policies")
    tf.gfile.MakeDirs(policy_folder)

    v_folder = os.path.join(self.summary_path, "value_functions")
    tf.gfile.MakeDirs(v_folder)
    epsilon = 0.0001
    with self.sess.as_default(), self.sess.graph.as_default():
      self.env.define_network(self.local_network)
      self.env.define_session(self.sess)
      for i in range(len(clusters)):
        """Do policy iteration"""
        discount = 0.9
        polIter = PolicyIteration(discount, self.env, augmentActionSet=True)
        """Use the direction of the eigenvector as intrinsic reward for the policy iteration algorithm"""
        self.env.define_reward_function(clusters[i])
        """Get the optimal value function and policy"""
        V, pi = polIter.solvePolicyIteration()

        for j in range(len(V)):
          if V[j] < epsilon:
            pi[j] = len(self.env.get_action_set())

        """Plot them"""
        self.plot_value_function(V[0:self.nb_states], str(i) + '_', v_folder)
        self.plot_policy(pi[0:self.nb_states], str(i) + '_', policy_folder)

  """Plot value functions"""
  def plot_value_function(self, value_function, prefix, v_folder):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    X, Y = np.meshgrid(np.arange(self.config.input_size[1]), np.arange(self.config.input_size[0]))
    reproj_value_function = value_function.reshape(self.config.input_size[0], self.config.input_size[1])

    """Build the support"""
    for i in range(len(X)):
      for j in range(int(len(X[i]) / 2)):
        tmp = X[i][j]
        X[i][j] = X[i][len(X[i]) - j - 1]
        X[i][len(X[i]) - j - 1] = tmp

    cm.jet(np.random.rand(reproj_value_function.shape[0], reproj_value_function.shape[1]))

    ax.plot_surface(X, Y, reproj_value_function, rstride=1, cstride=1,
                    cmap=plt.get_cmap('jet'))
    plt.gca().view_init(elev=30, azim=30)
    plt.savefig(os.path.join(v_folder, "SuccessorFeatures" + prefix + 'value_function.png'))
    plt.close()

  """Plot the policy"""

  def plot_policy(self, policy, prefix, policy_folder):
    plt.clf()
    for idx in range(len(policy)):
      i, j = self.env.get_state_xy(idx)

      dx = 0
      dy = 0
      if policy[idx] == 0:  # up
        dy = 0.35
      elif policy[idx] == 1:  # right
        dx = 0.35
      elif policy[idx] == 2:  # down
        dy = -0.35
      elif policy[idx] == 3:  # left
        dx = -0.35
      elif self.env.not_wall(i, j) and policy[idx] == 4:  # termination
        circle = plt.Circle(
          (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.025, color='k')
        plt.gca().add_artist(circle)

      if self.env.not_wall(i, j):
        plt.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, dx, dy,
                  head_width=0.05, head_length=0.05, fc='k', ec='k')
      else:
        plt.gca().add_patch(
          patches.Rectangle(
            (j, self.config.input_size[0] - i - 1),  # (x,y)
            1.0,  # width
            1.0,  # height
            facecolor="gray"
          )
        )

    plt.xlim([0, self.config.input_size[1]])
    plt.ylim([0, self.config.input_size[0]])

    for i in range(self.config.input_size[1]):
      plt.axvline(i, color='k', linestyle=':')
    plt.axvline(self.config.input_size[1], color='k', linestyle=':')

    for j in range(self.config.input_size[0]):
      plt.axhline(j, color='k', linestyle=':')
    plt.axhline(self.config.input_size[0], color='k', linestyle=':')

    plt.savefig(os.path.join(policy_folder, "SuccessorFeatures_" + prefix + 'policy.png'))
    plt.close()
