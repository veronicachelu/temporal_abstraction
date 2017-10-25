import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from collections import deque
from agents.schedules import LinearSchedule, TFLinearSchedule
from PIL import Image
import scipy.stats
FLAGS = tf.app.flags.FLAGS


class ACAgent():
  def __init__(self, game, thread_id, global_step, config):
    self.name = "worker_" + str(thread_id)
    self.thread_id = thread_id
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.model_path = os.path.join(config.stage_logdir, "models")
    self.summary_path = os.path.join(config.stage_logdir, "summaries")
    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)
    self.increment_global_step = self.global_step.assign_add(1)
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.episode_mean_q_values = []
    self.episode_mean_returns = []
    self.episode_mean_oterms = []
    self.episode_mean_options = []
    self.episode_options = []
    self.task = 0

    self.config = config
    self.total_steps_tensor = tf.Variable(0, dtype=tf.int32, name='total_steps_tensor', trainable=False)
    self.increment_total_steps_tensor = self.total_steps_tensor.assign_add(1)
    self.total_steps = 0
    self.action_size = game.action_space.n
    self._network_optimizer = self.config.network_optimizer(
      self.config.lr, name='network_optimizer')

    self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
    self.summary = tf.Summary()

    self.local_network = config.network(self.name, config, self.action_size)
    # self._random = tf.random_uniform(shape=[()], minval=0., maxval=1., dtype=tf.float32)

    self.update_local_vars = update_target_graph('global', self.name)
    self.env = game

  def evaluate_agent(self, sess):
    episode_reward = 0
    s = self.env.reset()
    feed_dict = {self.local_network.observation: np.stack([s])}
    d = False
    while not d:
      feed_dict = {self.local_network.observation: np.stack([s])}
      pi = sess.run(self.local_network.policy, feed_dict=feed_dict)
      action = np.random.choice(pi[0], p=pi[0])
      action = np.argmax(pi == action)
      s1, r, d, _ = self.env.step(action)

      r = np.clip(r, -1, 1)
      episode_reward += r

    return episode_reward

  def train(self, rollout, sess, bootstrap_value, summaries=False):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    actions = rollout[:, 1]
    rewards = rollout[:, 2]
    timesteps = rollout[:, 3]
    values = rollout[:, 5]

    # The advantage function uses "Generalized Advantage Estimation"
    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_rewards = discount(rewards_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_return: discounted_rewards,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions}

    _, ms, img_summ, loss, policy_loss, entropy_loss, critic_loss = \
      sess.run([self.local_network.apply_grads,
                self.local_network.merged_summary,
                self.local_network.image_summaries,
                self.local_network.loss,
                self.local_network.policy_loss,
                self.local_network.entropy_loss,
                self.local_network.critic_loss],
               feed_dict=feed_dict)
    return ms, img_summ, loss, policy_loss, entropy_loss, critic_loss

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if self.total_steps > self.config.steps:
          return 0

        sess.run(self.update_local_vars)
        episode_buffer = []
        episode_values = []
        episode_returns = []
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0
        R = 0

        s = self.env.reset()

        while not d:
          feed_dict = {self.local_network.observation: np.stack([s])}
          pi, v = sess.run([self.local_network.policy, self.local_network.value],
                                                     feed_dict=feed_dict)
          a = np.random.choice(pi[0], p=pi[0])
          a = np.argmax(pi == a)
          v = v[0, 0]

          s1, r, d, _ = self.env.step(a)

          r = np.clip(r, -1, 1)
          self.total_steps += 1
          sess.run(self.increment_total_steps_tensor)
          episode_buffer.append([s, a, r, t, d, v])
          episode_values.append(v)
          episode_reward += r
          t += 1
          t_counter += 1
          s = s1

          if t_counter == self.config.max_update_freq or d:
            feed_dict = {self.local_network.observation: np.stack([s])}
            value = sess.run([self.local_network.value],
                                      feed_dict=feed_dict)
            value = value[0][0]
            R = 0 if d else value
            ms, img_summ, loss, policy_loss, entropy_loss, critic_loss = self.train(episode_buffer, sess, R)
            episode_buffer = []
            t_counter = 0
          episode_returns.append(R)
          if self.name == "worker_0":
            print("Episode {} >>> Step {} >>> Length: {} >>> Reward: {} >>> Mean Value: {} "
                  " >>> Return {}".format(episode_count, self.total_steps, t, episode_reward,
                                                        np.mean(episode_values[-1]),
                                                        np.mean(episode_returns[-1])))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(t)
        self.episode_mean_values.append(np.mean(episode_values))
        self.episode_mean_returns.append(np.mean(episode_returns))

        if episode_count % self.config.eval_interval == 0 and self.total_steps != 0 and \
                self.name == 'worker_0':
          eval_reward = self.evaluate_agent(sess)
          self.summary.value.add(tag='Perf/EvalReward', simple_value=float(eval_reward))
          self.summary_writer.add_summary(self.summary, self.total_steps)
          self.summary_writer.flush()

        if episode_count % self.config.checkpoint_interval == 0 and self.name == 'worker_0' and \
                self.total_steps != 0:
          saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                     global_step=self.global_step)
          print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

        if episode_count % self.config.summary_interval == 0 and self.total_steps != 0 and \
                self.name == 'worker_0':

          last_reward = self.episode_rewards[-1]
          last_length = self.episode_lengths[-1]
          mean_value = np.mean(self.episode_mean_values[-1])
          mean_return = np.mean(self.episode_mean_returns[-1])

          self.summary.value.add(tag='Perf/Reward', simple_value=float(last_reward))
          self.summary.value.add(tag='Perf/Length', simple_value=float(last_length))
          self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
          self.summary.value.add(tag='Perf/Return', simple_value=float(mean_return))

          self.summary_writer.add_summary(ms, self.total_steps)

          self.summary_writer.add_summary(img_summ, self.total_steps)

          self.summary_writer.add_summary(self.summary, self.total_steps)
          self.summary_writer.flush()

        if self.name == 'worker_0':
          sess.run(self.increment_global_step)
        episode_count += 1

