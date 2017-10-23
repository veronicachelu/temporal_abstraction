import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from collections import deque
from agents.schedules import LinearSchedule, TFLinearSchedule
from PIL import Image
import scipy.stats
import matplotlib.patches as patches
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np
from matplotlib import cm
import tensorflow as tf
import os
FLAGS = tf.app.flags.FLAGS


class ACOptionAgent():
  def __init__(self, game, thread_id, global_step, config, option, eval, evect, stage=None):
    self.name = "worker_" + str(thread_id)
    self.thread_id = thread_id
    self.option = option
    self.eval = eval
    self.evect = evect
    self.config = config
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.model_path = os.path.join(config.stage_logdir, "models")
    self.summary_path = os.path.join(config.stage_logdir, "summaries")
    self.outputPath = os.path.join(config.stage_logdir, "visuals")
    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)
    tf.gfile.MakeDirs(self.outputPath)
    self.increment_global_step = self.global_step.assign_add(1)
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.episode_mean_q_values = []
    self.episode_mean_returns = []
    self.episode_mean_oterms = []
    self.episode_mean_options = []
    self.episode_options = []
    self.config = config
    self.total_steps_tensor = tf.Variable(0, dtype=tf.int32, name='total_steps_tensor', trainable=False)
    self.increment_total_steps_tensor = self.total_steps_tensor.assign_add(1)
    self.total_steps = 0
    self.action_size = game.action_space.n

    self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
    self.summary = tf.Summary()
    stage = 4 if stage is None else stage
    self.local_network = config.network(self.name, config, self.action_size, stage)
    self.local_network.option = option

    self.update_local_vars = update_target_graph('global', self.name)
    self.env = game
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # X, Y = np.meshgrid(np.arange(self.config.input_size[0]), np.arange(self.config.input_size[1]))


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

    _, ms, img_summ, loss, option_policy_loss, option_entropy_loss, option_critic_loss = \
      sess.run([self.local_network.apply_grads,
                self.local_network.merged_summary,
                self.local_network.image_summaries,
                self.local_network.loss,
                self.local_network.option_policy_loss,
                self.local_network.option_entropy_loss,
                self.local_network.option_critic_loss],
               feed_dict=feed_dict)
    return ms, img_summ, loss, option_policy_loss, option_entropy_loss, option_critic_loss

  def plot(self, sess, coord, saver):
    plt.clf()
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      print("Starting worker " + str(self.thread_id))

      sess.run(self.update_local_vars)

      for idx in range(self.config.input_size[0] * self.config.input_size[1]):
        s, i, j = self.env.get_state(idx)

        feed_dict = {self.local_network.observation: np.stack([s])}
        pi, v, sf = sess.run(
          [self.local_network.option_policy, self.local_network.option_value, self.local_network.sf],
          feed_dict=feed_dict)
        pi = pi[0, self.option]
        v = v[0, self.option, 0]
        a = np.random.choice(pi, p=pi)
        a = np.argmax(pi == a)
        dx = 0
        dy = 0
        if a == 0:  # up
          dy = 0.35
        elif a == 1:  # right
          dx = 0.35
        elif a == 2:  # down
          dy = -0.35
        elif a == 3:  # left
          dx = -0.35
        elif a == self.action_size and self.env.not_wall(i, j):  # termination
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

        plt.savefig(os.path.join(self.outputPath, "SuccessorFeatures_" + self.option + 'policy.png'))
        plt.close()

          # sf = sf[0, :, a]
          # s1, r, d, _ = self.env.step(a)
          # feed_dict = {self.local_network.observation: np.stack([s1])}
          # sf_next, pi_next = sess.run([self.local_network.sf, self.local_network.option_policy],
          #                             feed_dict=feed_dict)
          # pi_next = pi_next[0, self.option]
          # a_next = np.random.choice(pi_next, p=pi_next)
          # a_next = np.argmax(pi_next == a_next)
          # if a_next == self.action_size:
          #   r = 0
          #   d = True
          # else:
          #   sf_next = sf_next[0, :, a_next]
          #   r = self.get_reward(sf, sf_next)

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
        if episode_count > self.config.steps:
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
          pi, v, sf = sess.run([self.local_network.option_policy, self.local_network.option_value, self.local_network.sf],
                                                     feed_dict=feed_dict)
          pi = pi[0, self.option]
          v = v[0, self.option, 0]
          a = np.random.choice(pi, p=pi)
          a = np.argmax(pi == a)
          if a == self.action_size:
            break

          sf = sf[0, :, a]
          s1, r, d, _ = self.env.step(a)
          feed_dict = {self.local_network.observation: np.stack([s1])}
          sf_next, pi_next = sess.run([self.local_network.sf, self.local_network.option_policy],
                            feed_dict=feed_dict)
          pi_next = pi_next[0, self.option]
          a_next = np.random.choice(pi_next, p=pi_next)
          a_next = np.argmax(pi_next == a_next)
          if a_next == self.action_size:
            r = 0
            d = True
          else:
            sf_next = sf_next[0, :, a_next]
            r = self.get_reward(sf, sf_next)

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
            value = sess.run(self.local_network.option_value,
                                      feed_dict=feed_dict)
            value = value[0, self.option, 0]
            R = 0 if d else value
            ms, img_summ, loss, policy_loss, entropy_loss, critic_loss = self.train(episode_buffer, sess, R)

            episode_buffer = []
            t_counter = 0
          episode_returns.append(R)
          if self.name == "worker_0":
            print("Episode {} >>> Step {} >>> Length: {} >>> Reward: {} >>> Mean Value: {} "
                  " >>> Return {}".format(episode_count, self.total_steps, t, episode_reward,
                                                        np.mean(episode_values[-min(self.config.summary_interval, t):]),
                                                        np.mean(episode_returns[-min(self.config.summary_interval, t):])))
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

  def get_reward(self, sf_old, sf_new):
      state_dif = sf_new - sf_old
      state_dif_norm = np.linalg.norm(state_dif)
      state_dif_normalized = state_dif / (state_dif_norm + 1e-8)

      return np.dot(state_dif_normalized, self.evect)


