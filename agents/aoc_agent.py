import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from agents.schedules import LinearSchedule, TFLinearSchedule
from PIL import Image
import scipy.stats
FLAGS = tf.app.flags.FLAGS

def get_mode(arr):
  u, indices = np.unique(arr, return_inverse=True)
  return u[np.argmax(np.bincount(indices))]

class AOCAgent():
  def __init__(self, game, thread_id, global_step, config):
    self.name = "worker_" + str(thread_id)
    self.thread_id = thread_id
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.model_path = os.path.join(config.logdir, "models")
    self.summary_path = os.path.join(config.logdir, "summaries")
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
    feed_dict = {self.local_network.observation: np.stack([s]),
                 self.local_network.total_steps: self.total_steps}
    option = sess.run([self.local_network.current_option], feed_dict=feed_dict)[0][0]
    d = False
    while not d:
      feed_dict = {self.local_network.observation: np.stack([s]),
                   self.local_network.total_steps: self.total_steps}
      options, o_term = sess.run([self.local_network.options, self.local_network.termination], feed_dict=feed_dict)
      o_term = o_term[0, option] > np.random.uniform()
      pi = options[0, option]
      action = np.random.choice(pi, p=pi)
      action = np.argmax(pi == action)
      s1, r, d, _ = self.env.step(action)

      r = np.clip(r, -1, 1)
      episode_reward += r

      if not d and o_term:
          feed_dict = {self.local_network.observation: np.stack([s]),
                       self.local_network.total_steps: self.total_steps}
          option = sess.run([self.local_network.current_option], feed_dict=feed_dict)[0][0]
    return episode_reward

  def train(self, rollout, sess, bootstrap_value, summaries=False):
    rollout = np.array(rollout)
    observations = rollout[:, 0]
    options = rollout[:, 1]
    actions = rollout[:, 2]
    rewards = rollout[:, 3]
    timesteps = rollout[:, 4]
    done = rollout[:, 5]
    option_term = rollout[:, 6]
    values = rollout[:, 7]
    q_values = rollout[:, 8]

    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_rewards = discount(rewards_plus, self.config.discount)[:-1]

    feed_dict = {self.local_network.target_return: discounted_rewards,
                 # self.local_network.target_v: values,
                 self.local_network.delib: self.delib + self.config.margin_cost,
                 self.local_network.observation: np.stack(observations, axis=0),
                 self.local_network.actions_placeholder: actions,
                 self.local_network.options_placeholder: options}

    _, ms, img_summ, loss, policy_loss, entropy_loss, critic_loss, term_loss = \
      sess.run([self.local_network.apply_grads,
                self.local_network.merged_summary,
                self.local_network.image_summaries,
                self.local_network.loss,
                self.local_network.policy_loss,
                self.local_network.entropy_loss,
                self.local_network.critic_loss,
                self.local_network.term_loss],
               feed_dict=feed_dict)
    # sess.run(self.update_local_vars)
    return ms, img_summ, loss, policy_loss, entropy_loss, critic_loss, term_loss

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      eval_reward = 0
      # if not FLAGS.train:
      #     test_episode_count = 0
      # self.total_steps.assign(tf.zeros_like(self.total_steps))

      print("Starting worker " + str(self.thread_id))

      while not coord.should_stop():
      # while self.total_steps < self.config.steps:
        sess.run(self.update_local_vars)
        episode_buffer = []
        episode_values = []
        episode_q_values = []
        episode_oterm = []
        episode_returns = []
        episode_options = []
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0
        o_t = True
        self.delib = self.config.delib_cost
        self.frame_counter = 0
        R = 0

        s = self.env.reset()
        # pil_image = Image.fromarray(np.uint8(s[:, :, 0] * 255))
        # pil_image.show()

        feed_dict = {self.local_network.observation: np.stack([s])}
        option = sess.run([self.local_network.current_option], feed_dict=feed_dict)[0][0]
        episode_options.append(option)
        while not d:
          feed_dict = {self.local_network.observation: np.stack([s])}
          options, value, q_value, o_term = sess.run([self.local_network.options, self.local_network.v,
                                                      self.local_network.q_val, self.local_network.termination], feed_dict=feed_dict)
          o_term = o_term[0, option] > np.random.uniform()
          q_value = q_value[0, option]
          value = value[0]
          pi = options[0, option]
          action= np.random.choice(pi, p=pi)
          action = np.argmax(pi == action)
          s1, r, d, _ = self.env.step(action)

          r = np.clip(r, -1, 1)
          self.frame_counter += 1
          self.total_steps += 1
          sess.run(self.increment_total_steps_tensor)
          processed_reward = r #- (float(o_term) * self.delib * float(self.frame_counter > 1))
          episode_buffer.append([s, option, action, processed_reward, t, d, o_term, value, q_value])
          episode_values.append(value)
          episode_q_values.append(q_value)
          episode_reward += r
          episode_oterm.append(o_term)
          t += 1
          s = s1
          t_counter += 1

          option_term = (o_term and t_counter >= self.config.min_update_freq)
          if t_counter == self.config.max_update_freq or d or option_term:
            delib_cost = self.delib * float(self.frame_counter > 1)
            feed_dict = {self.local_network.observation: np.stack([s])}
            value, q_value = sess.run([self.local_network.v, self.local_network.q_val],
                                                       feed_dict=feed_dict)
            q_value = q_value[0, option]
            value = value[0]

            value = value #- delib_cost if o_term else q_value
            R = 0 if d else value

            ms, img_summ, loss, policy_loss, entropy_loss, critic_loss, term_loss = self.train(episode_buffer, sess, R)
            #print("Timestep {} >>> Ep_done {} >>> Option_Term {} >>> t_counter {} >>> loss {} >>> policy_loss {} >>> "
            #      "entropy_loss {} >>> critic_loss {} >>> term_loss {}".format(t, d, o_term, t_counter, loss,
            #                                                                   policy_loss, entropy_loss, critic_loss,
            #                                                                   term_loss))
            episode_buffer = []
            t_counter = 0
          episode_returns.append(R)
          if not d:
            self.delib = self.config.delib_cost
            if o_term:
              feed_dict = {self.local_network.observation: np.stack([s])}
              option = sess.run([self.local_network.current_option], feed_dict=feed_dict)[0][0]
              episode_options.append(option)

          print("Episode {} >>> Step {} >>> Length: {} >>> Reward: {} >>> Mean Value: {} >>> Mean Q_Value: {} "
                ">>> O_Term: {} >>> Return {}".format(episode_count, self.total_steps, t, episode_reward,
                                        np.mean(episode_values[-min(self.config.summary_interval, t):]),
                                        np.mean(episode_q_values[-min(self.config.summary_interval, t):]),
                                        np.mean(episode_oterm[-min(self.config.summary_interval, t):]),
                                        np.mean(episode_returns[-min(self.config.summary_interval, t):])))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(t)
        self.episode_mean_values.append(np.mean(episode_values))
        self.episode_mean_q_values.append(np.mean(episode_q_values))
        self.episode_mean_returns.append(np.mean(episode_returns))
        self.episode_mean_oterms.append(np.mean(episode_oterm))
        self.episode_mean_options.append(get_mode(episode_options))

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

          mean_reward = np.mean(self.episode_rewards[-min(self.config.summary_interval, t):])
          mean_length = np.mean(self.episode_lengths[-min(self.config.summary_interval, t):])
          mean_value = np.mean(self.episode_mean_values[-min(self.config.summary_interval, t):])
          mean_q_value = np.mean(self.episode_mean_q_values[-min(self.config.summary_interval, t):])
          mean_return = np.mean(self.episode_mean_returns[-min(self.config.summary_interval, t):])
          mean_oterm = np.mean(self.episode_mean_oterms[-min(self.config.summary_interval, t):])
          mean_option = get_mode(self.episode_mean_options[-min(self.config.summary_interval, t):])

          self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
          self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
          self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
          self.summary.value.add(tag='Perf/QValue', simple_value=float(mean_q_value))
          self.summary.value.add(tag='Perf/Return', simple_value=float(mean_return))
          self.summary.value.add(tag='Perf/Oterm', simple_value=float(mean_oterm))
          self.summary.value.add(tag='Perf/Options', simple_value=mean_option)

          self.summary_writer.add_summary(ms, self.total_steps)

          self.summary_writer.add_summary(img_summ, self.total_steps)

          self.summary_writer.add_summary(self.summary, self.total_steps)
          self.summary_writer.flush()


        if self.name == 'worker_0':
          sess.run(self.increment_global_step)
        # if not FLAGS.train:
        #     test_episode_count += 1
        episode_count += 1



