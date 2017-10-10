import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from agents.schedules import LinearSchedule, TFLinearSchedule

FLAGS = tf.app.flags.FLAGS


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
    self.config = config
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

  def get_policy_over_options(self, batch_size):
    self.probability_of_random_option = self._exploration_options.value(self.total_steps)
    return self.local_network.get_policy_over_options(batch_size, self.probability_of_random_option)

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
                 self.local_network.target_v: values,
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
    sess.run(self.update_local_vars)
    return ms, img_summ, loss, policy_loss, entropy_loss, critic_loss, term_loss

  def play(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      episode_count = sess.run(self.global_step)

      # if not FLAGS.train:
      #     test_episode_count = 0
      # self.total_steps.assign(tf.zeros_like(self.total_steps))

      print("Starting worker " + str(self.thread_id))

      # while not coord.should_stop():
      while episode_count < self.config.steps:
        sess.run(self.update_local_vars)
        episode_buffer = []
        episode_values = []
        episode_q_values = []
        episode_reward = 0
        d = False
        t = 0
        t_counter = 0
        o_t = True
        self.delib = self.config.delib_cost
        self.frame_counter = 0

        s = self.env.reset()
        feed_dict = {self.local_network.observation: np.stack([s]),
                     self.local_network.total_steps: self.total_steps}
        option = sess.run([self.local_network.current_option], feed_dict=feed_dict)[0]
        while not d:


          feed_dict = {self.local_network.observation: np.stack([s]),
                       self.local_network.total_steps: self.total_steps}
          try:
            options, value, q_value, o_term = sess.run([self.local_network.options, self.local_network.v,
                                                        self.local_network.q_val, self.local_network.termination], feed_dict=feed_dict)
            o_term = o_term[0, option] > np.random.uniform()
            q_value = q_value[0, option]
            value = value[0]
            pi = options[0, option]
            action= np.random.choice(pi[0], p=pi[0])
            action = np.argmax(pi == action)
          except:
            print("dadsad")
          s1, r, d, _ = self.env.step(action)

          r = np.clip(r, -1, 1)
          self.frame_counter += 1
          self.total_steps += 1
          processed_reward = float(r) - (float(o_term) * self.delib * float(self.frame_counter > 1))
          episode_buffer.append([s, option, action, processed_reward, t, d, o_term, value, q_value])
          episode_values.append(value)
          episode_q_values.append(q_value)
          episode_reward += r
          t += 1
          s = s1
          t_counter += 1

          option_term = (o_term and t_counter >= self.config.min_update_freq)
          if t_counter == self.config.max_update_freq or d or option_term:
            delib_cost = self.delib * float(self.frame_counter > 1)
            value = value - delib_cost if o_term else q_value
            R = 0 if d else value
            ms, img_summ, loss, policy_loss, entropy_loss, critic_loss, term_loss = self.train(episode_buffer, sess, R)
            #print("Timestep {} >>> Ep_done {} >>> Option_Term {} >>> t_counter {} >>> loss {} >>> policy_loss {} >>> "
            #      "entropy_loss {} >>> critic_loss {} >>> term_loss {}".format(t, d, o_term, t_counter, loss,
            #                                                                   policy_loss, entropy_loss, critic_loss,
            #                                                                   term_loss))
            t_counter = 0
          if not d:
            self.delib = self.config.delib_cost
            if o_term:
              feed_dict = {self.local_network.observation: np.stack([s]),
                           self.local_network.total_steps: self.total_steps}
              option = sess.run([self.local_network.current_option], feed_dict=feed_dict)[0]

        print("Episode {} >>> Step {} >>> Length: {} >>> Reward: {} >>> Mean Value: {} >>> Mean Q_Value: {} "
              ">>> O_Term: {}".format(episode_count, self.total_steps, t, episode_reward,
                                      np.mean(episode_values), np.mean(episode_q_values), o_term))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(t)
        self.episode_mean_values.append(np.mean(episode_values))
        self.episode_mean_q_values.append(np.mean(episode_q_values))

        if episode_count % self.config.checkpoint_interval == 0 and self.name == 'worker_0' and FLAGS.train == True and \
                episode_count != 0:
          saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                     global_step=self.global_step)
          print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

        if FLAGS.train and episode_count % self.config.summary_interval == 0 and episode_count != 0 and \
                self.name == 'worker_0':

          mean_reward = np.mean(self.episode_rewards[-self.config.summary_interval:])
          mean_length = np.mean(self.episode_lengths[-self.config.summary_interval:])
          mean_value = np.mean(self.episode_mean_values[-self.config.summary_interval:])
          mean_q_value = np.mean(self.episode_mean_q_values[-self.config.summary_interval:])

          self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
          self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
          self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
          self.summary.value.add(tag='Perf/QValue', simple_value=float(mean_q_value))

          if FLAGS.train:
            self.summary_writer.add_summary(ms, episode_count)

          self.summary_writer.add_summary(img_summ, episode_count)

          self.summary_writer.add_summary(self.summary, episode_count)
          self.summary_writer.flush()

        if self.name == 'worker_0':
          sess.run(self.increment_global_step)
        # if not FLAGS.train:
        #     test_episode_count += 1
        episode_count += 1
