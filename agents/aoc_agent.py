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
        self.model_path = os.path.join(FLAGS.logdir, "models")
        self.summary_path = os.path.join(FLAGS.logdir, "summaries")
        self.increment_global_step = self.global_step.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_mean_q_values = []
        self.config = config
        self.total_steps = tf.Variable(tf.zeros_like(self.global_step), False, name="total_steps")
        self.increment_total_steps = self.total_steps.assign_add(1)
        self.action_size = game.action_space.n
        self._network_optimizer = self.config.network_optimizer(
            self.config.lr, name='network_optimizer')
        self._exploration_options = TFLinearSchedule(self.config.explore_steps, self.config.final_random_action_prob,
                                                     self.config.initial_random_action_prob)
        self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
        self.summary = tf.Summary()

        self.local_network = config.network(self.name, config, self.action_size)
        # self._random = tf.random_uniform(shape=[()], minval=0., maxval=1., dtype=tf.float32)

        self.update_local_vars = update_target_graph('global', self.name)
        self.env = game

    def get_policy_over_options(self):
        self.probability_of_random_option = self._exploration_options.value(self.total_steps)
        return self.local_network.get_policy_over_options(self.probability_of_random_option)

    def get_v(self):
        self.probability_of_random_option = self._exploration_options.value(self.total_steps)
        return self.local_network.get_v(self.probability_of_random_option)

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
        discounted_rewards = discount(rewards_plus, FLAGS.gamma)[:-1]

        feed_dict = {self.local_network.target_return: discounted_rewards,
                     self.local_network.target_v: values,
                     self.local_network.delib: self.delib + self.config.margin_cost,
                     self.local_network.inputs: np.stack(observations, axis=0),
                     self.local_network.actions: actions,
                     self.local_network.options: options}

        _, ms, img_summ = sess.run([self.local_network.apply_grads,
                                    self.local_network.merged_summary,
                                    self.local_network.image_summaries],
                                   feed_dict=feed_dict)
        return ms, img_summ

    def play(self, sess, coord, saver):
        episode_count = sess.run(self.global_step)

        # if not FLAGS.train:
        #     test_episode_count = 0
        self.total_steps.assign(tf.zeros_like(self.total_steps))

        print("Starting worker " + str(self.thread_id))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
            # while episode_count < self.config.steps:
                sess.run(self.update_local_vars)
                episode_buffer = []
                episode_values = []
                episode_q_values = []
                episode_reward = 0
                d = False
                t = 0
                o_t = True
                self.delib = self.config.delib_cost
                self.frame_counter = 0

                s = self.env.reset()
                o = self.get_policy_over_options()
                while not d:

                    o_t = self.local_network.get_o_term(o)
                    v = self.get_v()
                    q = self.local_network.get_q(o)
                    a = self.local_network.get_action(o)

                    feed_dict = {self.local_network.observation: np.stack([s])}
                    option, action, value, q_value, o_term, _ = sess.run([o, a, v, q, o_t, self.increment_total_steps], feed_dict=feed_dict)
                    action, option, value, q_value, o_term = action[0], option[0], value[0], q_value[0], o_term[0]
                    s1, r, d, _ = self.env.step(action)

                    r = np.clip(r, -1, 1)
                    self.frame_counter += 1
                    processed_reward = float(r) - (float(o_term) * self.delib * float(self.frame_counter > 1))
                    episode_buffer.append([s, o, a, processed_reward, t, d, o_t, v[0], q[0]])
                    episode_values.append(v[0])
                    episode_q_values.append(q[0])
                    episode_reward += r
                    t += 1
                    s = s1

                    option_term = (o_term and t >= self.config.min_update_freq)
                    if t == self.config.max_update_freq or d or option_term:
                        delib_cost = self.delib * float(self.frame_counter > 1)
                        value = value - delib_cost if o_t else q_value
                        R = 0 if d else value
                        ms, img_summ = self.train(episode_buffer, sess, R)
                    if not d:
                        self.delib = self.config.delib_cost
                        if o_term:
                            o = self.get_policy_over_options()

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(t)
                self.episode_mean_values.append(np.mean(episode_values))
                self.episode_mean_q_values.append(np.mean(episode_q_values))

                if FLAGS.train and episode_count % FLAGS.summary_interval == 0 and episode_count != 0 and \
                                self.name == 'worker_0':
                    if episode_count % FLAGS.checkpoint_interval == 0 and self.name == 'worker_0' and FLAGS.train == True:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                                   global_step=self.global_step)
                        print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

                    mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                    mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
                    mean_value = np.mean(self.episode_mean_values[-FLAGS.summary_interval:])
                    mean_q_value = np.mean(self.episode_mean_q_values[-FLAGS.summary_interval:])

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    self.summary.value.add(tag='Perf/QValue', simple_value=float(mean_q_value))

                    if FLAGS.train:
                        self.summary_writer.add_summary(ms, global_step=self.global_step)

                    for s in img_sum:
                        self.summary_writer.add_summary(s, episode_count)

                    self.summary_writer.add_summary(self.summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment_global_step)
                if not FLAGS.train:
                    test_episode_count += 1
                episode_count += 1
