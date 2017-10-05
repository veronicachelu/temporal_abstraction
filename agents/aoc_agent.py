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
        self.increment_global_episode = self.global_step.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.episode_mean_q_values = []
        self.config = config
        self.action_size = game.action_space.n
        self._network_optimizer = self.config.network_optimizer(
            self.config.lr, name='network_optimizer')
        self._exploration_options = TFLinearSchedule(self.config.explore_steps, self.config.final_random_action_prob,
                                                     self.config.initial_random_action_prob)
        self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))
        self.summary = tf.Summary()

        self.local_network = config.network(self.name, config.conv_layers, config.fc_layers, 4, config.nb_options,
                                            config.num_agents)
        # self._random = tf.random_uniform(shape=[()], minval=0., maxval=1., dtype=tf.float32)

        self.update_local_vars = update_target_graph('global', self.name)
        self.env = game

    def get_policy_over_options(self, sess, s):
        self.probability_of_random_option = self._exploration_options.value(self.global_step)
        max_options = tf.cast(tf.argmax(self.local_network.q_val, 1), dtype=tf.int32)
        exp_options = tf.random_uniform(shape=[1], minval=0, maxval=self.config.nb_options,
                                        dtype=tf.int32)
        local_random = tf.random_uniform(shape=[(s.shape[0])], minval=0., maxval=1., dtype=tf.float32)
        options = tf.where(local_random > self.probability_of_random_option, max_options, exp_options)

        return options

    def get_action(self, sess, s, o):
        current_option_option_one_hot = tf.one_hot(o, self.config.nb_options, name="options_one_hot")
        current_option_option_one_hot = current_option_option_one_hot[:, :, None]
        current_option_option_one_hot = tf.tile(current_option_option_one_hot, [1, 1, self.action_size])
        self.action_probabilities = tf.reduce_sum(tf.multiply(self.local_network.options[:, 0, :], current_option_option_one_hot),
                                                  reduction_indices=1, name="P_a")
        policy = tf.multinomial(tf.log(self.action_probabilities), 1)[:, 0]
        return policy

    def get_v(self, sess, s):
        q_val = self.local_network.q_val
        v = tf.reduce_max(q_val, axis=2) * (1 - self.probability_of_random_option) + \
            self.probability_of_random_option * tf.reduce_mean(q_val, axis=2)
        return v

    def get_q(self, sess, s, o):
        current_option_option_one_hot = tf.one_hot(o, self.config.nb_option, name="options_one_hot")
        q_values = tf.reduce_sum(tf.multiply(self.local_network.q_val, current_option_option_one_hot),
                                 reduction_indices=2, name="Values_Q")
        return q_values

    def get_o_term(self, sess, s, o):
        current_option_option_one_hot = tf.one_hot(o, self.config.nb_option, name="options_one_hot")
        o_terminations = tf.reduce_sum(tf.multiply(self.local_network.termination, current_option_option_one_hot),
                                       reduction_indices=2, name="O_Terminations")
        return o_terminations

    def get_responsible_outputs(self, sess, policy, action):
        actions_onehot = tf.one_hot(action, self.action_size, dtype=tf.float32,
                                    name="Actions_Onehot")
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
        return responsible_outputs

    def get_intra_option_policy(self, sess, o):
        current_option_option_one_hot = tf.one_hot(o, self.config.nb_option, name="options_one_hot")
        current_option_option_one_hot = tf.tile(current_option_option_one_hot[..., None], [1, 1, 1, self.action_size])
        action_probabilities = tf.reduce_sum(tf.multiply(self.local_network.options, current_option_option_one_hot),
                                             reduction_indices=2, name="P_a")
        return action_probabilities

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

        feed_dict = {self.local_network.target_v: discounted_rewards,
                     self.local_network.inputs: np.stack(observations, axis=0),
                     self.local_network.actions: actions,
                     self.local_network.options: options}

        if summaries:
            l, v_l, p_l, e_l, g_n, v_n, _, ms, img_summ = sess.run([self.local_network.loss,
                                                                    self.local_network.value_loss,
                                                                    self.local_network.policy_loss,
                                                                    self.local_network.entropy,
                                                                    self.local_network.grad_norms,
                                                                    self.local_network.var_norms,
                                                                    self.local_network.apply_grads,
                                                                    self.local_network.merged_summary,
                                                                    self.local_network.image_summaries],
                                                                   feed_dict=feed_dict)
            return l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n, ms, img_summ
        else:
            _ = sess.run([self.local_network.apply_grads], feed_dict=feed_dict)
            return None

    def play(self, sess, coord, saver):
        episode_count = sess.run(self.global_step)

        # if not FLAGS.train:
        #     test_episode_count = 0
        total_steps = 0

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
                o = self.get_policy_over_options(sess, np.stack([s]))
                while not d:

                    o_t = self.get_o_term(sess, np.stack([s]), o)
                    v = self.get_v(sess, np.stack([s]))
                    q = self.get_q(sess, np.stack([s]))
                    a = self.get_action(sess, np.stack([s]), o)

                    feed_dict = {self.local_network.observation: np.stack([s])}
                    option, action, value, q_value, o_term = sess.run([o, a, v, q, o_t], feed_dict=feed_dict)
                    action, option, value, q_value, o_term = action[0], option[0], value[0], q_value[0], o_term[0]
                    s1, r, d, _ = self.env.step(action)

                    r = np.clip(r, -1, 1)
                    self.frame_counter += 1
                    processed_reward = float(r) - (float(o_term) * self.delib * float(self.frame_counter > 1))
                    episode_buffer.append([s, o, a, processed_reward, t, d, o_t, v[0], q[0]])
                    episode_values.append(v[0])
                    episode_q_values.append(q[0])
                    episode_reward += r
                    total_steps += 1
                    t += 1
                    s = s1

                    option_term = (o_term and t >= self.config.min_update_freq)
                    if t == self.config.max_update_freq or d or option_term:
                        delib_cost = self.delib * float(self.frame_counter > 1)
                        value = value - delib_cost if o_t else q_value
                        R = 0 if d else value
                        self.train(episode_buffer, sess, R)
                    if not d:
                        self.delib = self.config.delib_cost
                        if o_term:
                            o = self.get_policy_over_options(sess, np.stack([s]))


                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(t)
                self.episode_mean_values.append(np.mean(episode_values))
                self.episode_mean_q_values.append(np.mean(episode_q_values))

                if len(episode_buffer) != 0 and FLAGS.train == True:
                    if episode_count % FLAGS.summary_interval == 0 and episode_count != 0:
                        l, v_l, p_l, e_l, g_n, v_n, ms, img_sum = self.train(episode_buffer, sess, 0.0, summaries=True)
                    else:
                        self.train(episode_buffer, sess, 0.0)

                # if not FLAGS.train and test_episode_count == FLAGS.nb_test_episodes - 1:
                #     print("Mean reward for the model is {}".format(np.mean(self.episode_rewards)))
                #     return 1

                if FLAGS.train and episode_count % FLAGS.summary_interval == 0 and episode_count != 0 and \
                                self.name == 'worker_0':
                    if episode_count % FLAGS.checkpoint_interval == 0 and self.name == 'worker_0' and FLAGS.train == True:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                                   global_step=self.global_episode)
                        print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

                    mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                    mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
                    mean_value = np.mean(self.episode_mean_values[-FLAGS.summary_interval:])

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))

                    if FLAGS.train:
                        self.summary.value.add(tag='Losses/Total Loss', simple_value=float(l))
                        self.summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        self.summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        self.summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        self.summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        self.summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        summaries = tf.Summary().FromString(ms)
                        sub_summaries_dict = {}
                        for value in summaries.value:
                            value_field = value.WhichOneof('value')
                            value_ifo = sub_summaries_dict.setdefault(value.tag,
                                                                      {'value_field': None, 'values': []})
                            if not value_ifo['value_field']:
                                value_ifo['value_field'] = value_field
                            else:
                                assert value_ifo['value_field'] == value_field
                            value_ifo['values'].append(getattr(value, value_field))

                        for name, value_ifo in sub_summaries_dict.items():
                            summary_value = self.summary.value.add()
                            summary_value.tag = name
                            if value_ifo['value_field'] == 'histo':
                                values = value_ifo['values']
                                summary_value.histo.min = min([x.min for x in values])
                                summary_value.histo.max = max([x.max for x in values])
                                summary_value.histo.num = sum([x.num for x in values])
                                summary_value.histo.sum = sum([x.sum for x in values])
                                summary_value.histo.sum_squares = sum([x.sum_squares for x in values])
                                for lim in values[0].bucket_limit:
                                    summary_value.histo.bucket_limit.append(lim)
                                for bucket in values[0].bucket:
                                    summary_value.histo.bucket.append(bucket)
                            else:
                                print(
                                    'Warning: could not aggregate summary of type {}'.format(value_ifo['value_field']))
                    for s in img_sum:
                        self.summary_writer.add_summary(s, episode_count)
                    self.summary_writer.add_summary(self.summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment_global_episode)
                if not FLAGS.train:
                    test_episode_count += 1
                episode_count += 1
