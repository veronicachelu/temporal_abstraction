import numpy as np
import tensorflow as tf
from tools.utils import update_target_graph, discount, set_image_bandit, set_image_bandit_11_arms, make_gif
import os
from agents.schedules import LinearSchedule, TFLinearSchedule
FLAGS = tf.app.flags.FLAGS


class AOCAgent():
    def __init__(self, game, thread_id, optimizer, global_step, config):
        self.name = "worker_" + str(thread_id)
        self.thread_id = thread_id
        self.optimizer = optimizer
        self.global_step = global_step
        self.increment_global_episode = self.global_step.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.config = config
        self.action_size = game.action_space.n
        self._network_optimizer = self.config.network_optimizer(
            self.config.lr, name='network_optimizer')
        self._exploration_options = TFLinearSchedule(self.config.explore_steps, self.config.final_random_action_prob,
                                                     self.config.initial_random_action_prob)
        self.summary_writer = tf.summary.FileWriter(FLAGS.logdir + "/worker_" + str(self.thread_id))
        self.summary = tf.Summary()

        self.local_network = config.network(self.name, config.conv_layers, config.fc_layers, 4, config.nb_options,
                                            config.num_agents)
        self._random = tf.random_uniform(shape=[(1)], minval=0., maxval=1., dtype=tf.float32)

        self.update_local_vars = update_target_graph('global', self.name)
        self.env = game

    def get_policy_over_options(self, sess, s):
        self.probability_of_random_option = self._exploration_options.value(self.global_step)
        max_options = tf.cast(tf.argmax(self.local_network.q_val[:, 0, :], 1), dtype=tf.int32)
        exp_options = tf.random_uniform(shape=[1], minval=0, maxval=self.config.nb_options,
                                        dtype=tf.int32)
        options = tf.where(self._random > self.probability_of_random_option, max_options, exp_options)

        return options

    def get_action(self, sess, s, o):
        current_option_option_one_hot = tf.one_hot(o, self.config.nb_options, name="options_one_hot")
        current_option_option_one_hot = current_option_option_one_hot[:, :, None]
        current_option_option_one_hot = tf.tile(current_option_option_one_hot, [1, 1, self.action_size])
        self.action_probabilities = tf.reduce_sum(tf.multiply(self.local_network.options[:, 0, :], current_option_option_one_hot),
                                                  reduction_indices=1, name="P_a")
        policy = tf.multinomial(tf.log(self.action_probabilities), 1)[:, 0]
        return policy
    # pi, v, rnn_state_new = sess.run(
    #     [self.local_network.policy, self.local_network.value, self.local_network.state_out], feed_dict=feed_dict)
    # a = np.random.choice(pi[0], p=pi[0])
    # a = np.argmax(pi == a)
    def train(self, rollout, sess, bootstrap_value, summaries=False):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        timesteps = rollout[:, 3]
        values = rollout[:, 5]

        if FLAGS.meta:
            prev_rewards = [0] + rewards[:-1].tolist()
            prev_actions = [0] + actions[:-1].tolist()

        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, FLAGS.gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        policy_target = discounted_rewards - value_plus[:-1]
        if FLAGS.gen_adv:
            td_residuals = rewards + FLAGS.gamma * value_plus[1:] - value_plus[:-1]
            advantages = discount(td_residuals, FLAGS.gamma)
            policy_target = advantages

        rnn_state = self.local_AC.state_init
        if FLAGS.meta:
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.inputs: np.stack(observations, axis=0),
                         self.local_AC.prev_rewards: prev_rewards,
                         self.local_AC.prev_actions: prev_actions,
                         self.local_AC.actions: actions,
                         self.local_AC.timestep: np.vstack(timesteps),
                         self.local_AC.advantages: policy_target,
                         self.local_AC.state_in[0]: rnn_state[0],
                         self.local_AC.state_in[1]: rnn_state[1]}
        else:
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.inputs: np.stack(observations, axis=0),
                         self.local_AC.actions: actions,
                         self.local_AC.advantages: policy_target,
                         self.local_AC.state_in[0]: rnn_state[0],
                         self.local_AC.state_in[1]: rnn_state[1]}

        if summaries:
            l, v_l, p_l, e_l, g_n, v_n, _, ms, img_summ = sess.run([self.local_AC.loss,
                                                                    self.local_AC.value_loss,
                                                                    self.local_AC.policy_loss,
                                                                    self.local_AC.entropy,
                                                                    self.local_AC.grad_norms,
                                                                    self.local_AC.var_norms,
                                                                    self.local_AC.apply_grads,
                                                                    self.local_AC.merged_summary,
                                                                    self.local_AC.image_summaries],
                                                                   feed_dict=feed_dict)
            return l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n, ms, img_summ
        else:
            _ = sess.run([self.local_AC.apply_grads], feed_dict=feed_dict)
            return None

    def play(self, sess, saver):
        episode_count = sess.run(self.global_step)

        if not FLAGS.train:
            test_episode_count = 0

        total_steps = 0

        print("Starting worker " + str(self.thread_id))
        with sess.as_default(), sess.graph.as_default():
            while episode_count < self.config.steps:
                sess.run(self.update_local_vars)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                d = False
                t = 0
                o_t = True

                s, _, _, _ = self.env.reset()
                while not d:
                    o, v = self.get_policy_over_options(sess, [s])
                    a = self.get_action(sess, s, o)

                    feed_dict = {self.local_network.inputs: [s]}
                    option, action = sess.run([o, a], feed_dict=feed_dict)

                    s1, r, d, _ = self.env.step(action)

                    episode_buffer.append([s, o, a, r, t, d, o_t, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    total_steps += 1
                    t += 1
                    s = s1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(t)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0 and FLAGS.train == True:
                    if episode_count % FLAGS.summary_interval == 0 and episode_count != 0:
                        l, v_l, p_l, e_l, g_n, v_n, ms, img_sum = self.train(episode_buffer, sess, 0.0, summaries=True)
                    else:
                        self.train(episode_buffer, sess, 0.0)

                if not FLAGS.train and test_episode_count == FLAGS.nb_test_episodes - 1:
                    print("Mean reward for the model is {}".format(np.mean(self.episode_rewards)))
                    return 1

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
