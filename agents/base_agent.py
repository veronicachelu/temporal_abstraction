import numpy as np
import tensorflow as tf
from tools.agent_utils import get_mode, update_target_graph_aux, update_target_graph_sf, \
  update_target_graph_option, discount, reward_discount, set_image, make_gif
import os
import matplotlib.patches as patches
import matplotlib.pylab as plt
import numpy as np
from collections import deque
import seaborn as sns
sns.set()
import random
import matplotlib.pyplot as plt
FLAGS = tf.app.flags.FLAGS


class BaseAgent():
  def __init__(self, game, thread_id, global_step, config, global_network):
    self.name = "worker_" + str(thread_id)
    self.config = config
    self.thread_id = thread_id
    self.optimizer = config.network_optimizer
    self.global_step = global_step
    self.model_path = os.path.join(config.stage_logdir, "models")
    self.summary_path = os.path.join(config.stage_logdir, "summaries")
    self.test_path = os.path.join(config.stage_logdir, "test")
    tf.gfile.MakeDirs(self.test_path)
    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)
    self.global_network = global_network
    if config.sr_matrix is not None:
      self.directions = self.global_network.directions

    self.increment_global_step = self.global_step.assign_add(1)
    self.episode_rewards = []
    self.episode_lengths = []
    self.episode_mean_values = []
    self.episode_mean_q_values = []
    self.episode_mean_eigen_q_values = []
    self.episode_mean_returns = []
    self.episode_mean_oterms = []
    self.episode_mean_options = []
    self.episode_mean_actions = []
    self.episode_mean_options_lengths = np.zeros(self.config.nb_options)
    self.episode_options = []
    self.episode_actions = []

    self.total_steps_tensor = tf.Variable(0, dtype=tf.int32, name='total_steps_tensor', trainable=False)
    self.increment_total_steps_tensor = self.total_steps_tensor.assign_add(1)
    self.total_steps = 0
    self.action_size = game.action_space.n
    self.nb_options = config.nb_options
    self.nb_states = config.input_size[0] * config.input_size[1]
    self.summary_writer = tf.summary.FileWriter(self.summary_path + "/worker_" + str(self.thread_id))

    self.local_network = config.network(self.name, config, self.action_size, self.total_steps_tensor)

    self.update_local_vars_aux = update_target_graph_aux('global', self.name)
    self.update_local_vars_sf = update_target_graph_sf('global', self.name)
    self.update_local_vars_option = update_target_graph_option('global', self.name)
    self.env = game

  def load_directions(self):
    self.directions = self.global_network.directions

  def sync_threads(self, force=False):
    if force:
      self.sess.run(self.update_local_vars_aux)
      self.sess.run(self.update_local_vars_sf)
      self.sess.run(self.update_local_vars_option)
    else:
      if self.total_steps % self.config.target_update_iter_aux == 0:
        self.sess.run(self.update_local_vars_aux)
      if self.total_steps % self.config.target_update_iter_sf == 0:
        self.sess.run(self.update_local_vars_sf)
      if self.total_steps % self.config.target_update_iter_option == 0:
        self.sess.run(self.update_local_vars_option)

  def log_timestep(self):
    if self.name == "worker_0":
      tf.logging.info(
        "Episode {} >> Step {} >> Length: {}".format(self.episode_count, self.total_steps, self.episode_len))

  def log_episode(self):
    if self.name == "worker_0":
      tf.logging.info("Episode {} >> Step {} >> Length: {} >>> Reward: {}".format(self.episode_count,
                                                                                  self.total_steps, self.episode_len,
                                                                                  self.episode_reward))

  def update_episode_stats(self):
    self.episode_rewards.append(self.episode_reward)
    self.episode_lengths.append(self.episode_len)
    if len(self.episode_values) != 0:
      self.episode_mean_values.append(np.mean(self.episode_values))
    if len(self.episode_q_values) != 0:
      self.episode_mean_q_values.append(np.mean(self.episode_q_values))
    if self.config.eigen and len(self.episode_eigen_q_values) != 0:
      self.episode_mean_eigen_q_values.append(np.mean(self.episode_eigen_q_values))
    if len(self.episode_oterm) != 0:
      self.episode_mean_oterms.append(get_mode(self.episode_oterm))
    if len(self.episode_options) != 0:
      self.episode_mean_options.append(get_mode(self.episode_options))
    if len(self.episode_actions) != 0:
      self.episode_mean_actions.append(get_mode(self.episode_actions))
    for op, option_lengths in enumerate(self.episode_options_lengths):
      if len(option_lengths) != 0:
        self.episode_mean_options_lengths[op] = np.mean(option_lengths)

  def save_model(self):
    self.saver.save(self.sess, self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps),
                    global_step=self.global_step)
    tf.logging.info(
      "Saved Model at {}".format(self.model_path + '/model-{}.{}.cptk'.format(self.episode_count, self.total_steps)))

  def write_step_summary(self, ms_sf, ms_aux, ms_option, r):
    self.summary = tf.Summary()
    if ms_sf is not None:
      self.summary_writer.add_summary(ms_sf, self.total_steps)
    if ms_aux is not None:
      self.summary_writer.add_summary(ms_aux, self.total_steps)
    if ms_option is not None:
      self.summary_writer.add_summary(ms_option, self.total_steps)

    if self.total_steps > self.config.eigen_exploration_steps:
      self.summary.value.add(tag='Step/Reward', simple_value=r)
      self.summary.value.add(tag='Step/Action', simple_value=self.action)
      self.summary.value.add(tag='Step/Option', simple_value=self.option)
      self.summary.value.add(tag='Step/Q', simple_value=self.q_value)
      if self.config.eigen and not self.primitive_action and self.eigen_q_value is not None and self.evalue is not None:
        self.summary.value.add(tag='Step/EigenQ', simple_value=self.eigen_q_value)
        self.summary.value.add(tag='Step/EigenV', simple_value=self.evalue)
      self.summary.value.add(tag='Step/V', simple_value=self.value)
      self.summary.value.add(tag='Step/Term', simple_value=int(self.o_term))
      self.summary.value.add(tag='Step/R', simple_value=self.R)
      if self.config.eigen:
        self.summary.value.add(tag='Step/EigenR', simple_value=self.eigen_R)

    self.summary_writer.add_summary(self.summary, self.total_steps)
    self.summary_writer.flush()
    # tf.logging.warning("Writing step summary....")

  def write_episode_summary(self, ms_sf, ms_aux, ms_option, r):
    self.summary = tf.Summary()
    if len(self.episode_rewards) != 0:
      last_reward = self.episode_rewards[-1]
      self.summary.value.add(tag='Perf/Reward', simple_value=float(last_reward))
    if len(self.episode_lengths) != 0:
      last_length = self.episode_lengths[-1]
      self.summary.value.add(tag='Perf/Length', simple_value=float(last_length))
    if len(self.episode_mean_values) != 0:
      last_mean_value = self.episode_mean_values[-1]
      self.summary.value.add(tag='Perf/Value', simple_value=float(last_mean_value))
    if len(self.episode_mean_q_values) != 0:
      last_mean_q_value = self.episode_mean_q_values[-1]
      self.summary.value.add(tag='Perf/QValue', simple_value=float(last_mean_q_value))
    if self.config.eigen and len(self.episode_mean_eigen_q_values) != 0:
      last_mean_eigen_q_value = self.episode_mean_eigen_q_values[-1]
      self.summary.value.add(tag='Perf/EigenQValue', simple_value=float(last_mean_eigen_q_value))
    if len(self.episode_mean_oterms) != 0:
      last_mean_oterm = self.episode_mean_oterms[-1]
      self.summary.value.add(tag='Perf/Oterm', simple_value=float(last_mean_oterm))
    if len(self.episode_mean_options) != 0:
      last_frequent_option = self.episode_mean_options[-1]
      self.summary.value.add(tag='Perf/FreqOptions', simple_value=last_frequent_option)
    if len(self.episode_mean_options) != 0:
      last_frequent_action = self.episode_mean_actions[-1]
      self.summary.value.add(tag='Perf/FreqActions', simple_value=last_frequent_action)
    for op in range(self.config.nb_options):
      self.summary.value.add(tag='Perf/Option_length_{}'.format(op), simple_value=self.episode_mean_options_lengths[op])

    self.summary_writer.add_summary(self.summary, self.episode_count)
    self.summary_writer.flush()
    self.write_step_summary(ms_sf, ms_aux, ms_option, r)

  def cosine_similarity(self, next_sf, evect):
    state_dif_norm = np.linalg.norm(next_sf)
    state_dif_normalized = next_sf / (state_dif_norm + 1e-8)
    # evect_norm = np.linalg.norm(evect)
    # evect_normalized = evect / (evect_norm + 1e-8)
    res = np.dot(state_dif_normalized, evect)
    return res

  def write_eval_summary(self, eval_episodes_won, mean_ep_length):
    self.summary = tf.Summary()
    self.summary.value.add(tag='Eval/Episodes_won(of 100)', simple_value=float(eval_episodes_won))
    self.summary.value.add(tag='Eval/Mean eval episodes length', simple_value=float(mean_ep_length))
    self.summary_writer.add_summary(self.summary, self.episode_count)
    self.summary_writer.flush()

  def eval(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.sess = sess
      self.saver = saver
      self.episode_count = sess.run(self.global_step)
      self.total_steps = sess.run(self.total_steps_tensor)

      tf.logging.info("Starting eval agent")
      ep_rewards = []
      ep_lengths = []
      # episode_frames = []
      for i in range(self.config.nb_test_ep):
        episode_reward = 0
        s = self.env.reset()
        feed_dict = {self.local_network.observation: np.stack([s])}
        option, primitive_action = self.sess.run(
          [self.local_network.max_options, self.local_network.primitive_action], feed_dict=feed_dict)
        option, primitive_action = option[0], primitive_action[0]
        primitive_action = option >= self.config.nb_options
        d = False
        episode_length = 0
        while not d:
          feed_dict = {self.local_network.observation: np.stack([s])}
          options, o_term = self.sess.run([self.local_network.options, self.local_network.termination],
                                          feed_dict=feed_dict)

          if primitive_action:
            action = option - self.nb_options
            o_term = True
          else:
            pi = options[0, option]
            action = np.random.choice(pi, p=pi)
            action = np.argmax(pi == action)
            o_term = o_term[0, option] > np.random.uniform()

          # episode_frames.append(set_image(s, option, action, episode_length, primitive_action))
          s1, r, d, _ = self.env.step(action)

          r = np.clip(r, -1, 1)
          episode_reward += r
          episode_length += 1

          if not d and (o_term or primitive_action):
            feed_dict = {self.local_network.observation: np.stack([s1])}
            option, primitive_action = self.sess.run(
              [self.local_network.max_options, self.local_network.primitive_action], feed_dict=feed_dict)
            option, primitive_action = option[0], primitive_action[0]
            primitive_action = option >= self.config.nb_options
          s = s1
          if episode_length > self.config.max_length_eval:
            break

        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_length)
        tf.logging.info("Ep {} finished in {} steps with reward {}".format(i, episode_length, episode_reward))
      images = np.array(episode_frames)
      # make_gif(images, os.path.join(self.test_path, 'test_episodes.gif'),
      #          duration=len(images) * 1.0, true_image=True)
      tf.logging.info("Won {} episodes of {}".format(ep_rewards.count(1), self.config.nb_test_ep))

  def viz_options(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.sess = sess
      self.saver = saver
      folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "policies")
      tf.gfile.MakeDirs(folder_path)
      matrix_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "matrix.npy")
      self.matrix_sf = np.load(matrix_path)
      u, s, v = np.linalg.svd(self.matrix_sf)
      eigenvalues = s[1:1 + self.nb_options]
      eigenvectors = v[1:1 + self.nb_options]
      plt.clf()

      with sess.as_default(), sess.graph.as_default():
        for idx in range(self.nb_states):
          dx = 0
          dy = 0
          d = False
          s, i, j = self.env.get_state(idx)
          if not self.env.not_wall(i, j):
            plt.gca().add_patch(
              patches.Rectangle(
                (j, self.config.input_size[0] - i - 1),  # (x,y)
                1.0,  # width
                1.0,  # height
                facecolor="gray"
              )
            )
            continue

          feed_dict = {self.local_network.observation: np.stack([s])}
          max_q_val, q_vals, option, primitive_action, options, o_term = self.sess.run(
            [self.local_network.max_q_val, self.local_network.q_val, self.local_network.max_options,
             self.local_network.primitive_action, self.local_network.options, self.local_network.termination],
            feed_dict=feed_dict)
          max_q_val = max_q_val[0]
          # q_vals = q_vals[0]

          o, primitive_action = option[0], primitive_action[0]
          # q_val = q_vals[o]
          primitive_action = o >= self.config.nb_options
          if primitive_action:
            a = o - self.nb_options
            o_term = True
          else:
            pi = options[0, o]
            action = np.random.choice(pi, p=pi)
            a = np.argmax(pi == action)
            o_term = o_term[0, o] > np.random.uniform()

          if a == 0:  # up
            dy = 0.35
          elif a == 1:  # right
            dx = 0.35
          elif a == 2:  # down
            dy = -0.35
          elif a == 3:  # left
            dx = -0.35

          if o_term and not primitive_action:  # termination
            circle = plt.Circle(
              (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.025, color='r' if primitive_action else 'k')
            plt.gca().add_artist(circle)
            continue
          plt.text(j, self.config.input_size[0] - i - 1, str(o), color='r' if primitive_action else 'b', fontsize=8)
          plt.text(j + 0.5, self.config.input_size[0] - i - 1, '{0:.2f}'.format(max_q_val), fontsize=8)

          plt.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, dx, dy,
                    head_width=0.05, head_length=0.05, fc='r' if primitive_action else 'k',
                    ec='r' if primitive_action else 'k')

        plt.xlim([0, self.config.input_size[1]])
        plt.ylim([0, self.config.input_size[0]])

        for i in range(self.config.input_size[1]):
          plt.axvline(i, color='k', linestyle=':')
        plt.axvline(self.config.input_size[1], color='k', linestyle=':')

        for j in range(self.config.input_size[0]):
          plt.axhline(j, color='k', linestyle=':')
        plt.axhline(self.config.input_size[0], color='k', linestyle=':')

        plt.savefig(os.path.join(self.summary_path, 'Training_policy.png'))
        plt.close()

  def viz_options2(self, sess, coord, saver):
    with sess.as_default(), sess.graph.as_default():
      self.sess = sess
      self.saver = saver
      folder_path = os.path.join(os.path.join(self.config.stage_logdir, "summaries"), "policies")
      tf.gfile.MakeDirs(folder_path)
      matrix_path = os.path.join(os.path.join(self.config.stage_logdir, "models"), "matrix.npy")
      self.matrix_sf = np.load(matrix_path)
      u, s, v = np.linalg.svd(self.matrix_sf)
      eigenvalues = s[1:1 + self.nb_options]
      eigenvectors = v[1:1 + self.nb_options]

      for option in range(len(eigenvalues)):
        prefix = str(option) + '_'
        plt.clf()

        with sess.as_default(), sess.graph.as_default():
          for idx in range(self.nb_states):
            dx = 0
            dy = 0
            d = False
            s, i, j = self.env.get_state(idx)
            if not self.env.not_wall(i, j):
              plt.gca().add_patch(
                patches.Rectangle(
                  (j, self.config.input_size[0] - i - 1),  # (x,y)
                  1.0,  # width
                  1.0,  # height
                  facecolor="gray"
                )
              )
              continue

            feed_dict = {self.local_network.observation: np.stack([s])}
            fi, options, o_term = sess.run(
              [self.local_network.fi, self.local_network.options, self.local_network.termination],
              feed_dict=feed_dict)
            fi, options, o_term = fi[0], options[0], o_term[0]
            pi = options[option]
            action = np.random.choice(pi, p=pi)
            a = np.argmax(pi == action)
            o_term = o_term[option] > np.random.uniform()
            if a == 0:  # up
              dy = 0.35
            elif a == 1:  # right
              dx = 0.35
            elif a == 2:  # down
              dy = -0.35
            elif a == 3:  # left
              dx = -0.35

            if o_term:  # termination
              circle = plt.Circle(
                (j + 0.5, self.config.input_size[0] - i + 0.5 - 1), 0.025, color='k')
              plt.gca().add_artist(circle)
              continue

            plt.arrow(j + 0.5, self.config.input_size[0] - i + 0.5 - 1, dx, dy,
                      head_width=0.05, head_length=0.05, fc='k', ec='k')

          plt.xlim([0, self.config.input_size[1]])
          plt.ylim([0, self.config.input_size[0]])

          for i in range(self.config.input_size[1]):
            plt.axvline(i, color='k', linestyle=':')
          plt.axvline(self.config.input_size[1], color='k', linestyle=':')

          for j in range(self.config.input_size[0]):
            plt.axhline(j, color='k', linestyle=':')
          plt.axhline(self.config.input_size[0], color='k', linestyle=':')

          plt.savefig(os.path.join(self.summary_path, "Option_" + prefix + 'policy.png'))
          plt.close()
