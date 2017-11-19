import tensorflow as tf
import numpy as np
from collections import deque
import os
import pickle
from .base_vis_agent import BaseVisAgent

FLAGS = tf.app.flags.FLAGS


class DQNSFBaseAgent(BaseVisAgent):
  def __init__(self, game, _, global_step, config):
    self.global_step = global_step
    self.optimizer = config.network_optimizer
    self.increment_global_step = self.global_step.assign_add(1)
    self.increment_batch_global_step = self.global_step.assign_add(100)
    self.model_path = os.path.join(config.stage_logdir, "models")
    self.summary_path = os.path.join(config.stage_logdir, "summaries")
    self.buffer_path = os.path.join(self.model_path, "buffer")

    self.config = config
    if os.path.exists(self.buffer_path):
      self.load_buffer()
      self.buf_counter = self.episode_buffer['counter']
    else:
      tf.gfile.MakeDirs(self.buffer_path)
      self.episode_buffer = {
        'counter': 0,
        'observations': np.zeros(
          (self.config.observation_steps, config.input_size[0], config.input_size[1], config.history_size)),
        'fi': np.zeros((self.config.observation_steps, self.config.sf_layers[-1])),
        'next_observations': np.zeros(
          (self.config.observation_steps, config.input_size[0], config.input_size[1], config.history_size)),
        'actions': np.zeros(
          (self.config.observation_steps,)),
        'done': np.zeros(
          (self.config.observation_steps,)),
      }
      self.buf_counter = 0

    tf.gfile.MakeDirs(self.model_path)
    tf.gfile.MakeDirs(self.summary_path)

    self.action_size = game.action_space.n
    self.actions = np.zeros([self.action_size])
    self.nb_states = game.nb_states
    self.summary_writer = tf.summary.FileWriter(self.summary_path)
    self.summary = tf.Summary()
    self.env = game

  def update_target_graph_tao(self, from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
      op_holder.append(to_var.assign((1 - FLAGS.TAO) * to_var.value() + FLAGS.TAO * from_var.value()))
    return op_holder

  def update_target_graph(self, from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
      op_holder.append(to_var.assign(from_var))
    return op_holder

  def save_model(self, sess, saver, episode_count):
    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
               global_step=self.global_step)

    print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

  def save_buffer(self):
    # with open(self.buffer_path, 'wb') as output:
    #   pickle.dump(self.episode_buffer, output, protocol=4)
    np.save(os.path.join(self.buffer_path, "observations.npy"), self.episode_buffer["observations"])
    np.save(os.path.join(self.buffer_path, "fi.npy"), self.episode_buffer["fi"])
    np.save(os.path.join(self.buffer_path, "next_observations.npy"), self.episode_buffer["next_observations"])
    np.save(os.path.join(self.buffer_path, "actions.npy"), self.episode_buffer["actions"])
    np.save(os.path.join(self.buffer_path, "done.npy"), self.episode_buffer["done"])
    np.save(os.path.join(self.buffer_path, "buff_counter.npy"), self.episode_buffer["counter"])

  def load_buffer(self):
    self.episode_buffer = {
      'counter': np.load(os.path.join(self.buffer_path, "buff_counter.npy")),
      'observations': np.load(os.path.join(self.buffer_path, "observations.npy")),
      'fi': np.load(os.path.join(self.buffer_path, "fi.npy")),
      'next_observations': np.load(os.path.join(self.buffer_path, "next_observations.npy")),
      'actions': np.load(os.path.join(self.buffer_path, "actions.npy")),
      'done': np.load(os.path.join(self.buffer_path, "done.npy")),
    }
    # with open(self.buffer_path, "rb") as fp:
    #   self.episode_buffer = pickle.load(fp)

