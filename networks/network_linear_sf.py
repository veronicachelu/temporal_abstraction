import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
import os

class LinearSFNetwork():
  def __init__(self, scope, config, action_size):
    self._scope = scope
    self.nb_states = config.input_size[0] * config.input_size[1]
    self._conv_layers = config.conv_layers
    self._fc_layers = config.fc_layers
    self._action_size = action_size
    self._nb_options = config.nb_options
    self._nb_envs = config.num_agents
    self._config = config
    self.option = 0
    self._sf_layers = config.sf_layers
    self._deconv_layers = config.deconv_layers
    self._network_optimizer = config.network_optimizer(
      self._config.lr, name='network_optimizer')

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, self.nb_states],
                                        dtype=tf.float32, name="Inputs")
      self.sf = layers.fully_connected(self.observation, num_outputs=self.nb_states,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf")
      if scope != 'global':
        self.target_sf = tf.placeholder(shape=[None, self.nb_states], dtype=tf.float32, name="target_SF")

        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - self.sf
          self.sf_loss = tf.reduce_mean(tf.square(sf_td_error))

        self.loss = self.sf_loss  # + self.instant_r_loss
        loss_summaries = [tf.summary.scalar('avg_sf_loss', self.sf_loss)]

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(gradients, self._config.gradient_clip_value)

        # for grad, weight in zip(grads, local_vars):
        #   if grad is not None:
        #     self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
        #     self.summaries.append(tf.summary.histogram(weight.name, weight))

        self.merged_summary = tf.summary.merge(loss_summaries + [
          tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
          tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads)),
          gradient_summaries(zip(grads, local_vars))])
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = self._network_optimizer.apply_gradients(zip(grads, global_vars))

