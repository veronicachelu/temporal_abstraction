import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
import os


class DynSRNetwork():
  def __init__(self, scope, config, action_size):
    self._scope = scope
    # self.option = 0
    self.nb_states = config.input_size[0] * config.input_size[1]
    # self.conv_layers = config.conv_layers
    self.fc_layers = config.fc_layers
    self.sf_layers = config.sf_layers
    self.aux_fc_layers = config.aux_fc_layers
    # self.aux_deconv_layers = config.aux_deconv_layers
    self.action_size = action_size
    self.nb_options = config.nb_options
    self.nb_envs = config.num_agents
    self.config = config

    self.network_optimizer = config.network_optimizer(
      self.config.lr, name='network_optimizer')

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, config.input_size[0], config.input_size[1], config.history_size],
                                        dtype=tf.float32, name="Inputs")

      self.image_summaries = []
      self.image_summaries.append(tf.summary.image('input', self.observation, max_outputs=30))

      self.summaries_sf = []
      self.summaries_aux = []

      out = self.observation
      out = layers.flatten(out, scope="flatten")

      with tf.variable_scope("fc"):
        for i, nb_filt in enumerate(self.fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_{}".format(i))

          if i < len(self.fc_layers) - 1:
            # out = layer_norm_fn(out, relu=True)
            out = tf.nn.relu(out)
          self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
          self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.fi = out

      with tf.variable_scope("sf"):
        out = self.layer_norm_fn(self.fi, relu=True)
        out = tf.stop_gradient(out)
        for i, nb_filt in enumerate(self.sf_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf_{}".format(i))
          if i < len(self.sf_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
        self.sf = out

      with tf.variable_scope("action_fc"):
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
        actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.fc_layers[-1],
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="action_fc{}".format(i))

      with tf.variable_scope("aux_fc"):
        out = tf.add(self.fi, actions)
        # out = tf.nn.relu(out)
        for i, nb_filt in enumerate(self.aux_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="aux_fc_{}".format(i))
          if i < len(self.aux_fc_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.next_obs = tf.reshape(out, (-1, config.input_size[0], config.input_size[1], config.history_size))

        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs, max_outputs=30))

      if scope != 'global':
        self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
        self.target_next_obs = tf.placeholder(
          shape=[None, config.input_size[0], config.input_size[1], config.history_size], dtype=tf.float32,
          name="target_next_obs")
        self.image_summaries.append(tf.summary.image('target_next_obs', self.target_next_obs, max_outputs=30))

        self.matrix_sf = tf.placeholder(shape=[self.nb_states, self.sf_layers[-1]],
                                        dtype=tf.float32, name="matrix_sf")
        self.s, self.u, self.v = tf.svd(self.matrix_sf)

        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - self.sf
          self.sf_loss = tf.reduce_mean(huber_loss(sf_td_error))

        with tf.name_scope('aux_loss'):
          aux_error = self.next_obs - self.target_next_obs
          self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

        # regularizer_features = tf.reduce_mean(self.config.feat_decay * tf.nn.l2_loss(self.fi))
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        gradients_sf = tf.gradients(self.sf_loss, local_vars)
        gradients_aux = tf.gradients(self.aux_loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads_sf, self.grad_norms_sf = tf.clip_by_global_norm(gradients_sf, self.config.gradient_clip_norm_value)
        grads_aux, self.grad_norms_aux = tf.clip_by_global_norm(gradients_aux, self.config.gradient_clip_norm_value)

        self.merged_summary_sf = tf.summary.merge(
          self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss)] + [
            tf.summary.scalar('gradient_norm_sf', tf.global_norm(gradients_sf)),
            tf.summary.scalar('cliped_gradient_norm_sf', tf.global_norm(grads_sf)),
            gradient_summaries(zip(grads_sf, local_vars))])
        self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                                   [tf.summary.scalar('aux_loss', self.aux_loss)] + [
                                                     tf.summary.scalar('gradient_norm_sf',
                                                                       tf.global_norm(gradients_aux)),
                                                     tf.summary.scalar('cliped_gradient_norm_sf',
                                                                       tf.global_norm(grads_aux)),
                                                     gradient_summaries(zip(grads_aux, local_vars))])
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads_sf = self.network_optimizer.apply_gradients(zip(grads_sf, global_vars))
        self.apply_grads_aux = self.network_optimizer.apply_gradients(zip(grads_aux, global_vars))

  def layer_norm_fn(self, x, relu=True):
    x = layers.layer_norm(x, scale=True, center=True)
    if relu:
      x = tf.nn.relu(x)
    return x