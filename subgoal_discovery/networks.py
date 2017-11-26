from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
from utility import gradient_summaries, huber_loss
import numpy as np


class NNSFNetwork():
  def __init__(self, scope, config, action_size, nb_states):
    self._scope = scope
    # self.option = 0
    self.nb_states = nb_states
    self.conv_layers = config.conv_layers
    self.fc_layers = config.fc_layers
    self.sf_layers = config.sf_layers
    self.aux_fc_layers = config.aux_fc_layers
    self.aux_deconv_layers = config.aux_deconv_layers
    self.action_size = action_size
    self.nb_options = config.nb_options
    self.nb_envs = config.num_agents
    self.config = config

    self._network_optimizer = config.network_optimizer(
      self.config.lr, name='network_optimizer')

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, config.input_size[0], config.input_size[1], config.history_size],
                                        dtype=tf.float32, name="Inputs")
      self.image_summaries = []
      if self.config.history_size == 3:
        self.image_summaries.append(tf.summary.image('input', self.observation * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('input', self.observation[:, :, :, 0:1] * 255, max_outputs=30))
      self.summaries = []

      out = self.observation
      with tf.variable_scope('conv'):
        for i, (kernel_size, stride, nb_kernels) in enumerate(self.conv_layers):
          out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                              stride=stride, activation_fn=None,
                              variables_collections=tf.get_collection("variables"),
                              outputs_collections="activations", scope="conv_{}".format(i))
          out = layer_norm_fn(out, relu=True)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))
        out = layers.flatten(out, scope="flatten")

      with tf.variable_scope("fc"):
        for i, nb_filt in enumerate(self.fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_{}".format(i))

          if i < len(self.fc_layers) - 1:
            out = layer_norm_fn(out, relu=False)
            # out = layer_norm_fn(out, relu=True)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))
      self.fi = out

      out = tf.stop_gradient(layer_norm_fn(self.fi, relu=True))
      with tf.variable_scope("sf"):
        for i, nb_filt in enumerate(self.sf_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf_{}".format(i))
          if i < len(self.sf_layers) - 1:
            out = layer_norm_fn(out, relu=False)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      self.sf = out

      out = self.fi
      with tf.variable_scope("action_fc"):
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
        actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.fc_layers[-1],
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="action_fc{}".format(i))
        out = layer_norm_fn(out, relu=False)
      out = tf.add(out, actions)
      out = layer_norm_fn(out, relu=True)

      with tf.variable_scope("aux_fc"):
        for i, nb_filt in enumerate(self.aux_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="aux_fc_{}".format(i))
          out = layer_norm_fn(out, relu=False)
          if i > 0:
            # out = layer_norm_fn(out, relu=True)
            out= tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      with tf.variable_scope("aux_deconv"):
        decoder_out = tf.expand_dims(tf.expand_dims(out, 1), 1)
        for i, (kernel_size, stride, padding, nb_kernels) in enumerate(self.aux_deconv_layers):
          decoder_out = layers.conv2d_transpose(decoder_out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                                stride=stride, activation_fn=None,
                                                padding="same" if padding > 0 else "valid",
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="aux_deconv_{}".format(i))
          if i < len(self.aux_deconv_layers) - 1:
            decoder_out = layer_norm_fn(decoder_out, relu=False)
            decoder_out = tf.nn.relu(decoder_out)
          self.summaries.append(tf.contrib.layers.summarize_activation(decoder_out))

      self.next_obs = decoder_out

      if self.config.history_size == 3:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs[:, :, :, 0:1] * 255, max_outputs=30))

      if scope != 'global':
        self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
        self.target_next_obs = tf.placeholder(
          shape=[None, config.input_size[0], config.input_size[1], config.history_size], dtype=tf.float32,
          name="target_next_obs")
        if self.config.history_size == 3:
          self.image_summaries.append(tf.summary.image('target_next_obs', self.target_next_obs * 255, max_outputs=30))
        else:
          self.image_summaries.append(tf.summary.image('target_next_obs', self.target_next_obs[:, :, :, 0:1] * 255, max_outputs=30))
        self.matrix_sf = tf.placeholder(shape=[self.nb_states, self.sf_layers[-1]],
                                        dtype=tf.float32, name="matrix_sf")
        self.s, self.u, self.v = tf.svd(self.matrix_sf)

        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - self.sf
          self.sf_loss = tf.reduce_mean(tf.square(sf_td_error))

        with tf.name_scope('aux_loss'):
          aux_error = self.next_obs - self.target_next_obs
          self.aux_loss = tf.reduce_mean(self.config.aux_coef * tf.square(aux_error))

        self.loss = self.sf_loss + self.aux_loss
        loss_summaries = [tf.summary.scalar('avg_sf_loss', self.sf_loss),
                          tf.summary.scalar('aux_loss', self.aux_loss),
                          tf.summary.scalar('total_loss', self.loss)]

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_value)

        self.merged_summary = tf.summary.merge(self.image_summaries + self.summaries + loss_summaries + [
          tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
          tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads)),
          gradient_summaries(zip(grads, local_vars))])
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = self._network_optimizer.apply_gradients(zip(grads, global_vars))

class DQNSFNetwork:
  def __init__(self, scope, config, action_size, nb_states):
    self._scope = scope
    self.nb_states = nb_states
    self.conv_layers = config.conv_layers
    self.fc_layers = config.fc_layers
    self.sf_layers = config.sf_layers
    self.aux_fc_layers = config.aux_fc_layers
    self.aux_deconv_layers = config.aux_deconv_layers
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
      if self.config.history_size == 3:
        self.image_summaries.append(tf.summary.image('input', self.observation * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('input', self.observation[:, :, :, 0:1] * 255, max_outputs=30))
      self.summaries = []

      out = self.observation
      with tf.variable_scope('conv'):
        for i, (kernel_size, stride, nb_kernels) in enumerate(self.conv_layers):
          out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                              stride=stride, activation_fn=None,
                              variables_collections=tf.get_collection("variables"),
                              outputs_collections="activations", scope="conv_{}".format(i))
          out = layer_norm_fn(out, relu=True)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))
        out = layers.flatten(out, scope="flatten")

      with tf.variable_scope("fc"):
        for i, nb_filt in enumerate(self.fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_{}".format(i))

          if i < len(self.fc_layers) - 1:
            out = layer_norm_fn(out, relu=False)
            # out = layer_norm_fn(out, relu=True)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))
      self.fi = out

      out = tf.stop_gradient(layer_norm_fn(self.fi, relu=True))

      # ------------------- Adding option Q ---------------------
      self.q = layers.fully_connected(out, num_outputs=self.action_size + 1,
                                                              activation_fn=None,
                                                              variables_collections=tf.get_collection("variables"),
                                                              outputs_collections="activations", scope="Q")

      with tf.variable_scope("sf"):
        for i, nb_filt in enumerate(self.sf_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf_{}".format(i))
          if i < len(self.sf_layers) - 1:
            out = layer_norm_fn(out, relu=False)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      self.sf = out

      out = self.fi
      with tf.variable_scope("action_fc"):
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
        actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.fc_layers[-1],
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="action_fc{}".format(i))
        out = layer_norm_fn(out, relu=False)
      out = tf.add(out, actions)
      out = layer_norm_fn(out, relu=True)

      with tf.variable_scope("aux_fc"):
        for i, nb_filt in enumerate(self.aux_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="aux_fc_{}".format(i))
          out = layer_norm_fn(out, relu=False)
          if i > 0:
            # out = layer_norm_fn(out, relu=True)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      with tf.variable_scope("aux_deconv"):
        decoder_out = tf.expand_dims(tf.expand_dims(out, 1), 1)
        for i, (kernel_size, stride, padding, nb_kernels) in enumerate(self.aux_deconv_layers):
          decoder_out = layers.conv2d_transpose(decoder_out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                                stride=stride, activation_fn=None,
                                                padding="same" if padding > 0 else "valid",
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="aux_deconv_{}".format(i))
          if i < len(self.aux_deconv_layers) - 1:
            decoder_out = layer_norm_fn(decoder_out, relu=False)
            decoder_out = tf.nn.relu(decoder_out)
          self.summaries.append(tf.contrib.layers.summarize_activation(decoder_out))

      self.next_obs = decoder_out

      if self.config.history_size == 3:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs[:, :, :, 0:1] * 255, max_outputs=30))

      if scope != 'target':
        self.target_q_a = tf.placeholder(shape=[None], dtype=tf.float32, name="target_Q_a")
        self.actions_onehot = tf.one_hot(tf.cast(self.actions_placeholder, tf.int32), self.action_size + 1, dtype=tf.float32, name="actions_one_hot")
        self.q_a = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot),
                                          reduction_indices=1, name="Q_a")

        with tf.name_scope('q_loss'):
          td_error = self.q_a - self.target_q_a
          self.q_loss = tf.reduce_mean(huber_loss(td_error))

        self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
        self.target_next_obs = tf.placeholder(
          shape=[None, config.input_size[0], config.input_size[1], config.history_size], dtype=tf.float32,
          name="target_next_obs")
        if self.config.history_size == 3:
          self.image_summaries.append(tf.summary.image('target_next_obs', self.target_next_obs * 255, max_outputs=30))
        else:
          self.image_summaries.append(
            tf.summary.image('target_next_obs', self.target_next_obs[:, :, :, 0:1] * 255, max_outputs=30))
        self.matrix_sf = tf.placeholder(shape=[self.nb_states, self.sf_layers[-1]],
                                        dtype=tf.float32, name="matrix_sf")
        self.s, self.u, self.v = tf.svd(self.matrix_sf)

        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - self.sf
          self.sf_loss = tf.reduce_mean(tf.square(sf_td_error))

        with tf.name_scope('aux_loss'):
          aux_error = self.next_obs - self.target_next_obs
          self.aux_loss = tf.reduce_mean(self.config.aux_coef * tf.square(aux_error))

        self.loss = self.sf_loss + self.aux_loss
        loss_summaries = [tf.summary.scalar('avg_sf_loss', self.sf_loss),
                          tf.summary.scalar('aux_loss', self.aux_loss),
                          tf.summary.scalar('total_loss', self.loss)]

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        local_vars = [v for v in local_vars if 'Q' not in v.name]
        gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_value)

        self.merged_summary = tf.summary.merge(self.image_summaries + self.summaries + loss_summaries + [
          tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
          tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads)),
          gradient_summaries(zip(grads, local_vars))])
        self.apply_grads = self.network_optimizer.apply_gradients(zip(grads, local_vars))

        # --------------- Q_loss ----------------------
        local_q_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope) if "Q" in v.name]
        q_loss_summary = [tf.summary.scalar('q_loss', self.q_loss)]
        q_gradients = tf.gradients(self.q_loss, local_q_vars)
        q_grads, self.grad_norms = tf.clip_by_global_norm(q_gradients, self.config.gradient_clip_value)
        self.q_merged_summary = tf.summary.merge(q_loss_summary)
        self.q_apply_grads = self.network_optimizer.apply_gradients(zip(q_grads, local_q_vars))


class DQNSF_FCNetwork:
  def __init__(self, scope, config, action_size, nb_states):
    self._scope = scope
    self.nb_states = nb_states
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
      if self.config.history_size == 3:
        self.image_summaries.append(tf.summary.image('input', self.observation * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('input', self.observation[:, :, :, 0:1] * 255, max_outputs=30))
      self.summaries = []

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
          self.summaries.append(tf.contrib.layers.summarize_activation(out))
      self.fi = out

      out = tf.stop_gradient(tf.nn.relu(self.fi))

      # ------------------- Adding option Q ---------------------
      self.q = layers.fully_connected(out, num_outputs=self.action_size + 1,
                                      activation_fn=None,
                                      variables_collections=tf.get_collection("variables"),
                                      outputs_collections="activations", scope="Q")


      with tf.variable_scope("sf"):
        for i, nb_filt in enumerate(self.sf_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf_{}".format(i))
          if i < len(self.sf_layers) - 1:
            # out = layer_norm_fn(out, relu=True)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      self.sf = out

      out = self.fi
      with tf.variable_scope("action_fc"):
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
        actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.fc_layers[-1],
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="action_fc{}".format(i))
      out = tf.add(out, actions)
      out = tf.nn.relu(out)

      with tf.variable_scope("aux_fc"):
        for i, nb_filt in enumerate(self.aux_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="aux_fc_{}".format(i))
          if i > 0:
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      decoder_out = tf.reshape(out, (-1, config.input_size[0], config.input_size[1], config.history_size))
      self.next_obs = decoder_out

      if self.config.history_size == 3:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs[:, :, :, 0:1] * 255, max_outputs=30))

      if scope != 'target':
        self.target_q_a = tf.placeholder(shape=[None], dtype=tf.float32, name="target_Q_a")
        self.actions_onehot = tf.one_hot(tf.cast(self.actions_placeholder, tf.int32), self.action_size + 1,
                                         dtype=tf.float32, name="actions_one_hot")
        self.q_a = tf.reduce_sum(tf.multiply(self.q, self.actions_onehot),
                                 reduction_indices=1, name="Q_a")

        with tf.name_scope('q_loss'):
          td_error = self.q_a - self.target_q_a
          self.q_loss = tf.reduce_mean(huber_loss(td_error))

        self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
        self.target_next_obs = tf.placeholder(
          shape=[None, config.input_size[0], config.input_size[1], config.history_size], dtype=tf.float32,
          name="target_next_obs")
        if self.config.history_size == 3:
          self.image_summaries.append(tf.summary.image('target_next_obs', self.target_next_obs * 255, max_outputs=30))
        else:
          self.image_summaries.append(
            tf.summary.image('target_next_obs', self.target_next_obs[:, :, :, 0:1] * 255, max_outputs=30))
        self.matrix_sf = tf.placeholder(shape=[self.nb_states, self.sf_layers[-1]],
                                        dtype=tf.float32, name="matrix_sf")
        self.s, self.u, self.v = tf.svd(self.matrix_sf)



        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - self.sf
          self.sf_loss = tf.reduce_mean(tf.square(sf_td_error))

        with tf.name_scope('aux_loss'):
          aux_error = self.next_obs - self.target_next_obs
          self.aux_loss = tf.reduce_mean(self.config.aux_coef * tf.square(aux_error))

        self.loss = self.sf_loss + self.aux_loss
        loss_summaries = [tf.summary.scalar('avg_sf_loss', self.sf_loss),
                          tf.summary.scalar('aux_loss', self.aux_loss),
                          tf.summary.scalar('total_loss', self.loss)]

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_value)

        self.merged_summary = tf.summary.merge(self.image_summaries + self.summaries + loss_summaries + [
          tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
          tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads)),
          gradient_summaries(zip(grads, local_vars))])
        self.apply_grads = self.network_optimizer.apply_gradients(zip(grads, local_vars))

        # --------------- Q_loss ----------------------
        local_q_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope) if "Q" in v.name]
        q_loss_summary = [tf.summary.scalar('q_loss', self.q_loss)]
        q_gradients = tf.gradients(self.q_loss, local_q_vars)
        q_grads, self.grad_norms = tf.clip_by_global_norm(q_gradients, self.config.gradient_clip_value)
        self.q_merged_summary = tf.summary.merge(q_loss_summary)
        self.q_apply_grads = self.network_optimizer.apply_gradients(zip(q_grads, local_q_vars))

def layer_norm_fn(x, relu=True):
  x = layers.layer_norm(x, scale=True, center=True)
  if relu:
    x = tf.nn.relu(x)
  return x
