from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
from utility import gradient_summaries
import numpy as np
from agents.schedules import LinearSchedule, TFLinearSchedule


class AOCNetwork(tf.contrib.rnn.RNNCell):
  def __init__(self, scope, config, action_size):
    self._scope = scope
    self._conv_layers = config.conv_layers
    self._fc_layers = config.fc_layers
    self._action_size = action_size
    self._nb_options = config.nb_options
    self._nb_envs = config.num_agents
    self._config = config
    self._network_optimizer = config.network_optimizer(
      self._config.lr, name='network_optimizer')
    self._exploration_options = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                                 self._config.initial_random_action_prob)

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, config.input_size[0], config.input_size[1], config.history_size],
                                        dtype=tf.float32, name="Inputs")
      self.total_steps = tf.placeholder(shape=[], dtype=tf.int32, name="total_steps")

      if self._config.history_size == 3:
        self.image_summaries = tf.summary.image('input', self.observation * 255, max_outputs=30)
      else:
        self.image_summaries = tf.summary.image('input', self.observation[:, :, :, 0:1] * 255, max_outputs=30)
      self.summaries = []
      with tf.variable_scope('conv'):
        for i, (kernel_size, stride, nb_kernels) in enumerate(self._conv_layers):
          out = layers.conv2d(self.observation, num_outputs=nb_kernels, kernel_size=kernel_size,
                              stride=stride, activation_fn=tf.nn.relu,
                              variables_collections=tf.get_collection("variables"),
                              outputs_collections="activations", scope="conv_{}".format(i))
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

        out = layers.flatten(out, scope="flatten")
        with tf.variable_scope("fc"):
          for i, nb_filt in enumerate(self._fc_layers):
            out = layers.fully_connected(out, num_outputs=nb_filt,
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="fc_{}".format(i))
            out = layer_norm_fn(out, relu=True)
            self.summaries.append(tf.contrib.layers.summarize_activation(out))

        with tf.variable_scope("option_term"):
          self.termination = layers.fully_connected(out, num_outputs=self._nb_options,
                                                    activation_fn=tf.nn.sigmoid,
                                                    variables_collections=tf.get_collection("variables"),
                                                    outputs_collections="activations", scope="fc_option_term")
          self.summaries.append(tf.contrib.layers.summarize_activation(self.termination))

        with tf.variable_scope("q_val"):
          self.q_val = layers.fully_connected(out, num_outputs=self._nb_options,
                                              activation_fn=None,
                                              variables_collections=tf.get_collection("variables"),
                                              outputs_collections="activations", scope="fc_q_val")
          self.summaries.append(tf.contrib.layers.summarize_activation(self.q_val))

          max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
          exp_options = tf.random_uniform(shape=[1], minval=0, maxval=self._config.nb_options,
                                          dtype=tf.int32)
          local_random = tf.random_uniform(shape=[1], minval=0., maxval=1., dtype=tf.float32,
                                           name="rand_options")
          # probability_of_random_option = self._exploration_options.value(self.total_steps)
          probability_of_random_option = self._config.final_random_action_prob
          # condition = local_random > tf.tile(probability_of_random_option[None, ...], [1])
          condition = local_random > probability_of_random_option

          self.current_option = tf.where(condition, max_options, exp_options)
          self.v = tf.reduce_max(self.q_val, axis=1) * (1 - probability_of_random_option) + \
                   probability_of_random_option * tf.reduce_mean(self.q_val, axis=1)

        with tf.variable_scope("i_o_policies"):
          self.options = []
          for i in range(self._nb_options):
            option = layers.fully_connected(out, num_outputs=self._action_size,
                                            activation_fn=tf.nn.softmax,
                                            variables_collections=tf.get_collection("variables"),
                                            outputs_collections="activations", scope="option_{}".format(i))

            self.summaries.append(tf.contrib.layers.summarize_activation(option))
            self.options.append(option)
          self.options = tf.stack(self.options, 1)

        if scope != 'global':
          self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
          self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Options")
          self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
          # self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
          self.delib = tf.placeholder(shape=[None], dtype=tf.float32)

          self.policy = self.get_intra_option_policy(self.options_placeholder)
          self.responsible_outputs = self.get_responsible_outputs(self.policy, self.actions_placeholder)
          q_val = self.get_q(self.options_placeholder)
          termination = self.get_o_term(self.options_placeholder)

          with tf.name_scope('critic_loss'):
            td_error = self.target_return - q_val
            self.critic_loss = tf.reduce_mean(self._config.critic_coef * tf.square(td_error))
          with tf.name_scope('termination_loss'):
            self.term_loss = tf.reduce_mean(
              termination * (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + self.delib))
          with tf.name_scope('entropy_loss'):
            self.entropy_loss = -self._config.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.policy *
                                                                                          tf.log(self.policy +
                                                                                                 1e-7), axis=1))
          with tf.name_scope('policy_loss'):
            self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_outputs + 1e-7) * tf.stop_gradient(td_error))

          self.loss = self.policy_loss - self.entropy_loss + self.critic_loss + self.term_loss

          local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
          gradients = tf.gradients(self.loss, local_vars)
          self.var_norms = tf.global_norm(local_vars)
          grads, self.grad_norms = tf.clip_by_global_norm(gradients, self._config.gradient_clip_value)

          # for grad, weight in zip(grads, local_vars):
          #   self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
          #   self.summaries.append(tf.summary.histogram(weight.name, weight))

          self.merged_summary = tf.summary.merge([tf.summary.scalar('avg_critic_loss', self.critic_loss),
                                                  tf.summary.scalar('avg_termination_loss', self.term_loss),
                                                  tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                                                  tf.summary.scalar('avg_policy_loss', self.policy_loss),
                                                  tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
                                                  tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads)),
                                                  gradient_summaries(zip(grads, local_vars))] + self.summaries)

          global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
          self.apply_grads = self._network_optimizer.apply_gradients(zip(grads, global_vars))

  def get_intra_option_policy(self, options):
    current_option_option_one_hot = tf.one_hot(options, self._nb_options, dtype=tf.float32, name="options_one_hot")
    current_option_option_one_hot = tf.tile(current_option_option_one_hot[..., None], [1, 1, self._action_size])
    action_probabilities = tf.reduce_sum(tf.multiply(self.options, current_option_option_one_hot),
                                         reduction_indices=1, name="P_a")
    return action_probabilities

  def get_responsible_outputs(self, policy, actions):
    actions_onehot = tf.one_hot(actions, self._action_size, dtype=tf.float32,
                                name="actions_one_hot")
    responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
    return responsible_outputs

  def get_q(self, o):
    current_option_option_one_hot = tf.one_hot(o, self._config.nb_options, name="options_one_hot")
    q_values = tf.reduce_sum(tf.multiply(self.q_val, current_option_option_one_hot),
                             reduction_indices=1, name="Values_Q")
    return q_values

  def get_o_term(self, o, boolean_value=False):
    current_option_option_one_hot = tf.one_hot(o, self._config.nb_options, name="options_one_hot")
    o_terminations = tf.reduce_sum(tf.multiply(self.termination, current_option_option_one_hot),
                                   reduction_indices=1, name="O_Terminations")
    if boolean_value:
      local_random = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32, name="rand_o_term")
      o_terminations = o_terminations > local_random
    return o_terminations


class SFNetwork(tf.contrib.rnn.RNNCell):
  def __init__(self, scope, config, action_size):
    self._scope = scope
    self._conv_layers = config.conv_layers
    self._fc_layers = config.fc_layers
    self._sf_layers = config.sf_layers
    self._deconv_layers = config.deconv_layers
    self._action_size = action_size
    self._nb_options = config.nb_options
    self._nb_envs = config.num_agents
    self._config = config

    self._network_optimizer = config.network_optimizer(
      self._config.lr, name='network_optimizer')
    self._exploration_options = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                                 self._config.initial_random_action_prob)

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, config.input_size[0], config.input_size[1], config.history_size],
                                        dtype=tf.float32, name="Inputs")
      self.total_steps = tf.placeholder(shape=[], dtype=tf.int32, name="total_steps")

      if self._config.history_size == 3:
        self.image_summaries = tf.summary.image('input', self.observation * 255, max_outputs=30)
      else:
        self.image_summaries = tf.summary.image('input', self.observation[:, :, :, 0:1] * 255, max_outputs=30)
      self.summaries = []
      with tf.variable_scope('conv'):
        for i, (kernel_size, stride, nb_kernels) in enumerate(self._conv_layers):
          out = layers.conv2d(self.observation, num_outputs=nb_kernels, kernel_size=kernel_size,
                              stride=stride, activation_fn=tf.nn.relu,
                              variables_collections=tf.get_collection("variables"),
                              outputs_collections="activations", scope="conv_{}".format(i))
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

        out = layers.flatten(out, scope="flatten")
        with tf.variable_scope("fc"):
          for i, nb_filt in enumerate(self._fc_layers):
            out = layers.fully_connected(out, num_outputs=nb_filt,
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="fc_{}".format(i))
            out = layer_norm_fn(out, relu=True)
            self.summaries.append(tf.contrib.layers.summarize_activation(out))

        with tf.variable_scope("option_term"):
          self.termination = layers.fully_connected(out, num_outputs=self._nb_options,
                                                    activation_fn=tf.nn.sigmoid,
                                                    variables_collections=tf.get_collection("variables"),
                                                    outputs_collections="activations", scope="fc_option_term")
          self.summaries.append(tf.contrib.layers.summarize_activation(self.termination))

        with tf.variable_scope("sf"):
          self.sf = tf.tile(out[..., None], [1, 1, self._nb_options], name="sf_tile")
          self.sf = tf.split(self.sf, num_or_size_splits=self._nb_options, axis=2, name="sf_split")
          self.sf = [tf.squeeze(sf, 2) for sf in self.sf]
          for j in range(self._nb_options):
            for i, nb_filt in enumerate(self._sf_layers):
              self.sf[j] = layers.fully_connected(self.sf[j], num_outputs=nb_filt,
                                                  activation_fn=None,
                                                  variables_collections=tf.get_collection("variables"),
                                                  outputs_collections="activations", scope="sf_{}_fc_{}".format(j, i))
              self.sf[j] = layer_norm_fn(self.sf[j], relu=True)
              self.summaries.append(tf.contrib.layers.summarize_activation(self.sf[j]))
              # self.sf = tf.stack(self.sf, 2)

        with tf.variable_scope("instant_r"):
          self.instant_r = layers.fully_connected(out, num_outputs=1,
                                                  activation_fn=None,
                                                  variables_collections=tf.get_collection("variables"),
                                                  outputs_collections="activations", scope="instant_r_w")
          self.summaries.append(tf.contrib.layers.summarize_activation(self.instant_r))

        with tf.variable_scope("autoencoder"):
          decoder_out = tf.expand_dims(tf.expand_dims(out, 1), 1)
          for i, (kernel_size, stride, padding, nb_kernels) in enumerate(self._deconv_layers):
            decoder_out = layers.conv2d_transpose(decoder_out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                                  stride=stride, activation_fn=tf.nn.relu,
                                                  padding="same" if padding > 0 else "valid",
                                                  variables_collections=tf.get_collection("variables"),
                                                  outputs_collections="activations", scope="deconv_{}".format(i))
            self.summaries.append(tf.contrib.layers.summarize_activation(decoder_out))

        with tf.variable_scope("q_val"):
          w_r = [v for v in tf.trainable_variables() if "instant_r/instant_r_w/weights:0" in v.name][0]
          b_r = [v for v in tf.trainable_variables() if "instant_r/instant_r_w/biases:0" in v.name][0]
          self.q_val = [tf.matmul(sf, w_r) + b_r for sf in self.sf]
          self.q_val = tf.squeeze(tf.stack(self.q_val, 2), 1)
          self.summaries.append(tf.contrib.layers.summarize_activation(self.q_val))

        self.sf = tf.stack(self.sf, 2)

        max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
        exp_options = tf.random_uniform(shape=[1], minval=0, maxval=self._config.nb_options,
                                        dtype=tf.int32)
        local_random = tf.random_uniform(shape=[1], minval=0., maxval=1., dtype=tf.float32,
                                         name="rand_options")
        # probability_of_random_option = self._exploration_options.value(self.total_steps)
        probability_of_random_option = self._config.final_random_action_prob
        # condition = local_random > tf.tile(probability_of_random_option[None, ...], [1])
        condition = local_random > probability_of_random_option

        self.current_option = tf.where(condition, max_options, exp_options)
        self.v = tf.reduce_max(self.q_val, axis=1) * (1 - probability_of_random_option) + \
                 probability_of_random_option * tf.reduce_mean(self.q_val, axis=1)
        self.exp_sf = tf.reduce_max(self.sf, axis=2) * (2 - probability_of_random_option) + \
                      probability_of_random_option * tf.reduce_mean(self.sf, axis=2)

        with tf.variable_scope("i_o_policies"):
          self.options = []
          for i in range(self._nb_options):
            option = layers.fully_connected(out, num_outputs=self._action_size,
                                            activation_fn=tf.nn.softmax,
                                            variables_collections=tf.get_collection("variables"),
                                            outputs_collections="activations", scope="option_{}".format(i))

            self.summaries.append(tf.contrib.layers.summarize_activation(option))
            self.options.append(option)
          self.options = tf.stack(self.options, 1)

        if scope != 'global':
          self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
          self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Options")
          self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
          self.target_sf = tf.placeholder(shape=[None, self._sf_layers[-1]], dtype=tf.float32)
          self.target_r = tf.placeholder(shape=[None], dtype=tf.float32)
          # self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
          self.delib = tf.placeholder(shape=[None], dtype=tf.float32)

          self.policy = self.get_intra_option_policy(self.options_placeholder)
          self.responsible_outputs = self.get_responsible_outputs(self.policy, self.actions_placeholder)
          sf_o = self.get_sf(self.options_placeholder)
          q_val = self.get_q(self.options_placeholder)
          termination = self.get_o_term(self.options_placeholder)

          with tf.name_scope('sf_loss'):
            sf_td_error = self.target_sf - sf_o
            self.sf_loss = tf.reduce_mean(self._config.sf_coef * tf.square(sf_td_error))
          with tf.name_scope('instant_r_loss'):
            instant_r_error = self.target_r - self.instant_r
            self.instant_r_loss = tf.reduce_mean(self._config.instant_r_coef * tf.square(instant_r_error))
          with tf.name_scope('autoencoder_loss'):
            auto_error = decoder_out - self.observation
            self.auto_loss = tf.reduce_mean(self._config.auto_coef * tf.square(auto_error))

          with tf.name_scope('termination_loss'):
            self.term_loss = tf.reduce_mean(
              termination * (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + self.delib))
          with tf.name_scope('entropy_loss'):
            self.entropy_loss = -self._config.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.policy *
                                                                                          tf.log(self.policy +
                                                                                                 1e-7), axis=1))
          with tf.name_scope('policy_loss'):
            td_error = self.target_return - tf.stop_gradient(q_val)
            self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_outputs + 1e-7) * td_error)

          self.loss = self.policy_loss - self.entropy_loss + self.sf_loss + self.auto_loss + self.instant_r_loss + self.term_loss

          local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
          gradients = tf.gradients(self.loss, local_vars)
          self.var_norms = tf.global_norm(local_vars)
          grads, self.grad_norms = tf.clip_by_global_norm(gradients, self._config.gradient_clip_value)

          # for grad, weight in zip(grads, local_vars):
          #   self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
          #   self.summaries.append(tf.summary.histogram(weight.name, weight))

          self.merged_summary = tf.summary.merge([tf.summary.scalar('avg_sf_loss', self.sf_loss),
                                                  tf.summary.scalar('avg_instant_r_loss', self.instant_r_loss),
                                                  tf.summary.scalar('avg_auto_loss', self.auto_loss),
                                                  tf.summary.scalar('avg_termination_loss', self.term_loss),
                                                  tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                                                  tf.summary.scalar('avg_policy_loss', self.policy_loss),
                                                  tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
                                                  tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads)),
                                                  gradient_summaries(zip(grads, local_vars))] + self.summaries)

          global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
          self.apply_grads = self._network_optimizer.apply_gradients(zip(grads, global_vars))

  def get_intra_option_policy(self, options):
    current_option_option_one_hot = tf.one_hot(options, self._nb_options, dtype=tf.float32, name="options_one_hot")
    current_option_option_one_hot = tf.tile(current_option_option_one_hot[..., None], [1, 1, self._action_size])
    action_probabilities = tf.reduce_sum(tf.multiply(self.options, current_option_option_one_hot),
                                         reduction_indices=1, name="P_a")
    return action_probabilities

  def get_responsible_outputs(self, policy, actions):
    actions_onehot = tf.one_hot(actions, self._action_size, dtype=tf.float32,
                                name="actions_one_hot")
    responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
    return responsible_outputs

  def get_sf(self, o):
    current_option_option_one_hot = tf.expand_dims(tf.one_hot(o, self._config.nb_options, name="options_one_hot"), 1)
    current_option_option_one_hot = tf.tile(current_option_option_one_hot, [1, self._sf_layers[-1], 1])
    sf_values = tf.reduce_sum(tf.multiply(self.sf, current_option_option_one_hot),
                              reduction_indices=2, name="Values_SF")
    return sf_values

  def get_q(self, o):
    current_option_option_one_hot = tf.one_hot(o, self._config.nb_options, name="options_one_hot")
    q_values = tf.reduce_sum(tf.multiply(self.q_val, current_option_option_one_hot),
                             reduction_indices=1, name="Values_Q")
    return q_values

  def get_o_term(self, o, boolean_value=False):
    current_option_option_one_hot = tf.one_hot(o, self._config.nb_options, name="options_one_hot")
    o_terminations = tf.reduce_sum(tf.multiply(self.termination, current_option_option_one_hot),
                                   reduction_indices=1, name="O_Terminations")
    if boolean_value:
      local_random = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32, name="rand_o_term")
      o_terminations = o_terminations > local_random
    return o_terminations


class ACNetwork():
  def __init__(self, scope, config, action_size, stage=1):
    self._scope = scope
    self._conv_layers = config.conv_layers
    self._fc_layers = config.fc_layers
    self._action_size = action_size
    self._nb_options = config.nb_options
    self._nb_envs = config.num_agents
    self._config = config
    self.option = 0
    self._sf_layers = config.sf_layers
    self._deconv_layers = config.deconv_layers
    if stage == 1:
      self._network_optimizer = config.network_optimizer(
        self._config.lr, name='network_optimizer')
    elif stage == 2:
      self._network_optimizer = config.network_optimizer(
        self._config.sf_lr, name='network_optimizer')
    else:
      self._network_optimizer = config.network_optimizer(
        self._config.lr, name='network_optimizer')
    self._exploration_policy = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                                self._config.initial_random_action_prob)

    self.trainable_core = True if stage == 1 else False
    self.trainable_sf = True if stage == 2 else False
    self.trainable_options = True if stage == 4 else False

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, config.input_size[0], config.input_size[1], config.history_size],
                                        dtype=tf.float32, name="Inputs")
      self.total_steps = tf.placeholder(shape=[], dtype=tf.int32, name="total_steps")

      if self._config.history_size == 3:
        self.image_summaries = tf.summary.image('input', self.observation * 255, max_outputs=30)
      else:
        self.image_summaries = tf.summary.image('input', self.observation[:, :, :, 0:1] * 255, max_outputs=30)
      self.summaries = []
      with tf.variable_scope('conv'):
        for i, (kernel_size, stride, nb_kernels) in enumerate(self._conv_layers):
          out = layers.conv2d(self.observation, num_outputs=nb_kernels, kernel_size=kernel_size,
                              stride=stride, activation_fn=tf.nn.relu, trainable=self.trainable_core,
                              variables_collections=tf.get_collection("variables"),
                              outputs_collections="activations", scope="conv_{}".format(i))
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      out = layers.flatten(out, scope="flatten")

      with tf.variable_scope("fc"):
        for i, nb_filt in enumerate(self._fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None, trainable=self.trainable_core,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_{}".format(i))
          out = layer_norm_fn(out, relu=True)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      self.fi = tf.stop_gradient(out)

      with tf.variable_scope("sf"):
        self.sf = tf.tile(self.fi[..., None], [1, 1, self._action_size], name="sf_tile")
        self.sf = tf.split(self.sf, num_or_size_splits=self._action_size, axis=2, name="sf_split")
        self.sf = [tf.squeeze(sf, 2) for sf in self.sf]
        for j in range(self._action_size):
          for i, nb_filt in enumerate(self._sf_layers):
            self.sf[j] = layers.fully_connected(self.sf[j], num_outputs=nb_filt,
                                                activation_fn=None, trainable=self.trainable_sf,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="sf_{}_fc_{}".format(j, i))
            if i < len(self._sf_layers) - 1:
              self.sf[j] = layer_norm_fn(self.sf[j], relu=True)
          self.summaries.append(tf.contrib.layers.summarize_activation(self.sf[j]))
        self.sf = tf.stack(self.sf, 2)
        # for j in range(self._nb_options):
        # for i, nb_filt in enumerate(self._sf_layers):
        #   self.sf[j] = layers.fully_connected(self.sf[j], num_outputs=nb_filt,
        #                                       activation_fn=None,
        #                                       variables_collections=tf.get_collection("variables"),
        #                                       outputs_collections="activations", scope="sf_{}_fc_{}".format(j, i))
        #   self.sf[j] = layer_norm_fn(self.sf[j], relu=True)
        #   self.summaries.append(tf.contrib.layers.summarize_activation(self.sf[j]))

      # with tf.variable_scope("instant_r"):
      #   self.instant_r = layers.fully_connected(out, num_outputs=1,
      #                                           activation_fn=None, trainable=self.trainable_sf,
      #                                           variables_collections=tf.get_collection("sf_variables"),
      #                                           outputs_collections="activations", scope="instant_r_w")
      #   self.summaries.append(tf.contrib.layers.summarize_activation(self.instant_r))


      # with tf.variable_scope("autoencoder"):
      #   decoder_out = tf.expand_dims(tf.expand_dims(out, 1), 1)
      #   for i, (kernel_size, stride, padding, nb_kernels) in enumerate(self._deconv_layers):
      #     decoder_out = layers.conv2d_transpose(decoder_out, num_outputs=nb_kernels, kernel_size=kernel_size,
      #                                           stride=stride, activation_fn=tf.nn.relu,
      #                                           padding="same" if padding > 0 else "valid",
      #                                           variables_collections=tf.get_collection("variables"),
      #                                           outputs_collections="activations", scope="deconv_{}".format(i))
      #     self.summaries.append(tf.contrib.layers.summarize_activation(decoder_out))


      self.policy = layers.fully_connected(out, num_outputs=self._action_size,
                                           activation_fn=tf.nn.softmax, trainable=self.trainable_core,
                                           variables_collections=tf.get_collection("variables"),
                                           outputs_collections="activations", scope="Policy")

      self.summaries.append(tf.contrib.layers.summarize_activation(self.policy))

      self.value = layers.fully_connected(out, num_outputs=1,
                                          activation_fn=None, trainable=self.trainable_core,
                                          variables_collections=tf.get_collection("variables"),
                                          outputs_collections="activations", scope="Value")
      self.summaries.append(tf.contrib.layers.summarize_activation(self.value))

      with tf.variable_scope("options"):
        self.option_policy = []
        self.option_value = []
        for i in range(self._nb_options):
          option = layers.fully_connected(self.fi, num_outputs=self._action_size + 1,
                                          activation_fn=tf.nn.softmax, trainable=self.trainable_options,
                                          variables_collections=tf.get_collection("variables"),
                                          outputs_collections="activations", scope="option_{}".format(i))
          option_value = layers.fully_connected(self.fi, num_outputs=1,
                                                activation_fn=None, trainable=self.trainable_options,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="option_value_{}".format(i))
          self.summaries.append(tf.contrib.layers.summarize_activation(option))
          self.summaries.append(tf.contrib.layers.summarize_activation(option_value))
          self.option_policy.append(option)
          self.option_value.append(option_value)
        self.option_policy = tf.stack(self.option_policy, 1)
        self.option_value = tf.stack(self.option_value, 1)

      if scope != 'global':
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
        self.actions_onehot = tf.one_hot(self.actions_placeholder, self._action_size, dtype=tf.float32,
                                         name="Actions_Onehot")
        self.option_actions_onehot = tf.one_hot(self.actions_placeholder, self._action_size + 1, dtype=tf.float32,
                                                name="Actions_Onehot")
        self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_sf = tf.placeholder(shape=[None, self._sf_layers[-1]], dtype=tf.float32)
        # self.target_r = tf.placeholder(shape=[None], dtype=tf.float32)

        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
        self.responsible_outputs_options = tf.reduce_sum(
          self.option_policy[:, self.option, :] * self.option_actions_onehot, [1])
        self.value_options = self.option_value[:, self.option, :]
        sf_a = self.get_sf(self.actions_placeholder)

        # Loss functions
        with tf.name_scope('critic_loss'):
          td_error = self.target_return - self.value
          self.critic_loss = tf.reduce_mean(self._config.critic_coef * tf.square(td_error))

        with tf.name_scope('option_critic_loss'):
          option_td_error = self.target_return - self.value_options
          self.option_critic_loss = tf.reduce_mean(self._config.critic_coef * tf.square(option_td_error))

        with tf.name_scope('entropy_loss'):
          self.entropy_loss = -self._config.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.policy *
                                                                                        tf.log(self.policy +
                                                                                               1e-7), axis=1))
        with tf.name_scope('policy_loss'):
          self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_outputs + 1e-7) * tf.stop_gradient(td_error))

        with tf.name_scope('option_policy_loss'):
          self.option_policy_loss = -tf.reduce_mean(
            tf.log(self.responsible_outputs_options + 1e-7) * tf.stop_gradient(option_td_error))

        with tf.name_scope('option_entropy_loss'):
          self.option_entropy_loss = -self._config.option_entropy_coef * tf.reduce_mean(
            tf.reduce_sum(self.option_policy[:, self.option, :] *
                          tf.log(self.option_policy[:, self.option, :] +
                                 1e-7), axis=1))

        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - sf_a
          self.sf_loss = tf.reduce_mean(self._config.sf_coef * tf.square(sf_td_error))

        # with tf.name_scope('instant_r_loss'):
        #   instant_r_error = self.target_r - self.instant_r
        #   self.instant_r_loss = tf.reduce_mean(self._config.instant_r_coef * tf.square(instant_r_error))

        if self.trainable_core:
          self.loss = self.policy_loss - self.entropy_loss + self.critic_loss
          loss_summaries = [tf.summary.scalar('avg_critic_loss', self.critic_loss),
                            tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                            tf.summary.scalar('avg_policy_loss', self.policy_loss)]

        elif self.trainable_sf:
          self.loss = self.sf_loss  # + self.instant_r_loss
          loss_summaries = [tf.summary.scalar('avg_sf_loss', self.sf_loss)]
        elif self.trainable_options:
          self.loss = self.option_policy_loss + self.option_critic_loss + self.option_entropy_loss
          loss_summaries = [tf.summary.scalar('option_critic_loss', self.option_critic_loss),
                            tf.summary.scalar('option_policy_loss', self.option_policy_loss),
                            tf.summary.scalar('option_entropy_loss', self.option_entropy_loss)]

        if hasattr(self, "loss"):
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
            gradient_summaries(zip(grads, local_vars))] + self.summaries)
          global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
          self.apply_grads = self._network_optimizer.apply_gradients(zip(grads, global_vars))

  def get_sf(self, a):
    current_action_one_hot = tf.expand_dims(tf.one_hot(a, self._action_size, name="actions_one_hot"), 1)
    current_action_one_hot = tf.tile(current_action_one_hot, [1, self._sf_layers[-1], 1])
    sf_values = tf.reduce_sum(tf.multiply(self.sf, current_action_one_hot),
                              reduction_indices=2, name="Values_SF")
    return sf_values

  def get_responsible_outputs(self, policy, actions):
    actions_onehot = tf.one_hot(actions, self._action_size, dtype=tf.float32,
                                name="actions_one_hot")
    responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
    return responsible_outputs


class LinearSFNetwork():
  def __init__(self, scope, config, action_size, nb_states):
    self._scope = scope
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
      self.observation = tf.placeholder(shape=[None, nb_states],
                                        dtype=tf.float32, name="Inputs")
      self.sf = layers.fully_connected(self.observation, num_outputs=nb_states,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf")
      if scope != 'global':
        # self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
        # self.actions_onehot = tf.one_hot(self.actions_placeholder, self._action_size, dtype=tf.float32,
        #                                  name="Actions_Onehot")
        self.target_sf = tf.placeholder(shape=[None, nb_states], dtype=tf.float32, name="target_SF")

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


class DIFNetwork():
  def __init__(self, scope, config, action_size, nb_states):
    self._scope = scope
    # self.option = 0
    self._conv_layers = config.conv_layers
    self._fc_layers = config.fc_layers
    self._sf_layers = config.sf_layers
    self._aux_fc_layers = config.aux_fc_layers
    self._aux_deconv_layers = config.aux_deconv_layers
    self._action_size = action_size
    self._nb_options = config.nb_options
    self._nb_envs = config.num_agents
    self._config = config

    self._network_optimizer = config.network_optimizer(
      self._config.lr, name='network_optimizer')

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, config.input_size[0], config.input_size[1], config.history_size],
                                        dtype=tf.float32, name="Inputs")
      self.image_summaries = []
      if self._config.history_size == 3:
        self.image_summaries.append(tf.summary.image('input', self.observation * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('input', self.observation[:, :, :, 0:1] * 255, max_outputs=30))
      self.summaries = []

      with tf.variable_scope('conv'):
        for i, (kernel_size, stride, nb_kernels) in enumerate(self._conv_layers):
          out = layers.conv2d(self.observation, num_outputs=nb_kernels, kernel_size=kernel_size,
                              stride=stride, activation_fn=None,
                              variables_collections=tf.get_collection("variables"),
                              outputs_collections="activations", scope="conv_{}".format(i))
          # out = layer_norm_fn(out, relu=True)
          out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))
        out = layers.flatten(out, scope="flatten")

      with tf.variable_scope("fc"):
        for i, nb_filt in enumerate(self._fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_{}".format(i))

          if i < len(self._fc_layers) - 1:
            # out = layer_norm_fn(out, relu=False)
            # out = layer_norm_fn(out, relu=True)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))
      self.fi = out

      # out = tf.stop_gradient(layer_norm_fn(self.fi, relu=True))
      out = tf.stop_gradient(tf.nn.relu(self.fi))
      with tf.variable_scope("sf"):
        for i, nb_filt in enumerate(self._sf_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf_{}".format(i))
          if i < len(self._sf_layers) - 1:
            # out = layer_norm_fn(out, relu=False)
            out = tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      self.sf = out

      out = self.fi
      with tf.variable_scope("action_fc"):
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
        actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self._fc_layers[-1],
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="action_fc{}".format(i))
        # out = layer_norm_fn(out, relu=False)
      out = tf.add(out, actions)
      # out = layer_norm_fn(out, relu=True)
      out = tf.nn.relu(out)

      with tf.variable_scope("aux_fc"):
        for i, nb_filt in enumerate(self._aux_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="aux_fc_{}".format(i))
          # out = layer_norm_fn(out, relu=False)
          if i < len(self._aux_fc_layers) - 1:
            # out = layer_norm_fn(out, relu=True)
            out= tf.nn.relu(out)
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      with tf.variable_scope("aux_deconv"):
        decoder_out = tf.expand_dims(tf.expand_dims(out, 1), 1)
        for i, (kernel_size, stride, padding, nb_kernels) in enumerate(self._aux_deconv_layers):
          decoder_out = layers.conv2d_transpose(decoder_out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                                stride=stride, activation_fn=None,
                                                padding="same" if padding > 0 else "valid",
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="aux_deconv_{}".format(i))
          if i < len(self._aux_deconv_layers) - 1:
            # decoder_out = layer_norm_fn(decoder_out, relu=False)
            decoder_out = tf.nn.relu(decoder_out)
          self.summaries.append(tf.contrib.layers.summarize_activation(decoder_out))

      # with tf.variable_scope("aux_deconv"):
      #   decoder_out = tf.expand_dims(tf.expand_dims(out, 1), 1)
      #   for i, (kernel_size, stride, padding, nb_kernels) in enumerate(self._aux_deconv_layers):
      #     decoder_out = layers.conv2d(decoder_out, num_outputs=nb_kernels, kernel_size=kernel_size,
      #                                           stride=stride, activation_fn=None,
      #                                           padding="same" if padding > 0 else "valid",
      #                                           variables_collections=tf.get_collection("variables"),
      #                                           outputs_collections="activations", scope="aux_deconv_{}".format(i))
      #     if i < len(self._aux_deconv_layers) - 1:
      #       # decoder_out = layer_norm_fn(decoder_out, relu=False)
      #       decoder_out = tf.nn.relu(decoder_out)
      #     self.summaries.append(tf.contrib.layers.summarize_activation(decoder_out))
      #
      # decoder_out = tf.depth_to_space(decoder_out, 13, "depth_to_space")
      self.next_obs = decoder_out

      if self._config.history_size == 3:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs * 255, max_outputs=30))
      else:
        self.image_summaries.append(tf.summary.image('next_obs', self.next_obs[:, :, :, 0:1] * 255, max_outputs=30))

      if scope != 'global':
        self.target_sf = tf.placeholder(shape=[None, self._sf_layers[-1]], dtype=tf.float32, name="target_SF")
        self.target_next_obs = tf.placeholder(
          shape=[None, config.input_size[0], config.input_size[1], config.history_size], dtype=tf.float32,
          name="target_next_obs")
        if self._config.history_size == 3:
          self.image_summaries.append(tf.summary.image('target_next_obs', self.target_next_obs * 255, max_outputs=30))
        else:
          self.image_summaries.append(tf.summary.image('target_next_obs', self.target_next_obs[:, :, :, 0:1] * 255, max_outputs=30))
        self.matrix_sf = tf.placeholder(shape=[self._config.sf_transition_matrix_size, self._sf_layers[-1]],
                                     dtype=tf.float32, name="matrix_sf")
        self.s, self.u, self.v = tf.svd(self.matrix_sf)

        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - self.sf
          self.sf_loss = tf.reduce_mean(tf.square(sf_td_error))

        with tf.name_scope('aux_loss'):
          aux_error = self.next_obs - self.target_next_obs
          self.aux_loss = tf.reduce_mean(self._config.aux_coef * tf.square(aux_error))

        self.loss = self.sf_loss + self.aux_loss
        loss_summaries = [tf.summary.scalar('avg_sf_loss', self.sf_loss),
                          tf.summary.scalar('aux_loss', self.aux_loss),
                          tf.summary.scalar('total_loss', self.loss)]

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads, self.grad_norms = tf.clip_by_global_norm(gradients, self._config.gradient_clip_value)

        self.merged_summary = tf.summary.merge(self.image_summaries + self.summaries + loss_summaries + [
          tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
          tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads)),
          gradient_summaries(zip(grads, local_vars))])
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = self._network_optimizer.apply_gradients(zip(grads, global_vars))

class DIFNetwork_FC():
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

      self.summaries_sf = []
      self.summaries_aux = []

      out = self.observation
      out = layers.flatten(out, scope="flatten")

      with tf.variable_scope("fc"):
        for i, nb_filt in enumerate(self.config.fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_{}".format(i))

          if i < len(self.config.fc_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
          self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.fi = out

      with tf.variable_scope("sf"):
        out = tf.stop_gradient(tf.nn.relu(self.fi))
        for i, nb_filt in enumerate(self.config.sf_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf_{}".format(i))
          if i < len(self.config.sf_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
        self.sf = out


      with tf.variable_scope("action_fc"):
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
        actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.fc_layers[-1],
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="action_fc{}".format(i))
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(actions))
        out = tf.add(self.fi, actions)
        out = tf.nn.relu(out)

      with tf.variable_scope("aux_fc"):
        for i, nb_filt in enumerate(self.config.aux_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="aux_fc_{}".format(i))
          if i < len(self.config.aux_fc_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.next_obs = tf.reshape(out, (-1, config.input_size[0], config.input_size[1], config.history_size))

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

        # regularizer_features = tf.reduce_mean(self.config.feat_decay * tf.nn.l2_loss(self.fi))
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        
        # regularizer_sf_weights = tf.add_n(
        #   [self.config.sf_weight_decay * tf.nn.l2_loss(w) for w in local_vars if 'sf' in w.name])
        # self.loss = self.sf_loss + self.aux_loss + regularizer_features + regularizer_sf_weights
        # loss_summaries = [tf.summary.scalar('avg_sf_loss', self.sf_loss),
        #                   tf.summary.scalar('aux_loss', self.aux_loss),
        #                   tf.summary.scalar('total_loss', self.loss)]

        gradients_sf = tf.gradients(self.sf_loss, local_vars)
        gradients_aux = tf.gradients(self.aux_loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        grads_sf, self.grad_norms = tf.clip_by_global_norm(gradients_sf, self.config.gradient_clip_value)
        grads_aux, self.grad_norms = tf.clip_by_global_norm(gradients_aux, self.config.gradient_clip_value)

        self.merged_summary_sf = tf.summary.merge(self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss)] + [
          tf.summary.scalar('gradient_norm', tf.global_norm(gradients_sf)),
          tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads_sf)),
          gradient_summaries(zip(grads_sf, local_vars))])
        self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                                   [tf.summary.scalar('aux_loss', self.aux_loss)] + [
          tf.summary.scalar('gradient_norm', tf.global_norm(gradients_aux)),
          tf.summary.scalar('cliped_gradient_norm', tf.global_norm(grads_aux)),
          gradient_summaries(zip(grads_aux, local_vars))])
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads_sf = self.network_optimizer.apply_gradients(zip(grads_sf, global_vars))
        self.apply_grads_aux = self.network_optimizer.apply_gradients(zip(grads_aux, global_vars))

def layer_norm_fn(x, relu=True):
  x = layers.layer_norm(x, scale=True, center=True)
  if relu:
    x = tf.nn.relu(x)
  return x
