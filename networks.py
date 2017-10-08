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
      self.observation = tf.placeholder(shape=[None, 84, 84, 4],
                                   dtype=tf.float32, name="Inputs")
      self.total_steps = tf.placeholder(shape=[], dtype=tf.int32, name="total_steps")

      self.image_summaries = tf.summary.image('input', self.observation, max_outputs=1)
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
                                                                    outputs_collections="activations")
          self.summaries.append(tf.contrib.layers.summarize_activation(self.termination))

        with tf.variable_scope("q_val"):
          self.q_val = layers.fully_connected(out, num_outputs=self._nb_options,
                                                      activation_fn=None,
                                                      variables_collections=tf.get_collection("variables"),
                                                      outputs_collections="activations")
          self.summaries.append(tf.contrib.layers.summarize_activation(self.q_val))

          max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
          exp_options = tf.random_uniform(shape=[1], minval=0, maxval=self._config.nb_options,
                                          dtype=tf.int32)
          local_random = tf.random_uniform(shape=[1], minval=0., maxval=1., dtype=tf.float32,
                                           name="rand_options")
          probability_of_random_option = self._exploration_options.value(self.total_steps)
          condition = local_random > tf.tile(probability_of_random_option[None, ...], [1])
          self.current_option = tf.where(condition, max_options, exp_options)
          self.v = tf.reduce_max(self.q_val, axis=1) * (1 - probability_of_random_option) + \
              probability_of_random_option * tf.reduce_mean(self.q_val, axis=1)

        with tf.variable_scope("i_o_policies"):
          self.options = []
          for _ in range(self._nb_options):
            option = layers.fully_connected(out, num_outputs=self._action_size,
                                                activation_fn=tf.nn.softmax,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations")

            self.summaries.append(tf.contrib.layers.summarize_activation(option))
            self.options.append(option)
          self.options = tf.stack(self.options, 1)

        if scope != 'global':
          self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
          self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Options")
          self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
          self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
          self.delib = tf.placeholder(shape=[], dtype=tf.float32)

          policy = self.get_intra_option_policy(self.options_placeholder)
          responsible_outputs = self.get_responsible_outputs(policy, self.actions_placeholder)
          q_val = self.get_q(self.options_placeholder)
          termination = self.get_o_term(self.options_placeholder)

          with tf.name_scope('critic_loss'):
            td_error = tf.stop_gradient(self.target_return) - q_val
            self.critic_loss = tf.reduce_mean(self._config.critic_coef * 0.5 * tf.square((td_error)))
          with tf.name_scope('termination_loss'):
            self.term_loss = tf.reduce_mean(termination * (tf.stop_gradient(q_val) - self.target_v + self.delib))
          with tf.name_scope('entropy_loss'):
            self.entropy_loss = self._config.entropy_coef * tf.reduce_mean(tf.reduce_sum(policy *
                                                                                           tf.log(policy +
                                                                                    1e-7), axis=1))
          with tf.name_scope('policy_loss'):
            self.policy_loss = -tf.reduce_sum(tf.log(responsible_outputs + 1e-7) * td_error)

          self.loss = self.policy_loss + self.entropy_loss + self.critic_loss + self.term_loss

          local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
          gradients = tf.gradients(self.loss, local_vars)
          self.var_norms = tf.global_norm(local_vars)
          grads, self.grad_norms = tf.clip_by_global_norm(gradients, self._config.gradient_clip_value)

          # for grad, weight in zip(grads, local_vars):
          #   self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
          #   self.summaries.append(tf.summary.histogram(weight.name, weight))

          self.merged_summary = tf.summary.merge([tf.summary.scalar('avg_critic_loss', tf.reduce_mean(self.critic_loss)),
                                                  tf.summary.scalar('avg_termination_loss', tf.reduce_mean(self.term_loss)),
                                                  tf.summary.scalar('avg_entropy_loss', tf.reduce_mean(self.entropy_loss)),
                                                  tf.summary.scalar('avg_policy_loss', tf.reduce_mean(self.policy_loss)),
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

  def get_policy_over_options(self, batch_size, probability_of_random_option):
    max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
    exp_options = tf.random_uniform(shape=[batch_size], minval=0, maxval=self._config.nb_options,
                                    dtype=tf.int32)
    local_random = tf.random_uniform(shape=[batch_size], minval=0., maxval=1., dtype=tf.float32, name="rand_options")
    condition = local_random > tf.tile(probability_of_random_option[None, ...], [batch_size])
    options = tf.where(condition, max_options, exp_options)

    return options

  def get_action(self, o):
    current_option_option_one_hot = tf.one_hot(o, self._config.nb_options, name="options_one_hot")
    current_option_option_one_hot = current_option_option_one_hot[:, :, None]
    current_option_option_one_hot = tf.tile(current_option_option_one_hot, [1, 1, self._action_size])
    self.action_probabilities = tf.reduce_sum(tf.multiply(self.options, current_option_option_one_hot),
                                              reduction_indices=1, name="P_a")
    policy = tf.multinomial(tf.log(self.action_probabilities), 1)[:, 0]
    return policy

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


def layer_norm_fn(x, relu=True):
  x = layers.layer_norm(x, scale=True, center=True)
  if relu:
    x = tf.nn.relu(x)
  return x