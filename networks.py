from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers


class AOCNetwork(tf.contrib.rnn.RNNCell):

  def __init__(self, scope, conv_layers, fc_layers, action_size, nb_options, nb_envs, config):
    self._scope = scope
    self._conv_layers = conv_layers
    self._fc_layers = fc_layers
    self._action_size = action_size
    self._nb_options = nb_options
    self._nb_envs = nb_envs
    self._config = config

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, 84, 84, 4],
                                   dtype=tf.float32, name="Inputs")
      with tf.variable_scope('conv'):
        for i, (kernel_size, stride, nb_kernels) in enumerate(self._conv_layers):
          out = layers.conv2d(self.observation, num_outputs=nb_kernels, kernel_size=kernel_size,
                              stride=stride, activation_fn=tf.nn.relu,
                              variables_collections=tf.get_collection("variables"),
                              outputs_collections="activations", scope="conv_{}".format(i))
        out = layers.flatten(out, scope="flatten")
        with tf.variable_scope("fc"):
          for i, nb_filt in enumerate(self._fc_layers):
            out = layers.fully_connected(out, num_outputs=nb_filt,
                                             activation_fn=None,
                                             variables_collections=tf.get_collection("variables"),
                                             outputs_collections="activations", scope="fc_{}".format(i))
            out = layer_norm_fn(out, relu=True)
        with tf.variable_scope("option_term"):
          self.termination = layers.fully_connected(out, num_outputs=self._nb_options,
                                                                    activation_fn=tf.nn.sigmoid,
                                                                    variables_collections=tf.get_collection("variables"),
                                                                    outputs_collections="activations")
        with tf.variable_scope("q_val"):
          self.q_val = layers.fully_connected(out, num_outputs=self._nb_options,
                                                      activation_fn=None,
                                                      variables_collections=tf.get_collection("variables"),
                                                      outputs_collections="activations")
        with tf.variable_scope("i_o_policies"):
          self.options = []
          for _ in range(self._nb_options):
            option = layers.fully_connected(out, num_outputs=self._action_size,
                                                activation_fn=tf.nn.softmax,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations")
            self.options.append(tf.expand_dims(option, 1))
          self.options = tf.concat(self.options, 1)

        if scope != 'global':
          self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
          self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Options")
          self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
          policy = self.get_intra_option_policy(self.options_placeholder)
          responsible_outputs = self.get_responsible_outputs(policy, self.actions_placeholder)
          self.v =
          self.delib =
          # # Loss functions
          # self.value_loss = FLAGS.beta_v * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
          # self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))

          with tf.name_scope('critic_loss'):
            td_error = tf.stop_gradient(self.target_v) - self.q_val
            self.critic_loss = tf.reduce_mean(self._config.critic_coef * 0.5 * tf.square((td_error)))
          with tf.name_scope('termination_loss'):
            self.term_loss = tf.reduce_mean(self.termination * (tf.stop_gradient(self.q_val) - tf.stop_gradient(self.v) +
                                                               tf.tile(self.delib[:, None],
                                                                       [1, self._config.max_length])))
          with tf.name_scope('entropy_loss'):
            self.entropy_loss = self._config.entropy_coef * tf.reduce_mean(tf.reduce_sum(policy *
                                                                                           tf.log(policy +
                                                                                                  1e-7), axis=1))
          with tf.name_scope('policy_loss'):
            self.policy_loss = -tf.reduce_sum(tf.log(responsible_outputs + 1e-7) * td_error)

          self.loss = self.policy_loss + self.entropy_loss + self.critic_loss + self.term_loss

          local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
          self.gradients = tf.gradients(self.loss, local_vars)
          self.var_norms = tf.global_norm(local_vars)
          grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, FLAGS.gradient_clip_value)

          self.worker_summaries = [summary_conv_act, summary_hidden_act, summary_rnn_act, summary_policy_act,
                                   summary_value_act]
          for grad, weight in zip(grads, local_vars):
            self.worker_summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
            self.worker_summaries.append(tf.summary.histogram(weight.name, weight))

          self.merged_summary = tf.summary.merge(self.worker_summaries)

          global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
          self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

  def get_intra_option_policy(self, options):
    current_option_option_one_hot = tf.one_hot(options, self._nb_options, dtype=tf.float32, name="options_one_hot")
    current_option_option_one_hot = tf.tile(current_option_option_one_hot[..., None], [1, 1, 1, self.action_size])
    action_probabilities = tf.reduce_sum(tf.multiply(self.local_network.options, current_option_option_one_hot),
                                         reduction_indices=2, name="P_a")
    return action_probabilities

  def get_responsible_outputs(self, policy, actions):
    actions_onehot = tf.one_hot(actions, self._action_size, dtype=tf.float32,
                                     name="actions_one_hot")
    responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
    return responsible_outputs

def layer_norm_fn(x, relu=True):
  x = layers.layer_norm(x, scale=True, center=True)
  if relu:
    x = tf.nn.relu(x)
  return x