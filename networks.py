from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers


class AOCNetwork(tf.contrib.rnn.RNNCell):

  def __init__(self, scope, conv_layers, fc_layers, action_size, nb_options, nb_envs):
    self._scope = scope
    self._conv_layers = conv_layers
    self._fc_layers = fc_layers
    self._action_size = action_size
    self._nb_options = nb_options
    self._nb_envs = nb_envs

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


def layer_norm_fn(x, relu=True):
  x = layers.layer_norm(x, scale=True, center=True)
  if relu:
    x = tf.nn.relu(x)
  return x