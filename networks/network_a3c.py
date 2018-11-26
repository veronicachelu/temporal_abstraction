import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_eigenoc import EignOCNetwork
import os
from online_clustering import OnlineCluster

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

class A3CNetwork(EignOCNetwork):
  def __init__(self, scope, config, action_size):
    self.goal_embedding_size = config.sf_layers[-1]
    super(A3CNetwork, self).__init__(scope, config, action_size)

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
      self.observation = tf.placeholder(
        shape=[None, self.nb_states],
        dtype=tf.float32, name="observation_placeholder")
      self.observation_image = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], 1],
        dtype=tf.float32, name="observation_image_placeholder")

      self.image_summaries.append(
        tf.summary.image('observation', self.observation_image, max_outputs=30))

      with tf.variable_scope("option_policy"):
        input_features = layers.fully_connected(self.observation,
                                                    num_outputs=self.action_size * self.goal_embedding_size,
                                                    activation_fn=tf.nn.relu,
                                                    scope="input_features")
        v = layers.fully_connected(input_features,
                                                    num_outputs=1,
                                                    activation_fn=None,
                                                    scope="v")
        self.v = tf.squeeze(v, 1)
        policy = layers.fully_connected(input_features,
                                                    num_outputs=self.action_size,
                                                    activation_fn=None,
                                                    scope="v")
        self.policy = tf.nn.softmax(policy, name="policy")

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  def build_losses(self):
    """Get the probabilities for each action taken under the intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.policy, self.actions_placeholder)

    with tf.name_scope('critic_loss'):
      td_error = self.target_return - self.v
      self.critic_loss = tf.reduce_mean(0.5 * tf.square(td_error))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(self.policy * tf.log(self.policy + 1e-7))
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
                          td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.critic_loss

  def l2_normalize(self, x, axis):
      norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True))
      return tf.maximum(x, 1e-8) / tf.maximum(norm, 1e-8)

  def cosine_similarity(self, v1, v2, axis):
    v1_norm = self.l2_normalize(v1, axis)
    v2_norm = self.l2_normalize(v2, axis)
    sim = tf.matmul(
      v1_norm, v2_norm, transpose_b=True)

    return sim

  """Build gradients for the losses with respect to the network params.
      Build summaries and update ops"""
  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    """Gradients and update ops"""
    self.grads_option, self.apply_grads_option = self.take_gradient(self.option_loss)

    """Summaries"""
    self.merged_summary_option = tf.summary.merge(self.summaries_option + \
                                                  [tf.summary.scalar('Entropy_loss', self.entropy_loss),
                                                   tf.summary.scalar('Policy_loss', self.policy_loss),
                                                   tf.summary.scalar('Critic_loss', self.critic_loss), ])


