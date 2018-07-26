import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_eigenoc import EignOCNetwork
import os

"""Function approximation network for the option critic policies and value functions when options are given as embeddings corresponding to the spectral decomposition of the SR matrix"""
class EmbeddingNetwork(EignOCNetwork):
  def __init__(self, scope, config, action_size):
    super(EmbeddingNetwork, self).__init__(scope, config, action_size)

  """Build the encoder for the latent representation space"""
  def build_feature_net(self, out):
    super(EmbeddingNetwork, self).build_feature_net(out)
    self.option_direction_placeholder = tf.placeholder(shape=[None, self.config.sf_layers[-1]],
                                                       dtype=tf.float32,
                                                       name="option_direction")

    self.fi_option = tf.add(tf.stop_gradient(self.fi), self.option_direction_placeholder)

  """Build the intra-option policies critics"""
  def build_eigen_option_q_val_net(self):
    with tf.variable_scope("eigen_option_q_val"):
      out = tf.stop_gradient(self.fi_option)
      self.eigen_q_val = layers.fully_connected(out, num_outputs=1,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="eigen_q_val")
      self.eigen_q_val = tf.squeeze(self.eigen_q_val, 1)
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.eigen_q_val))

  """Build the intra-option policies"""
  def build_intraoption_policies_nets(self):
    with tf.variable_scope("option_pi"):
      out = tf.stop_gradient(self.fi_option)
      self.option = layers.fully_connected(out,
                                           num_outputs=self.action_size,
                                           activation_fn=tf.nn.softmax,
                                           biases_initializer=None,
                                           variables_collections=tf.get_collection("variables"),
                                           outputs_collections="activations", scope="policy")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.option))

  """Build the option termination stochastic conditions"""
  def build_option_term_net(self):
    with tf.variable_scope("option_term"):
      out = tf.stop_gradient(self.fi_option)
      self.termination = layers.fully_connected(out, num_outputs=1,
                                                activation_fn=tf.nn.sigmoid,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="termination")
      self.termination = tf.squeeze(self.termination, 1)
      self.summaries_term.append(tf.contrib.layers.summarize_activation(self.termination))

  def build_losses(self):
    """Get the probabilities for each action taken under the intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.option, self.actions_placeholder)

    """Get the option-value functions for each option"""
    self.q_val_o = self.get_option_value_function(self.options_placeholder)

    self.non_primitve_option_mask = tf.map_fn(lambda x: tf.cond(tf.less(x, self.nb_options), lambda: x, lambda: 0),
                                               self.options_placeholder)

    """Adding comparison of predicted frame and actual next frame to tensorboard"""
    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

    """Perform singular value decomposition on the SR matrix buffer.
        Transopose eigenvectors and cojugate to be equivalent to the numpy decomposition"""
    self.matrix_sf = tf.placeholder(shape=[1, None, self.config.sf_layers[-1]],
                                    dtype=tf.float32, name="matrix_sf")
    self.eigenvalues, _, ev = tf.svd(self.matrix_sf, full_matrices=False, compute_uv=True)
    self.eigenvectors = tf.transpose(tf.conj(ev), perm=[0, 2, 1])

    """Building losses"""
    with tf.name_scope('sf_loss'):
      """TD error of successor representations"""
      sf_td_error = self.target_sf - self.sf
      self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    with tf.name_scope('aux_loss'):
      """L2 loss for the next frame prediction"""
      aux_error = self.next_obs - self.target_next_obs
      self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

    """If we use eigendirections for the options, than do TD on the eigen intra-option critics"""
    if self.config.use_eigendirections:
      with tf.name_scope('eigen_critic_loss'):
        """Zero out where the option was a primitve one"""
        eigen_td_error = tf.where(self.primitive_actions_placeholder, tf.zeros_like(self.target_eigen_return),
                                  self.target_eigen_return - self.eigen_q_val)
        self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(eigen_td_error))

    with tf.name_scope('critic_loss'):
      """TD error of the critic option-value function"""
      td_error = self.target_return - self.q_val_o
      self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * tf.square(td_error))

    with tf.name_scope('termination_loss'):
      """The advantage function for the option termination condition gradients.
                Adds a small margin for deliberation cost to drive options to extend in time. Sadly, doesn't work very well in practice"""
      self.term_err = (tf.stop_gradient(self.q_val_o) - tf.stop_gradient(self.v) + self.config.delib_margin)
      """Zero out where the option was primitve. Otherwise increase the probability of termination if the option-value function has an advantage larger than the deliberation margin over the expected value of the state"""
      self.term_loss = tf.reduce_mean(tf.where(self.primitive_actions_placeholder, tf.zeros_like(self.q_val_o),
                                               self.termination * self.term_err))
    """Add an entropy regularization for each intra-option policy, driving exploration in the action space of intra-option policies"""
    with tf.name_scope('entropy_loss'):
      """Zero out primitive options"""
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.where(self.primitive_actions_placeholder,
                                                                       tf.zeros_like(self.option),
                                                                       self.option * tf.log(self.option + 1e-7)))
    """Learn intra-option policies with policy gradients"""
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(
        tf.where(self.primitive_actions_placeholder, tf.zeros_like(self.responsible_actions),
                 tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
                   eigen_td_error)))

    self.option_loss = self.policy_loss - self.entropy_loss + self.eigen_critic_loss

  """Build gradients for the losses with respect to the network params.
      Build summaries and update ops"""
  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    """Gradients and update ops"""
    self.grads_sf, self.apply_grads_sf = self.take_gradient(self.sf_loss)
    self.grads_aux, self.apply_grads_aux = self.take_gradient(self.aux_loss)
    self.grads_critic, self.apply_grads_critic = self.take_gradient(self.critic_loss)
    self.grads_option, self.apply_grads_option = self.take_gradient(self.option_loss)
    self.grads_term, self.apply_grads_term = self.take_gradient(self.term_loss)

    """Summaries"""
    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss),
        gradient_summaries(zip(self.grads_sf, local_vars))])
    self.merged_summary_aux = tf.summary.merge(self.image_summaries +
                                               self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss),
                                                 gradient_summaries(zip(self.grads_aux, local_vars))])
    options_to_merge = self.summaries_option +\
                       [tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                        tf.summary.scalar('avg_policy_loss', self.policy_loss),
                        tf.summary.scalar('random_option_prob', self.random_option_prob),
                        # tf.summary.scalar('LR', self.lr),
                        tf.summary.scalar('avg_eigen_critic_loss', self.eigen_critic_loss),
                        gradient_summaries(zip(self.grads_option, local_vars),)]

    self.merged_summary_option = tf.summary.merge(options_to_merge)

    self.merged_summary_term = tf.summary.merge(
      self.summaries_term +
      [tf.summary.scalar('avg_termination_loss', self.term_loss),
       tf.summary.scalar('avg_termination_error', tf.reduce_mean(self.term_err)),
        gradient_summaries(zip(self.grads_term, local_vars))])

    self.merged_summary_critic = tf.summary.merge(
      self.summaries_critic + \
      [tf.summary.scalar('avg_critic_loss', self.critic_loss),
       gradient_summaries(zip(self.grads_critic, local_vars))])

