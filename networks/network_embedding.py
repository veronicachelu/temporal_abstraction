import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_base import BaseNetwork
import os


class EmbeddingNetwork(BaseNetwork):
  def __init__(self, scope, config, action_size, total_steps_tensor=None):
    super(EmbeddingNetwork, self).__init__(scope, config, action_size, total_steps_tensor)
    self.summaries_term = []
    self.summaries_critic = []
    self.summaries_eigen_critic = []
    self.build_network()

  def build_feature_net(self, out):
    with tf.variable_scope("fi"):
      for i, nb_filt in enumerate(self.fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fc_{}".format(i))

        if i < len(self.fc_layers) - 1:
          out = tf.nn.relu(out)
        self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.summaries_option.append(tf.contrib.layers.summarize_activation(out))
      self.fi = out
      self.fi_relu = tf.nn.relu(self.fi)
      self.option_direction_placeholder = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32,
                                                         name="option_direction")
      self.fi_option = tf.add(tf.stop_gradient(self.fi), self.option_direction_placeholder)

      return out

  def build_next_frame_prediction_net(self):
    with tf.variable_scope("aux_action_fc"):
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
      actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.fc_layers[-1],
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc")

    with tf.variable_scope("aux_next_frame"):
      out = tf.add(self.fi, actions)
      # out = tf.nn.relu(out)
      for i, nb_filt in enumerate(self.aux_fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fc_{}".format(i))
        if i < len(self.aux_fc_layers) - 1:
          out = tf.nn.relu(out)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      self.next_obs = tf.reshape(out,
                                 (-1, self.config.input_size[0], self.config.input_size[1], self.config.history_size))

  def build_eigen_option_q_val_net(self):
    with tf.variable_scope("eigen_option_q"):
      out = tf.stop_gradient(self.fi_option)
      self.eigen_q_val = layers.fully_connected(out, num_outputs=1,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="eigen_q_val")
      self.eigen_q_val = tf.squeeze(self.eigen_q_val, 1)
      self.summaries_eigen_critic.append(tf.contrib.layers.summarize_activation(self.eigen_q_val))

  def build_intraoption_policies_nets(self):
    with tf.variable_scope("option_pi"):
      out = tf.stop_gradient(self.fi_option)
      self.option = layers.fully_connected(out, num_outputs=self.action_size,
                                           activation_fn=tf.nn.softmax,
                                           biases_initializer=None,
                                           variables_collections=tf.get_collection("variables"),
                                           outputs_collections="activations", scope="policy")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.option))

  def build_option_term_net(self):
    with tf.variable_scope("option_term"):
      out = tf.stop_gradient(self.fi_option)
      self.termination = layers.fully_connected(out, num_outputs=1,
                                                activation_fn=tf.nn.sigmoid,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="termination")
      self.termination = tf.squeeze(self.termination, 1)
      self.summaries_term.append(tf.contrib.layers.summarize_activation(self.termination))

  def build_option_q_val_net(self):
    with tf.variable_scope("option_q"):
      out = tf.stop_gradient(self.fi_relu)
      self.q_val = layers.fully_connected(out, num_outputs=(
        self.nb_options + self.action_size) if self.config.include_primitive_options else self.nb_options,
                                          activation_fn=None,
                                          variables_collections=tf.get_collection("variables"),
                                          outputs_collections="activations", scope="q_val")
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.q_val))
      self.max_q_val = tf.reduce_max(self.q_val, 1)
      self.max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
      self.exp_options = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0, maxval=(
        self.nb_options + self.action_size) if self.config.include_primitive_options else self.nb_options,
                                           dtype=tf.int32)
      self.local_random = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0., maxval=1., dtype=tf.float32,
                                            name="rand_options")
      self.condition = self.local_random > self.config.final_random_option_prob

      self.current_option = tf.where(self.condition, self.max_options, self.exp_options)
      self.primitive_action = tf.where(self.current_option >= self.nb_options,
                                       tf.ones_like(self.current_option),
                                       tf.zeros_like(self.current_option))
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.current_option))
      self.v = self.max_q_val * (1 - self.config.final_random_option_prob) + \
               self.config.final_random_option_prob * tf.reduce_mean(self.q_val, axis=1)
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.v))

      return out

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.observation = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = layers.flatten(out, scope="flatten")

      _ = self.build_feature_net(out)
      _ = self.build_option_term_net()
      _ = self.build_option_q_val_net()

      self.build_eigen_option_q_val_net()

      self.build_intraoption_policies_nets()
      self.build_SF_net(layer_norm=False)
      self.build_next_frame_prediction_net()
      self.build_placeholders(self.config.history_size)

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  def build_losses(self):
    self.responsible_actions = self.get_responsible_actions(self.option, self.actions_placeholder)

    self.q_val_o = self.get_q(self.options_placeholder)
    # o_term = self.get_o_term(self.options_placeholder)

    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

    if self.config.sr_matrix == "dynamic":
      self.sf_matrix_size = self.config.sf_matrix_size
    else:
      self.sf_matrix_size = 104
    self.matrix_sf = tf.placeholder(shape=[1, self.sf_matrix_size, self.sf_layers[-1]],
                                    dtype=tf.float32, name="matrix_sf")
    self.eigenvalues, _, ev = tf.svd(self.matrix_sf, full_matrices=False, compute_uv=True)
    self.eigenvectors = tf.transpose(tf.conj(ev), perm=[0, 2, 1])

    with tf.name_scope('sf_loss'):
      sf_td_error = self.target_sf - self.sf
    self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    with tf.name_scope('aux_loss'):
      aux_error = self.next_obs - self.target_next_obs
    self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

    with tf.name_scope('eigen_critic_loss'):
      eigen_td_error = self.target_eigen_return - self.eigen_q_val
      self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * huber_loss(eigen_td_error))

    with tf.name_scope('critic_loss'):
      td_error = self.target_return - self.q_val_o
    self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * huber_loss(td_error))

    with tf.name_scope('termination_loss'):
      self.term_loss = tf.reduce_mean(
        self.termination * (tf.stop_gradient(self.q_val_o) - tf.stop_gradient(self.v) + 0.01))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(self.option * tf.log(self.option + 1e-7))
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
        eigen_td_error))

    self.option_loss = self.policy_loss - self.entropy_loss

  def gradients_and_summaries(self):
    self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    self.gradients_sf = tf.gradients(self.sf_loss, self.local_vars)
    self.gradients_aux = tf.gradients(self.aux_loss, self.local_vars)
    self.gradients_option = tf.gradients(self.option_loss, self.local_vars)
    self.gradients_critic = tf.gradients(self.critic_loss, self.local_vars)
    self.gradients_eigen_critic = tf.gradients(self.eigen_critic_loss, self.local_vars)
    self.gradients_primitive_option = tf.gradients(self.critic_loss, self.local_vars)
    self.gradients_term = tf.gradients(self.term_loss, self.local_vars)

    self.var_norms = tf.global_norm(self.local_vars)

    self.grad_norms_sf = tf.global_norm(self.gradients_sf)
    self.grads_sf, self.grad_norms_sf = tf.clip_by_global_norm(self.gradients_sf, self.config.gradient_clip_norm_value,
                                                               use_norm=self.grad_norms_sf)
    self.grad_norms_term = tf.global_norm(self.gradients_term)
    self.grads_term, self.grad_norms_term = tf.clip_by_global_norm(self.gradients_term,
                                                                   self.config.gradient_clip_norm_value,
                                                                   use_norm=self.grad_norms_term)
    self.grad_norms_aux = tf.global_norm(self.gradients_aux)
    self.grads_aux, self.grad_norms_aux = tf.clip_by_global_norm(self.gradients_aux,
                                                                 self.config.gradient_clip_norm_value,
                                                                 use_norm=self.grad_norms_aux)
    self.grad_norms_option = tf.global_norm(self.gradients_option)
    self.grads_option, self.grad_norms_option = tf.clip_by_global_norm(self.gradients_option,
                                                                       self.config.gradient_clip_norm_value,
                                                                       use_norm=self.grad_norms_option)
    self.grad_norms_critic = tf.global_norm(self.gradients_critic)
    self.grads_critic, self.grad_norms_critic = tf.clip_by_global_norm(self.gradients_critic,
                                                                       self.config.gradient_clip_norm_value,
                                                                       use_norm=self.grad_norms_critic)
    self.grad_norms_eigen_critic = tf.global_norm(self.gradients_eigen_critic)
    self.grads_eigen_critic, self.grad_norms_eigen_critic = tf.clip_by_global_norm(self.gradients_eigen_critic,
                                                                       self.config.gradient_clip_norm_value,
                                                                       use_norm=self.grad_norms_eigen_critic)

    grad_check_option = [tf.check_numerics(g, "GRAD option is NAN") for g in self.grads_option if g is not None]
    self.grads_primitive_option, self.grad_norms_primitive_option = tf.clip_by_global_norm(
      self.gradients_primitive_option,
      self.config.gradient_clip_norm_value)

    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss)] + [
        tf.summary.scalar('gradient_norm_sf', tf.global_norm(self.gradients_sf)),
        tf.summary.scalar('cliped_gradient_norm_sf', tf.global_norm(self.grads_sf)),
        gradient_summaries(zip(self.grads_sf, self.local_vars))])
    self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss)] + [
                                                 tf.summary.scalar('gradient_norm_aux',
                                                                   tf.global_norm(self.gradients_aux)),
                                                 tf.summary.scalar('cliped_gradient_norm_aux',
                                                                   tf.global_norm(self.grads_aux)),
                                                 gradient_summaries(zip(self.grads_aux, self.local_vars))])
    options_to_merge = self.summaries_option + [tf.summary.scalar('avg_critic_loss', self.critic_loss),
                                                tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                                                tf.summary.scalar('avg_policy_loss', self.policy_loss),
                                                tf.summary.scalar('gradient_norm_option',
                                                                  tf.global_norm(self.gradients_option)),
                                                tf.summary.scalar('cliped_gradient_norm_option',
                                                                  tf.global_norm(self.grads_option)),
                                                tf.summary.scalar('avg_eigen_critic_loss', self.eigen_critic_loss),
                                                gradient_summaries(zip(self.grads_option, self.local_vars))]

    self.merged_summary_option = tf.summary.merge(options_to_merge)

    self.merged_summary_term = tf.summary.merge(
      self.summaries_term + [tf.summary.scalar('avg_termination_loss', self.term_loss),
                             gradient_summaries(zip(self.grads_term, self.local_vars))])

    self.merged_summary_critic = tf.summary.merge(
      self.summaries_critic + [tf.summary.scalar('avg_critic_loss', self.critic_loss),
                             gradient_summaries(zip(self.grads_critic, self.local_vars))])

    self.merged_summary_eigen_critic = tf.summary.merge(
      self.summaries_eigen_critic + [tf.summary.scalar('avg_eigen_critic_loss', self.eigen_critic_loss),
                             gradient_summaries(zip(self.grads_eigen_critic, self.local_vars))])

    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    self.apply_grads_sf = self.network_optimizer.apply_gradients(zip(self.grads_sf, global_vars))
    self.apply_grads_term = self.network_optimizer.apply_gradients(zip(self.grads_term, global_vars))
    self.apply_grads_aux = self.network_optimizer.apply_gradients(zip(self.grads_aux, global_vars))
    with tf.control_dependencies(grad_check_option):
      self.apply_grads_option = self.network_optimizer.apply_gradients(zip(self.grads_option, global_vars))

    self.apply_grads_critic = self.network_optimizer.apply_gradients(zip(self.grads_critic, global_vars))
    self.apply_grads_eigen_critic = self.network_optimizer.apply_gradients(zip(self.grads_eigen_critic, global_vars))

    self.apply_grads_primitive_option = self.network_optimizer.apply_gradients(
      zip(self.grads_primitive_option, global_vars))
