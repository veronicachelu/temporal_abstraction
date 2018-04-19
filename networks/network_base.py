import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
import os


class BaseNetwork():
  def __init__(self, scope, config, action_size, total_steps_tensor=None):
    self.scope = scope
    self.nb_states = config.input_size[0] * config.input_size[1]
    self.fc_layers = config.fc_layers
    self.sf_layers = config.sf_layers
    self.aux_fc_layers = config.aux_fc_layers
    self.action_size = action_size
    self.nb_options = config.nb_options
    self.nb_envs = config.num_agents
    self.config = config

    self.image_summaries = []
    self.summaries_sf = []
    self.summaries_aux = []
    self.summaries_option = []

    self.network_optimizer = config.network_optimizer(
      learning_rate=self.config.lr,
      epsilon=self.config.adam_epsilon, name='network_optimizer')

    if scope == 'global' and self.config.sr_matrix is not None:
      self.directions = np.zeros((config.nb_options, config.sf_layers[-1]))
      if self.config.sr_matrix == "dynamic":
        self.sf_matrix_path = os.path.join(config.stage_logdir, "sf_matrix.npy")
        if os.path.exists(self.sf_matrix_path):
          self.sf_matrix_buffer = np.load(self.sf_matrix_path)
        else:
          self.sf_matrix_buffer = np.zeros(shape=(self.config.sf_matrix_size, self.config.sf_layers[-1]), dtype=np.float32)

    self.entropy_coef = self.config.final_random_action_prob

  def build_option_term_net(self):
    with tf.variable_scope("eigen_option_term"):
      out = tf.stop_gradient(self.fi_relu)
      self.termination = layers.fully_connected(out, num_outputs=self.nb_options,
                                                activation_fn=tf.nn.sigmoid,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="fc_option_term")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.termination))

      return out

  def build_option_q_val_net(self):
    with tf.variable_scope("option_q_val"):
      out = tf.stop_gradient(self.fi_relu)
      self.q_val = layers.fully_connected(out, num_outputs=(
        self.nb_options + self.action_size) if self.config.include_primitive_options else self.nb_options,
                                          activation_fn=None,
                                          variables_collections=tf.get_collection("variables"),
                                          outputs_collections="activations", scope="fc_q_val")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.q_val))
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
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.v))

      return out

  def build_eigen_option_q_val_net(self):
    with tf.variable_scope("eigen_option_q_val"):
      out = tf.stop_gradient(self.fi_relu)
      self.eigen_q_val = layers.fully_connected(out, num_outputs=self.nb_options,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="fc_q_val")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.eigen_q_val))
    if self.config.include_primitive_options:
      concatenated_eigen_q = tf.concat([self.q_val[:, self.config.nb_options:], self.eigen_q_val], 1)
    else:
      concatenated_eigen_q = self.eigen_q_val
    self.eigenv = tf.reduce_max(concatenated_eigen_q, axis=1) * \
                  (1 - self.config.final_random_option_prob) + \
                  self.config.final_random_option_prob * tf.reduce_mean(concatenated_eigen_q, axis=1)
    self.summaries_option.append(tf.contrib.layers.summarize_activation(self.eigenv))

  def build_intraoption_policies_nets(self):
    with tf.variable_scope("eigen_option_i_o_policies"):
      out = tf.stop_gradient(self.fi_relu)
      self.options = []
      for i in range(self.nb_options):
        option = layers.fully_connected(out, num_outputs=self.action_size,
                                        activation_fn=tf.nn.softmax,
                                        biases_initializer=None,
                                        variables_collections=tf.get_collection("variables"),
                                        outputs_collections="activations", scope="policy_{}".format(i))
        self.summaries_option.append(tf.contrib.layers.summarize_activation(option))
        self.options.append(option)
      self.options = tf.stack(self.options, 1)

  def build_SF_net(self, layer_norm=False):
    with tf.variable_scope("sf"):
      out = tf.stop_gradient(self.fi_relu)
      for i, nb_filt in enumerate(self.sf_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     biases_initializer=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="sf_{}".format(i))
        if i < len(self.sf_layers) - 1:
          if layer_norm:
            out = self.layer_norm_fn(out, relu=True)
          else:
            out = tf.nn.relu(out)
        self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
      self.sf = out

  def build_placeholders(self, next_frame_channel_size):
    self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
    self.target_next_obs = tf.placeholder(
      shape=[None, self.config.input_size[0], self.config.input_size[1], next_frame_channel_size], dtype=tf.float32,
      name="target_next_obs")
    self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="options")
    self.target_eigen_return = tf.placeholder(shape=[None], dtype=tf.float32)
    self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)

  def build_losses(self):
    self.policies = self.get_intra_option_policies(self.options_placeholder)
    self.responsible_actions = self.get_responsible_actions(self.policies, self.actions_placeholder)

    if self.config.eigen:
      eigen_q_val = self.get_eigen_q(self.options_placeholder)
    q_val = self.get_q(self.options_placeholder)
    o_term = self.get_o_term(self.options_placeholder)

    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

    # self.matrix_sf = tf.placeholder(shape=[self.config.sf_matrix_size, self.sf_layers[-1]],
    #                                 dtype=tf.float32, name="matrix_sf")
    if self.config.sf_matrix_size is None:
      self.config.sf_matrix_size = self.nb_states
    self.matrix_sf = tf.placeholder(shape=[1, self.config.sf_matrix_size, self.sf_layers[-1]],
                                    dtype=tf.float32, name="matrix_sf")
    self.eigenvalues, _, ev = tf.svd(self.matrix_sf, full_matrices=True, compute_uv=True)
    self.eigenvectors = tf.transpose(tf.conj(ev), perm=[0, 2, 1])

    with tf.name_scope('sf_loss'):
      sf_td_error = self.target_sf - self.sf
    self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    with tf.name_scope('aux_loss'):
      aux_error = self.next_obs - self.target_next_obs
    self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

    if self.config.eigen:
      with tf.name_scope('eigen_critic_loss'):
        eigen_td_error = self.target_eigen_return - eigen_q_val
        self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(eigen_td_error))

    with tf.name_scope('critic_loss'):
      td_error = self.target_return - q_val
    self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * tf.square(td_error))

    with tf.name_scope('termination_loss'):
      self.term_loss = tf.reduce_mean(
        o_term * (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + 0.01))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.policies *
                                                                            tf.log(self.policies + 1e-7),
                                                                            axis=1))
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
        eigen_td_error if self.config.eigen else td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.critic_loss + self.term_loss
    if self.config.eigen:
      self.option_loss += self.eigen_critic_loss

  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    gradients_sf = tf.gradients(self.sf_loss, local_vars)
    gradients_aux = tf.gradients(self.aux_loss, local_vars)
    gradients_option = tf.gradients(self.option_loss, local_vars)
    gradients_primitive_option = tf.gradients(self.critic_loss, local_vars)

    self.var_norms = tf.global_norm(local_vars)
    grads_sf, self.grad_norms_sf = tf.clip_by_global_norm(gradients_sf, self.config.gradient_clip_norm_value)
    grads_aux, self.grad_norms_aux = tf.clip_by_global_norm(gradients_aux, self.config.gradient_clip_norm_value)
    grads_option, self.grad_norms_option = tf.clip_by_global_norm(gradients_option,
                                                                  self.config.gradient_clip_norm_value)
    grads_primitive_option, self.grad_norms_primitive_option = tf.clip_by_global_norm(gradients_primitive_option,
                                                                                      self.config.gradient_clip_norm_value)

    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss)] + [
        tf.summary.scalar('gradient_norm_sf', tf.global_norm(gradients_sf)),
        tf.summary.scalar('cliped_gradient_norm_sf', tf.global_norm(grads_sf)),
        gradient_summaries(zip(grads_sf, local_vars))])
    self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss)] + [
                                                 tf.summary.scalar('gradient_norm_aux',
                                                                   tf.global_norm(gradients_aux)),
                                                 tf.summary.scalar('cliped_gradient_norm_aux',
                                                                   tf.global_norm(grads_aux)),
                                                 gradient_summaries(zip(grads_aux, local_vars))])
    options_to_merge = self.summaries_option + [tf.summary.scalar('avg_critic_loss', self.critic_loss),
                                                tf.summary.scalar('avg_termination_loss', self.term_loss),
                                                tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                                                tf.summary.scalar('avg_policy_loss', self.policy_loss),
                                                tf.summary.scalar('gradient_norm_option',
                                                                  tf.global_norm(gradients_option)),
                                                tf.summary.scalar('cliped_gradient_norm_option',
                                                                  tf.global_norm(grads_option)),
                                                gradient_summaries(zip(grads_option, local_vars))]
    if self.config.eigen:
      options_to_merge += [tf.summary.scalar('avg_eigen_critic_loss', self.eigen_critic_loss)]

    self.merged_summary_option = tf.summary.merge(options_to_merge)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    self.apply_grads_sf = self.network_optimizer.apply_gradients(zip(grads_sf, global_vars))
    self.apply_grads_aux = self.network_optimizer.apply_gradients(zip(grads_aux, global_vars))
    self.apply_grads_option = self.network_optimizer.apply_gradients(zip(grads_option, global_vars))
    self.apply_grads_primitive_option = self.network_optimizer.apply_gradients(zip(grads_primitive_option, global_vars))

  def get_intra_option_policies(self, options):
    options_taken_one_hot = tf.one_hot(options, self.nb_options, dtype=tf.float32, name="options_one_hot")
    options_taken_one_hot = tf.tile(options_taken_one_hot[..., None], [1, 1, self.action_size])
    pi_o = tf.reduce_sum(tf.multiply(self.options, options_taken_one_hot),
                         reduction_indices=1, name="pi_o")
    return pi_o

  def get_responsible_actions(self, policies, actions):
    actions_onehot = tf.one_hot(tf.cast(actions, tf.int32), self.action_size, dtype=tf.float32,
                                name="actions_one_hot")
    responsible_actions = tf.reduce_sum(policies * actions_onehot, [1])
    return responsible_actions

  def get_eigen_q(self, o):
    options_taken_one_hot = tf.one_hot(o, self.config.nb_options,
                                       name="options_one_hot")
    eigen_q_values_o = tf.reduce_sum(tf.multiply(self.eigen_q_val, options_taken_one_hot),
                                     reduction_indices=1, name="eigen_values_Q")
    return eigen_q_values_o

  def get_q(self, o):
    options_taken_one_hot = tf.one_hot(o, (
      self.config.nb_options + self.action_size) if self.config.include_primitive_options else self.config.nb_options,
                                       name="options_one_hot")
    q_values_o = tf.reduce_sum(tf.multiply(self.q_val, options_taken_one_hot),
                               reduction_indices=1, name="values_Q")
    return q_values_o

  def get_primitive(self, o):
    primitive_actions = tf.where(o >= self.nb_options,
                                 tf.ones_like(self.current_option),
                                 tf.zeros_like(self.current_option))
    return primitive_actions

  def get_o_term(self, o, boolean_value=False):
    options_taken_one_hot = tf.one_hot(o, self.config.nb_options,
                                       name="options_one_hot")
    o_term = tf.reduce_sum(tf.multiply(self.termination, options_taken_one_hot),
                           reduction_indices=1, name="o_terminations")
    if boolean_value:
      local_random = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32, name="rand_o_term")
      o_term = o_term > local_random
    return o_term

  def get_o_r_i(self, o):
    options_taken_one_hot = tf.one_hot(o, (
      self.config.nb_options + self.action_size) if self.config.include_primitive_options else self.config.nb_options,
                                       name="options_one_hot")
    q_values_o = tf.reduce_sum(tf.multiply(self.r_i, options_taken_one_hot),
                               reduction_indices=1, name="values_Q")
    return q_values_o

  def layer_norm_fn(self, x, relu=True):
    x = layers.layer_norm(x, scale=True, center=True)
    if relu:
      x = tf.nn.relu(x)
    return x

  def compute_gradients(self, losses):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    var_norms = tf.global_norm(local_vars)
    gradients_list = []
    grads_list = []
    apply_grads_list = []
    grad_norm_list = []
    for loss in losses:
      gradients = tf.gradients(loss, local_vars)
      gradients_list.append(gradients)
      grads, grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm_value)
      grads_list.append(grads)
      grad_norm_list.append(grad_norms)
      apply_grads = self.network_optimizer.apply_gradients(zip(grads, global_vars))
      apply_grads_list.append(apply_grads)

    return grads_list, grad_norm_list, apply_grads_list