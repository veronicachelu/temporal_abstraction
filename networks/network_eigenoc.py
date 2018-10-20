import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
import os


"""Function approximation network for the option critic policies and value functions"""
class EignOCNetwork():
  def __init__(self, scope, config, action_size):
    self.scope = scope
    self.config = config

    """The size of the input space flatten out"""
    self.nb_states = config.input_size[0] * config.input_size[1]

    """The size of the action space"""
    self.action_size = action_size

    """The the option space of the policy over options."""
    self.nb_options = config.nb_options

    """Creating buffers for holding summaries"""
    self.image_summaries = []
    self.summaries_sf = []
    self.summaries_aux = []
    self.summaries_option = []
    self.summaries_term = []
    self.summaries_critic = []
    self.summaries_eigen_critic = []

    """Instantiating optimizer"""
    self.network_optimizer = config.network_optimizer(
      self.config.lr, name='network_optimizer')

    """Initialize the eigendirections of the options"""
    if self.config.use_eigendirections:
      self.init_eigendirections()

    """This helps with exploration inside intra-option policies"""
    self.entropy_coef = self.config.final_random_action_prob

    """The probability of taking a random option"""
    self.random_option_prob = tf.Variable(self.config.initial_random_option_prob, trainable=False,
                                          name="prob_of_random_option", dtype=tf.float32)

    """If we want to gradually converge to a deterministic option"""
    self.decrease_prob_of_random_option = tf.assign_sub(self.random_option_prob, tf.constant(
      (self.config.initial_random_option_prob - self.config.final_random_option_prob) / self.config.explore_options_episodes))

    self.build_network()

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.observation = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = layers.flatten(out, scope="flatten")

      """Build the encoder for the latent representation space"""
      self.build_feature_net(out)

      """Build the branch for next frame prediction that trains the latent state representation"""
      self.build_next_frame_prediction_net()

      """Build the option termination stochastic conditions"""
      self.build_option_term_net()

      """Build the option action-value functions"""
      self.build_option_q_val_net()

      """If we use eigendirections we need another critic for the intra-option policies that uses the mixture of pseudo internal rewards and external rewards"""
      if self.config.use_eigendirections:
        """Build the intra-option policies critics"""
        self.build_eigen_option_q_val_net()

      """Build the intra-option policies"""
      self.build_intraoption_policies_nets()

      """Build the branch for constructing the successor representation latent space"""
      self.build_SF_net(layer_norm=False)

      """Add additional placeholders for losses and such"""
      self.build_placeholders(self.config.history_size)

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  """Build the encoder for the latent representation space"""
  def build_feature_net(self, out):
    with tf.variable_scope("fi"):
      for i, nb_filt in enumerate(self.config.fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fi_{}".format(i))
        if i < len(self.config.fc_layers) - 1:
          out = tf.nn.relu(out)

      """This is the latent state representation"""
      self.fi = out
      out = tf.nn.relu(out)
      self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))

      """This is the latent state representation with relu on top, will be used for all the other layers later on"""
      self.fi_relu = out

  """Build the branch for next frame prediction that trains the latent state representation"""
  def build_next_frame_prediction_net(self):
    """Plugging in the current action taken into the environment for next frame prediction"""
    with tf.variable_scope("action_fc"):
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
      actions = tf.one_hot(self.actions_placeholder, depth=self.action_size)
      actions = layers.fully_connected(actions, num_outputs=self.config.fc_layers[-1],
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="action_fc")

    """Decoder from latent space fi(s) to the next state"""
    with tf.variable_scope("aux_next_frame"):
      out = tf.add(self.fi, actions)
      out = tf.nn.relu(out)
      for i, nb_filt in enumerate(self.config.aux_fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="aux_fc_{}".format(i))
        if i < len(self.config.aux_fc_layers) - 1:
          out = tf.nn.relu(out)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      self.next_obs = tf.reshape(out,
                                 (-1, self.config.input_size[0], self.config.input_size[1], self.config.history_size))

  """Build the option termination stochastic conditions"""
  def build_option_term_net(self):
    with tf.variable_scope("option_term"):
      out = tf.stop_gradient(self.fi_relu)
      self.termination = layers.fully_connected(out, num_outputs=self.nb_options,
                                                activation_fn=tf.nn.sigmoid,
                                                weights_initializer=layers.xavier_initializer(uniform=False),
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="option_term")
      self.summaries_term.append(tf.contrib.layers.summarize_activation(self.termination))

  """Build the option action-value functions"""
  def build_option_q_val_net(self):
    with tf.variable_scope("option_q_val"):
      out = tf.stop_gradient(self.fi_relu)
      """If we accept primitive actions as options, we add action-value functions to those as well and we increase the number of units to include them at the end"""
      self.q_val = layers.fully_connected(out, num_outputs=
      self.nb_options + self.action_size if self.config.include_primitive_options else self.nb_options,
                                          activation_fn=None,
                                          variables_collections=tf.get_collection("variables"),
                                          outputs_collections="activations", scope="q_val")
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.q_val))
      """The maximum action-value function"""
      self.max_q_val = tf.reduce_max(self.q_val, 1)
      """The option corresponding the max Q value"""
      self.max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
      """The expected Q value under a random uniform policy over options"""
      self.exp_options = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0,
                                           maxval=self.nb_options + self.action_size if self.config.include_primitive_options else self.nb_options,
                                           dtype=tf.int32)
      """Take the random option with probability self.random_option_prob"""
      self.local_random = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0., maxval=1., dtype=tf.float32,
                                            name="rand_options")
      self.condition = self.local_random > self.random_option_prob
      """The option taken"""
      self.current_option = tf.where(self.condition, self.max_options, self.exp_options, name="current_option")

      """Boolean value indicating wheather the option took is a primitive action (in the case that primitive actions are allowed to be options) or a regular temporally extened option"""
      self.primitive_action = tf.where(self.current_option >= self.nb_options,
                                       tf.ones_like(self.current_option),
                                       tf.zeros_like(self.current_option))
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.current_option))

      """The expected value of a state - takes into account the random option probability"""
      self.v = tf.identity(self.max_q_val * (1 - self.random_option_prob) + \
                           self.random_option_prob * tf.reduce_mean(self.q_val, axis=1), name="V")
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.v))

  """Build the intra-option policies critics"""
  def build_eigen_option_q_val_net(self):
    with tf.variable_scope("eigen_option_q_val"):
      out = tf.stop_gradient(self.fi_relu)
      self.eigen_q_val = layers.fully_connected(out, num_outputs=self.nb_options,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="eigen_q_val")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.eigen_q_val))

    """If we allow primitive actions to be options, we need to account for them when calculating the expected value of a state-option pair, so for primitive options we compute the option-value function using the action-value function at option level as opposed to the eigen-option-value function"""
    if self.config.include_primitive_options:
      concatenated_eigen_q = tf.concat([self.q_val[:, self.config.nb_options:], self.eigen_q_val], 1)
    else:
      concatenated_eigen_q = self.eigen_q_val
    """Expected eigen value function which takes into account pseude rewards"""
    self.eigenv = tf.identity(tf.reduce_max(concatenated_eigen_q, axis=1) * \
                              (1 - self.random_option_prob) + \
                              self.random_option_prob * tf.reduce_mean(concatenated_eigen_q, axis=1), name="eigen_V")
    self.summaries_option.append(tf.contrib.layers.summarize_activation(self.eigenv))

  """Build the intra-option policies"""
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

  """Build the branch for constructing the successor representation latent space"""
  def build_SF_net(self, layer_norm=False):
    with tf.variable_scope("succ_feat"):
      out = tf.stop_gradient(self.fi_relu)
      for i, nb_filt in enumerate(self.config.sf_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     biases_initializer=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="sf_{}".format(i))
        if i < len(self.config.sf_layers) - 1:
          if layer_norm:
            out = self.layer_norm_fn(out, relu=True)
          else:
            out = tf.nn.relu(out)
        self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
      self.sf = out

  """Add additional placeholders for losses and such"""
  def build_placeholders(self, next_frame_channel_size):
    """Placeholder for the target next frame successor representation for learning is TD the successor representation"""
    self.target_sf = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32, name="target_SF")
    """Placeholder for the target next frame observation"""
    self.target_next_obs = tf.placeholder(
      shape=[None, self.config.input_size[0], self.config.input_size[1], next_frame_channel_size], dtype=tf.float32,
      name="target_next_obs")
    """Placeholder for the current option"""
    self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="options")
    """Placeholder for the target return for n-step prediction of the intra-option critics"""
    self.target_eigen_return = tf.placeholder(shape=[None], dtype=tf.float32)
    """Placeholder for the target return for n-step prediction of the option critics"""
    self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
    """Placeholder indicating wheather the current option is a primitive action or not"""
    self.primitive_actions_placeholder = tf.placeholder(shape=[None], dtype=tf.bool,
                                                        name="primitive_actions_placeholder")

  def build_losses(self):
    """Get the option-value functions for each option"""
    q_val = self.get_option_value_function(self.options_placeholder)

    self.non_primitve_option_mask = tf.map_fn(lambda x: tf.cond(tf.less(x, self.nb_options), lambda: x, lambda: 0), self.options_placeholder)

    """Get the intra-option policies for each option taken"""
    self.policies = self.get_intra_option_policies(self.non_primitve_option_mask)
    """Get the probabilities for each action taken of each intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.policies, self.actions_placeholder)

    """If we've used a different critic for the intra-option policies, we need to mask out primitive options taken - we use the first option - 0, but it doesn't matter because we mask out the loss at those points anyways. This is just to prevent indexing errors for non-primitive options corresponding to indices larger that the number of non-primitive options"""
    if self.config.use_eigendirections:
      eigen_q_val = tf.where(self.primitive_actions_placeholder, q_val, self.get_eigen_option_value_function(self.non_primitve_option_mask))

    """The option termination probabilities for the options executed. If we executed primitive options, we zero out the probabilities, although it doesn't matter since we will zero out their contributions in the loss"""
    self.o_term = tf.where(self.primitive_actions_placeholder, tf.zeros(tf.shape(self.primitive_actions_placeholder), dtype=tf.float32),
                      self.get_option_termination(self.non_primitve_option_mask))

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
                                  self.target_eigen_return - eigen_q_val)
        self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(eigen_td_error))

    with tf.name_scope('critic_loss'):
      """TD error of the critic option-value function"""
      td_error = self.target_return - q_val
      self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * tf.square(td_error))

    with tf.name_scope('termination_loss'):
      """The advantage function for the option termination condition gradients.
          Adds a small margin for deliberation cost to drive options to extend in time. Sadly, doesn't work very well in practice"""
      self.term_err = (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + self.config.delib_margin)
      """Zero out where the option was primitve. Otherwise increase the probability of termination if the option-value function has an advantage larger than the deliberation margin over the expected value of the state"""
      self.term_loss = tf.reduce_mean(tf.where(self.primitive_actions_placeholder, tf.zeros_like(q_val), self.o_term * self.term_err))

    """Add an entropy regularization for each intra-option policy, driving exploration in the action space of intra-option policies"""
    with tf.name_scope('entropy_loss'):
      """Zero out primitive options"""
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.where(self.primitive_actions_placeholder, tf.zeros(tf.shape(self.primitive_actions_placeholder), dtype=tf.float32), tf.reduce_sum(self.policies * tf.log(self.policies + 1e-7), axis=1)))

    """Learn intra-option policies with policy gradients"""
    with tf.name_scope('policy_loss'):
      """Zero out primitive options.
         Use the TD-error of the eigen intra-option critics if we use eigen-directions with mixture of reward signals.
         Otherwise use the regular TD-error of the option critic"""
      self.policy_loss = -tf.reduce_mean(tf.where(self.primitive_actions_placeholder, tf.zeros_like(self.responsible_actions),
                                                  tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(eigen_td_error if self.config.use_eigendirections else td_error)))

    """The option loss is composed of the policy loss and the entropy loss"""
    self.option_loss = self.policy_loss - self.entropy_loss

    """If we use eigendirections, than we add the loss for the eigen intra-option critic with mixed reward signal to the option loss as well"""
    if self.config.use_eigendirections:
      self.option_loss += self.eigen_critic_loss

  """Return the gradients of the loss with respect to the local parameters and the update ops for applying them to the global shared network params."""
  def take_gradient(self, loss):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    gradients = tf.gradients(loss, local_vars)
    var_norms = tf.global_norm(local_vars)
    grads, grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm_value)
    apply_grads = self.network_optimizer.apply_gradients(zip(grads, global_vars))
    return grads, apply_grads

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
    self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss),
                                                 gradient_summaries(zip(self.grads_aux, local_vars))])
    options_to_merge = self.summaries_option +\
                       self.summaries_critic + \
                       [tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                        tf.summary.scalar('avg_policy_loss', self.policy_loss),
                        tf.summary.scalar('random_option_prob', self.random_option_prob),
                        # tf.summary.scalar('LR', self.lr),
                        gradient_summaries(zip(self.grads_option, local_vars))]
    self.merged_summary_term = tf.summary.merge(
      self.summaries_term + [tf.summary.scalar('avg_termination_loss', self.term_loss)] + [
        tf.summary.scalar('avg_termination_error', tf.reduce_mean(self.term_err)),
        gradient_summaries(zip(self.grads_term, local_vars))])

    self.merged_summary_critic = tf.summary.merge(
      self.summaries_term + [tf.summary.scalar('avg_critic_loss', self.critic_loss),] + [
        gradient_summaries(zip(self.grads_critic, local_vars))])

    if self.config.use_eigendirections:
      options_to_merge += [tf.summary.scalar('avg_eigen_critic_loss', self.eigen_critic_loss),]

    self.merged_summary_option = tf.summary.merge(options_to_merge)

  """Get the option-value function for the current option"""
  def get_option_value_function(self, o):
    indices = tf.stack([tf.range(tf.shape(o)[0]), o], axis=1)
    q_o = tf.gather_nd(self.q_val, indices, "q_val_o")

    return q_o

  """Get the option-value function for the current option using the intra-option critic which uses the mixture of reward signals: internal & external"""
  def get_eigen_option_value_function(self, o):
    indices = tf.stack([tf.range(tf.shape(o)[0]), o], axis=1)
    eigen_q_o = tf.gather_nd(self.eigen_q_val, indices, name="eigen_q_val_o")

    return eigen_q_o


  """Get the termination condition for the current option"""
  def get_option_termination(self, o, boolean_value=False):
    indices = tf.stack([tf.range(tf.shape(o)[0]), o], axis=1)
    o_term = tf.gather_nd(self.termination, indices)
    if boolean_value:
      local_random = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32, name="rand_o_term")
      o_term = o_term > local_random
    return o_term

  """Return a mask indicating for each option wheather it is primitive or not"""
  def get_primitive_option_mask(self, o):
    primitive_actions = tf.where(o >= self.nb_options,
                                 tf.ones_like(self.current_option),
                                 tf.zeros_like(self.current_option))
    return primitive_actions

  """Get the policy probabilities corresponding to the actions executed in the environment"""
  def get_responsible_actions(self, policies, actions):
    indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
    responsible_actions = tf.gather_nd(policies, indices)

    # actions_onehot = tf.one_hot(tf.cast(actions, tf.int32), self.action_size, dtype=tf.float32, name="actions_one_hot")
    # responsible_actions = tf.reduce_sum(policies * actions_onehot, [1])
    return responsible_actions

  """Get the intra-option corresponding to each option taken in the environment"""
  def get_intra_option_policies(self, options):
    indices = tf.stack([tf.range(tf.shape(options)[0]), options], axis=1)
    pi_o = tf.gather_nd(self.options, indices, name="pi_o")
    # options_taken_one_hot = tf.one_hot(options, self.nb_options, dtype=tf.float32, name="options_one_hot")
    # options_taken_one_hot = tf.tile(options_taken_one_hot[..., None], [1, 1, self.action_size])
    # pi_o = tf.reduce_sum(tf.multiply(self.options, options_taken_one_hot),
    #                      reduction_indices=1, name="pi_o")
    return pi_o

  def init_eigendirections(self):
    """If this is the shared global network between thread agents, we keep here the eigendirections shared between all agents"""
    if self.scope == 'global' and self.config.sr_matrix is not None:
      l = "0"
      if self.config.resume:
        """If we resume training, load the eigen directions as well from saved checkpoint"""
        with open(os.path.join(self.config.load_from, "models/checkpoint")) as f:
          l = f.readline()
          l = l.split(" ")
          l = l[1]
          l = l[1: -1]
          l = l.split("-")[1]
          l = l.split(".")[0]

      self.directions_path = os.path.join(self.config.logdir, "eigen_directions_{}.npy".format(l))

      """If the path exists, load them. Otherwise initialize all directions with zeros"""
      if os.path.exists(self.directions_path):
        self.directions = np.load(self.directions_path)
        self.directions_init = True
      else:
        self.directions = np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
        self.directions_init = False

      """If we use a dynamic buffer with a ring structure for keeping track of successor representations as we explore"""
      if self.config.sr_matrix == "dynamic":
        self.sf_matrix_path = os.path.join(self.config.logdir, "sf_matrix_{}.npy".format(l))
        """If the path exists, we load the buffer from disk. Otherwise we init with zeros"""
        if os.path.exists(self.sf_matrix_path):
          self.sf_matrix_buffer = np.load(self.sf_matrix_path)
        else:
          self.sf_matrix_buffer = np.zeros(shape=(self.config.sf_matrix_size, self.config.sf_layers[-1]),
                                           dtype=np.float32)

  def layer_norm_fn(self, x, relu=True):
    x = layers.layer_norm(x, scale=True, center=True)
    if relu:
      x = tf.nn.relu(x)
    return x