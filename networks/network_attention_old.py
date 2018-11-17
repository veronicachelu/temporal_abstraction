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

"""Function approximation network for the option critic policies and value functions when options are given as embeddings corresponding to the spectral decomposition of the SR matrix"""
class AttentionNetwork(EignOCNetwork):
  def __init__(self, scope, config, action_size):
    self.goal_embedding_size = config.sf_layers[-1]
    super(AttentionNetwork, self).__init__(scope, config, action_size)

    if self.config.use_clustering:
      self.init_clustering()

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
      # self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))

      """This is the latent state representation with relu on top, will be used for all the other layers later on"""
      self.fi_relu = out

    with tf.variable_scope("option_features"):
      intra_features = layers.fully_connected(tf.stop_gradient(self.fi_relu),
                                        num_outputs=self.action_size * self.goal_embedding_size,
                                        activation_fn=None,
                                        variables_collections=tf.get_collection("variables"),
                                        outputs_collections="activations", scope="U")
      self.policy_features = tf.reshape(intra_features, [-1, self.action_size, self.goal_embedding_size],
                                 name="policy_features")
      self.value_features = tf.identity(intra_features, name="value_features")

  """Build the intra-option policies critics"""
  def build_eigen_option_q_val_net(self):
    with tf.variable_scope("eigen_option_q_val"):
      value_embedding = tf.get_variable("value_embedding",
                                     shape=[self.action_size * self.goal_embedding_size + self.goal_embedding_size, 1],
                                     initializer=normalized_columns_initializer(1.0))
      self.value_mix = tf.matmul(tf.concat([self.value_features, self.current_option_direction], 1), value_embedding,
                                 name="fc_option_value")
      self.value_mix = tf.squeeze(self.value_mix, 1)
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.value_mix))

  """Build the intra-option policies"""
  def build_intraoption_policies_nets(self):
    with tf.variable_scope("option_pi"):
      policy = tf.squeeze(tf.matmul(self.policy_features, tf.expand_dims(self.current_option_direction, 2)), 2)
      policy = tf.contrib.layers.flatten(policy)
      self.option_policy = tf.nn.softmax(policy, name="policy")

      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.option_policy))

  """Build the option action-value functions"""
  def build_option_q_val_net(self):
    with tf.variable_scope("option_policy"):
      out = tf.stop_gradient(self.fi_relu)
      """If we accept primitive actions as options, we add action-value functions to those as well and we increase the number of units to include them at the end"""

      direction_features = layers.fully_connected(out, num_outputs=self.goal_embedding_size,
                                          activation_fn=None,
                                          variables_collections=tf.get_collection("variables"),
                                          outputs_collections="activations", scope="direction_features")
      self.value = layers.fully_connected(out, num_outputs=1, activation_fn=None,
                                                  variables_collections=tf.get_collection("variables"),
                                                  outputs_collections="activations", scope="V")
      self.value = tf.squeeze(self.value, 1)
      # self.summaries_critic.append(tf.contrib.layers.summarize_activation(direction_features))

      content_match = tf.tensordot(direction_features, self.direction_clusters, axes=[[1], [1]])
      self.attention_weights = tf.nn.softmax(content_match)

      current_direction = tf.tensordot(self.attention_weights, self.direction_clusters, axes=[[1], [0]])
      self.current_option_direction = tf.check_numerics(
                            current_direction,
                            "NaN in current_direction",
                            name=None
                          )
      #tf.cast(tf.nn.l2_normalize(tf.cast(current_direction, tf.float64), axis=1), tf.float32)

      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.current_option_direction))


  def build_losses(self):
    """Get the probabilities for each action taken under the intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.option_policy, self.actions_placeholder)

    """Adding comparison of predicted frame and actual next frame to tensorboard"""
    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

    """Building losses"""
    with tf.name_scope('sf_loss'):
      """TD error of successor representations"""
      sf_td_error = self.target_sf - self.sf
      self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    with tf.name_scope('aux_loss'):
      """L2 loss for the next frame prediction"""
      aux_error = self.next_obs - self.target_next_obs
      self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

    with tf.name_scope('critic_loss'):
      td_error = self.target_return - self.value
      self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * tf.square(td_error))

    with tf.name_scope('direction_loss'):
      self.direction_loss = -tf.reduce_mean(self.cosine_similarity(self.target_fi_horiz - self.fi,
                                                                   self.current_option_direction, 1) *
                                            tf.stop_gradient(td_error))

    """If we use eigendirections for the options, than do TD on the eigen intra-option critics"""
    with tf.name_scope('mix_critic_loss'):
      """Zero out where the option was a primitve one"""
      mix_td_error = self.target_mix_return - self.value_mix
      self.mix_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(mix_td_error))

    """Add an entropy regularization for each intra-option policy, driving exploration in the action space of intra-option policies"""
    with tf.name_scope('entropy_loss'):
      """Zero out primitive options"""
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(self.option_policy * tf.log(self.option_policy + 1e-7))
    """Learn intra-option policies with policy gradients"""
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
                          mix_td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.mix_critic_loss

  def cosine_similarity(self, v1, v2, axis):
    def l2_normalize(x, axis):
        norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True))
        return tf.maximum(x, 1e-8) / tf.maximum(norm, 1e-8)
    v1_norm = l2_normalize(v1, axis)
    v2_norm = l2_normalize(v2, axis)
    sim = tf.matmul(
      v1_norm, v2_norm, transpose_b=True)

    return sim

  """Build gradients for the losses with respect to the network params.
      Build summaries and update ops"""
  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    """Gradients and update ops"""
    self.grads_sf, self.apply_grads_sf = self.take_gradient(self.sf_loss)
    self.grads_aux, self.apply_grads_aux = self.take_gradient(self.aux_loss)
    self.grads_option, self.apply_grads_option = self.take_gradient(self.option_loss)
    self.grads_critic, self.apply_grads_critic = self.take_gradient(self.critic_loss)
    self.grads_direction, self.apply_grads_direction = self.take_gradient(self.direction_loss)

    """Summaries"""
    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss),
        gradient_summaries(zip(self.grads_sf, local_vars))])
    self.merged_summary_aux = tf.summary.merge(self.image_summaries +
                                               self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss),])
                                                 # gradient_summaries(zip(self.grads_aux, local_vars))])
    options_to_merge = self.summaries_option +\
                       [tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                        tf.summary.scalar('avg_policy_loss', self.policy_loss),
                        tf.summary.scalar('avg_eigen_critic_loss', self.mix_critic_loss), ]
                        # gradient_summaries(zip(self.grads_option, local_vars),)]

    self.merged_summary_option = tf.summary.merge(options_to_merge)

    self.merged_summary_critic = tf.summary.merge(
                                              [tf.summary.scalar('critic_loss', self.critic_loss), ])
    self.merged_summary_direction = tf.summary.merge(
      [tf.summary.scalar('direction_loss', self.direction_loss), ])

  """Build the branch for constructing the successor representation latent space"""

  def build_SF_net(self, layer_norm=False):
    with tf.variable_scope("succ_feat"):
      out = layers.fully_connected(self.observation,
                                   num_outputs=self.nb_states,
                                   activation_fn=None,
                                   biases_initializer=None,
                                   variables_collections=tf.get_collection("variables"),
                                   outputs_collections="activations", scope="sf")
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
    self.target_mix_return = tf.placeholder(shape=[None], dtype=tf.float32)
    """Placeholder for the target return for n-step prediction of the option critics"""
    self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
    """Placeholder indicating wheather the current option is a primitive action or not"""
    self.primitive_actions_placeholder = tf.placeholder(shape=[None], dtype=tf.bool,
                                                        name="primitive_actions_placeholder")

    self.target_fi_horiz = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32, name="target_fi_horiz")


  def init_clustering(self):
    if self.scope == 'global':
      l = "0"
      if self.config.resume:
        checkpoint = self.config.load_from
        ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint, "models"))
        model_checkpoint_path = ckpt.model_checkpoint_path
        episode_checkpoint = os.path.basename(model_checkpoint_path).split(".")[0].split("-")[1]
        l = episode_checkpoint

      self.direction_clusters_path = os.path.join(self.config.logdir, "direction_clusters_{}.npy".format(l))

      """If the path exists, load them. Otherwise initialize all directions with zeros"""
      if os.path.exists(self.direction_clusters_path):
        self.direction_clusters = np.load(self.direction_clusters_path)
        self.directions_init = True
      else:
        self.direction_clusters = OnlineCluster(self.config.nb_options, self.goal_embedding_size)#np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
        self.directions_init = False

  def build_network(self):
    with tf.variable_scope(self.scope):
      # self.observation = tf.placeholder(
      #   shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
      #   dtype=tf.float32, name="Inputs")
      # out = self.observation
      # out = layers.flatten(out, scope="flatten")
      self.observation = tf.placeholder(
        shape=[None, self.nb_states],
        dtype=tf.float32, name="Inputs")

      self.direction_clusters = tf.placeholder(shape=[self.config.nb_options, self.goal_embedding_size],
                                               dtype=tf.float32,
                                               name="direction_clusters")
      """Build the encoder for the latent representation space"""
      # self.build_feature_net(out)
      self.fi = self.fi_relu = self.observation
      """Build the branch for next frame prediction that trains the latent state representation"""
      # self.build_next_frame_prediction_net()

      """Build the option action-value functions"""
      self.build_option_q_val_net()

      """If we use eigendirections we need another critic for the intra-option policies that uses the mixture of pseudo internal rewards and external rewards"""
      if self.config.use_clustering:
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