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
class AttentionWTermNetwork(EignOCNetwork):
  def __init__(self, scope, config, action_size):
    self.goal_embedding_size = config.sf_layers[-1]
    super(AttentionWTermNetwork, self).__init__(scope, config, action_size)
    if self.config.use_clustering:
      self.init_clustering()

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.target_sf = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32, name="target_SF")
      self.target_direction = tf.placeholder(shape=[None, self.goal_embedding_size], dtype=tf.float32, name="target_direction")

      self.target_mix_return = tf.placeholder(shape=[None], dtype=tf.float32, name="target_mix_return")
      self.target_v_ext = tf.placeholder(shape=[None], dtype=tf.float32, name="target_v_ext")
      self.target_return = tf.placeholder(shape=[None], dtype=tf.float32, name="target_return")
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="actions_placeholder")
      self.observation = tf.placeholder(
        shape=[None, self.nb_states],
        dtype=tf.float32, name="observation_placeholder")
      self.observation_image = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], 1],
        dtype=tf.float32, name="observation_image_placeholder")

      self.image_summaries.append(
        tf.summary.image('observation', self.observation_image, max_outputs=30))

      with tf.variable_scope("succ_feat"):
        self.sf = layers.fully_connected(self.observation,
                                     num_outputs=self.goal_embedding_size,
                                     activation_fn=None,
                                     biases_initializer=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="sf")
      direction_clusters = tf.placeholder(shape=[self.config.nb_options,
                                                      self.goal_embedding_size],
                                               dtype=tf.float32,
                                               name="direction_clusters")
      self.direction_clusters = tf.nn.l2_normalize(direction_clusters, 1)

      with tf.variable_scope("option_features"):
        intrinsic_features = layers.fully_connected(self.observation,
                                                num_outputs=self.action_size * self.goal_embedding_size,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="intrinsic_features")
        policy_features = tf.reshape(intrinsic_features, [-1, self.action_size,
                                                           self.goal_embedding_size],
                                          name="policy_features")
        value_features = tf.identity(intrinsic_features, name="value_features")

      with tf.variable_scope("option_policy"):
        direction_features = layers.fully_connected(self.observation,
                                                    num_outputs=self.goal_embedding_size,
                                                    activation_fn=None,
                                                    variables_collections=tf.get_collection("variables"),
                                                    outputs_collections="activations", scope="direction_features")
        self.query_direction = self.l2_normalize(direction_features, 1)

        self.query_content_match = tf.einsum('bj, ij -> bi', self.query_direction, self.direction_clusters, name="query_content_match")
        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.query_content_match))

        self.attention_weights = tf.nn.softmax(self.query_content_match, name="attention_weights")
        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.attention_weights))

        self.current_unnormalized_direction = tf.einsum('bi, ij -> bj', self.attention_weights, self.direction_clusters, name="unnormalized_current_option_direction")

        self.current_option_direction = tf.identity(self.l2_normalize(self.current_unnormalized_direction, 1), name="current_option_direction")
        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.current_option_direction))

      # self.target_current_option_direction = tf.placeholder_with_default(self.current_option_direction,
      #                                                                    shape=[None, self.goal_embedding_size], name="target_crt_option_direction")

      with tf.variable_scope("option_value_ext"):
        extrinsic_features = layers.fully_connected(self.observation,
                                               num_outputs=self.goal_embedding_size,
                                               activation_fn=None,
                                               variables_collections=tf.get_collection("variables"),
                                               outputs_collections="activations",
                                               scope="extrinsic_features")
        # value_ext = layers.fully_connected(extrinsic_features,
        #                                         num_outputs=1,
        #                                         activation_fn=None,
        #                                         variables_collections=tf.get_collection("variables"),
        #                                         outputs_collections="activations", scope="value_ext")
        # self.value_ext = tf.squeeze(value_ext, 1)
        # adv_ext = layers.fully_connected(tf.concat([extrinsic_features,
        #                                       tf.stop_gradient(self.target_current_option_direction)], 1),
        #                                         num_outputs=1,
        #                                         activation_fn=None,
        #                                         variables_collections=tf.get_collection("variables"),
        #                                         outputs_collections="activations", scope="adv_ext")
        # self.adv_ext = tf.squeeze(adv_ext, 1)
        # self.q_ext = self.value_ext + self.adv_ext
        q_ext_embedding = tf.get_variable("q_ext_embedding",
                                          shape=[
                                            2 * self.goal_embedding_size,
                                            1],
                                          initializer=normalized_columns_initializer(1.0))
        q_ext = tf.matmul(tf.concat([extrinsic_features, self.current_option_direction], 1), q_ext_embedding, name="q_ext")

        self.q_ext = tf.squeeze(q_ext, 1)
				#
        # value_extrinsic_features = tf.tile(tf.expand_dims(extrinsic_features, 1), [1, self.direction_clusters.shape[0].value, 1])
        # tiled_direction_clusters_ext = tf.tile(self.direction_clusters[None, ...], [tf.shape(extrinsic_features)[0], 1, 1])
				#
        # value_extrinsic_features_concat = tf.concat([value_extrinsic_features, tiled_direction_clusters_ext], 2)
        # q_ext_clusters = tf.squeeze(tf.einsum('bij, jk -> bik', value_extrinsic_features_concat, q_ext_embedding, name="q_ext_clusters"), -1)
        # # q_ext_clusters = tf.matmul(value_extrinsic_features_concat, q_ext_embedding,
        # #                   name="q_ext_clusters")
				#
        # self.v_ext = tf.reduce_sum(self.attention_weights * q_ext_clusters, 1)
        # q_ext = layers.fully_connected(tf.concat([extrinsic_features,
        #                                           tf.stop_gradient(self.target_current_option_direction)], 1),
        #                                num_outputs=1,
        #                                activation_fn=None,
        #                                variables_collections=tf.get_collection("variables"),
        #                                outputs_collections="activations", scope="q_ext")

      with tf.variable_scope("option_term"):
        term_features = layers.fully_connected(self.observation,
                                               num_outputs=self.goal_embedding_size,
                                               activation_fn=None,
                                               variables_collections=tf.get_collection("variables"),
                                               outputs_collections="activations", scope="term_feat")
        termination = layers.fully_connected(tf.concat([term_features,
                                                        self.current_option_direction], 1),
                                             num_outputs=1,
                                             activation_fn=tf.nn.sigmoid,
                                             variables_collections=tf.get_collection("variables"),
                                             outputs_collections="activations", scope="termination")
        self.termination = tf.squeeze(termination, 1, name="termination")
        self.summaries_term.append(tf.contrib.layers.summarize_activation(self.termination))

      with tf.variable_scope("option_value_int"):
        q_mix_embedding = tf.get_variable("q_mix_embedding",
                                          shape=[
                                            self.action_size * self.goal_embedding_size + self.goal_embedding_size,
                                            1],
                                          initializer=normalized_columns_initializer(1.0))
        q_mix = tf.matmul(tf.concat([value_features,
                                              self.current_option_direction], 1),q_mix_embedding,
                                   name="fc_option_value")
        self.q_mix = tf.squeeze(q_mix, 1)
				#
				#
        # value_mixed_features = tf.tile(tf.expand_dims(value_features, 1),
        #                                    [1, self.direction_clusters.shape[0].value, 1])
        # tiled_direction_clusters_mix = tf.tile(self.direction_clusters[None, ...],
        #                                        [tf.shape(value_features)[0], 1, 1])
				#
        # value_mixed_features_concat = tf.concat([value_mixed_features, tiled_direction_clusters_mix], 2)
				#
        # q_mix_clusters = tf.squeeze(
        #   tf.einsum('bij, jk -> bik', value_mixed_features_concat, q_mix_embedding,
        #                            name="q_mix_clusters"), -1)
        # self.v_mix = tf.reduce_sum(self.attention_weights * q_mix_clusters, 1)

      with tf.variable_scope("option_pi"):
        policy = tf.einsum('bj,bij->bi', self.current_option_direction, policy_features)
        self.option_policy = tf.nn.softmax(policy, name="policy")

        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.option_policy))


      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  def build_losses(self):
    """Get the probabilities for each action taken under the intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.option_policy, self.actions_placeholder)

    """Building losses"""
    with tf.name_scope('sf_loss'):
      """TD error of successor representations"""
      sf_td_error = self.target_sf - self.sf
      self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    with tf.name_scope('mix_critic_loss'):
      mix_td_error = self.target_mix_return - self.q_mix
      self.mix_critic_loss = tf.reduce_mean(0.5 * tf.square(mix_td_error))

    with tf.name_scope('direction_critic_loss'):
      td_error = self.target_return - self.q_ext
      self.critic_loss = tf.reduce_mean(0.5 * tf.square(td_error))

    with tf.name_scope('termination_loss'):
      self.term_loss = tf.reduce_mean(self.termination * tf.stop_gradient(self.q_ext - self.target_v_ext))

    with tf.name_scope('direction_loss'):
      self.direction_loss = tf.reduce_mean(
        self.cosine_similarity(self.target_direction, self.current_option_direction, 1) * self.target_return)
        # tf.stop_gradient(td_error))

    """Add an entropy regularization for each intra-option policy, driving exploration in the action space of intra-option policies"""
    with tf.name_scope('entropy_loss'):
      """Zero out primitive options"""
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(self.option_policy * tf.log(self.option_policy + 1e-7))
    """Learn intra-option policies with policy gradients"""
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
                          mix_td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.mix_critic_loss

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
    self.grads_sf, self.apply_grads_sf = self.take_gradient(self.sf_loss)
    self.grads_option, self.apply_grads_option = self.take_gradient(self.option_loss)
    self.grads_critic, self.apply_grads_critic = self.take_gradient(self.critic_loss)
    self.grads_term, self.apply_grads_term = self.take_gradient(self.term_loss)
    self.grads_direction, self.apply_grads_direction = self.take_gradient(self.direction_loss)

    """Summaries"""
    self.merged_summary_sf = tf.summary.merge(self.image_summaries +
      self.summaries_sf + [tf.summary.scalar('SF_loss', self.sf_loss),
        gradient_summaries(zip(self.grads_sf, local_vars))])

    self.merged_summary_option = tf.summary.merge(self.summaries_option +\
                       [tf.summary.scalar('Entropy_loss', self.entropy_loss),
                        tf.summary.scalar('Policy_loss', self.policy_loss),
                        tf.summary.scalar('Mix_critic_loss', self.mix_critic_loss), ])
    self.merged_summary_critic = tf.summary.merge(self.summaries_critic +\
                                                  [tf.summary.scalar('Critic_loss', self.critic_loss)])
    self.merged_summary_term = tf.summary.merge(self.summaries_term + \
                                                  [tf.summary.scalar('Term_loss', self.term_loss)])
    self.merged_summary_direction = tf.summary.merge(
                                                [tf.summary.scalar('Direction_loss', self.direction_loss)])

  def init_clustering(self):
    if self.scope == 'global':
      l = "0"
      if self.config.resume:
        checkpoint = self.config.load_from
        ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint, "models"))
        model_checkpoint_path = ckpt.model_checkpoint_path
        episode_checkpoint = os.path.basename(model_checkpoint_path).split(".")[0].split("-")[1]
        l = episode_checkpoint

      cluster_model_path = os.path.join(self.config.logdir, "cluster_models")
      self.direction_clusters_path = os.path.join(cluster_model_path, "direction_clusters_{}.pkl".format(l))

      """If the path exists, load them. Otherwise initialize all directions with zeros"""
      if os.path.exists(self.direction_clusters_path):
        self.direction_clusters = np.load(self.direction_clusters_path)
        self.directions_init = True
      else:
        self.direction_clusters = OnlineCluster(self.config.max_clusters, self.config.nb_options, self.goal_embedding_size)#np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
        self.directions_init = False