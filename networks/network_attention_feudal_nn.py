import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_eigenoc import EignOCNetwork
import os
from online_clustering import OnlineCluster
import tensorflow_probability as tfp
from auxilary.lstm_model import SingleStepLSTM

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

class AttentionFeudalNNNetwork(EignOCNetwork):
  def __init__(self, scope, config, action_size):
    self.goal_embedding_size = config.sf_layers[-1]
    self.image_summaries_goal = []
    self.network_optimizer = config.network_optimizer(
      config.lr, name='network_optimizer')

    super(AttentionFeudalNNNetwork, self).__init__(scope, config, action_size)

    if self.config.use_clustering:
      self.init_clustering()

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.prob_of_random_goal = tf.Variable(self.config.initial_random_goal_prob, trainable=False, name="prob_of_random_goal", dtype=tf.float32)

      self.target_sf = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32, name="target_SF")
      self.target_goal = tf.placeholder(shape=[None, self.goal_embedding_size], dtype=tf.float32, name="target_goal")
      self.prev_goals = tf.placeholder(shape=[None, None, self.goal_embedding_size], dtype=tf.float32, name="prev_goals")
      self.target_mix_return = tf.placeholder(shape=[None], dtype=tf.float32, name="target_mix_return")
      self.target_return = tf.placeholder(shape=[None], dtype=tf.float32, name="target_return")
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="actions_placeholder")
      self.target_next_obs = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size], dtype=tf.float32,
        name="target_next_obs")
      self.prev_rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="Prev_Rewards")
      self.prev_rewards_expanded = tf.expand_dims(self.prev_rewards, 1)

      self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
      self.prev_actions_onehot = tf.one_hot(self.prev_actions, self.action_size, dtype=tf.float32, name="Prev_Actions_OneHot")

      self.observation = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
        dtype=tf.float32, name="Inputs")
      self.input = layers.flatten(self.observation, scope="flatten")

      actions = layers.fully_connected(tf.one_hot(self.actions_placeholder, depth=self.action_size), num_outputs=self.config.fc_layers[-1],
                                       activation_fn=None,
                                       scope="action_fc")

      with tf.variable_scope("fi"):
        out = layers.fully_connected(self.input, num_outputs=self.config.fc_layers[-1],
                                     activation_fn=None,
                                    scope="fi")
        self.fi = out
        self.fi_relu = tf.nn.relu(out)


      with tf.variable_scope("aux_next_frame"):
        out = tf.add(self.fi, actions)
        out = tf.nn.relu(out)
        out = layers.fully_connected(out, num_outputs=self.config.aux_fc_layers[-1],
                                     activation_fn=None,
                                     scope="aux")
        self.next_obs = tf.reshape(out,
                                   (-1, self.config.input_size[0], self.config.input_size[1], self.config.history_size))

      goal_clusters = tf.placeholder(shape=[None, self.config.nb_options,
                                                 self.goal_embedding_size],
                                          dtype=tf.float32,
                                          name="goal_clusters")
      self.goal_sr_clusters = tf.nn.l2_normalize(goal_clusters, 2)


      self.image_summaries.append(
        tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

      with tf.variable_scope("succ_feat"):
        self.sf = layers.fully_connected(tf.stop_gradient(self.fi_relu),
                                     num_outputs=self.goal_embedding_size,
                                     activation_fn=None,
                                     biases_initializer=None,
                                     scope="sf")

      with tf.variable_scope("option_manager_policy"):
        """The merged representation of the input"""
        goal_hat = layers.fully_connected(tf.stop_gradient(self.fi_relu),
                                                    num_outputs=self.goal_embedding_size,
                                                    activation_fn=None,
                                                    scope="goal_hat")
        self.query_goal = self.l2_normalize(goal_hat, 1)
        self.query_content_match = tf.einsum('bj, bij -> bi', self.query_goal, self.goal_sr_clusters, name="query_content_match")
        self.query_content_match_sharp = self.query_content_match * self.config.starpening_factor
        self.goal_distribution = tfp.distributions.RelaxedOneHotCategorical(self.config.temperature,
                                                                                   logits=self.query_content_match_sharp)
        self.attention_weights = self.goal_distribution.sample()
        self.which_goal = tf.argmax(self.attention_weights, 1)
        self.current_unnormalized_goal = tf.einsum('bi, bij -> bj', self.attention_weights, self.goal_sr_clusters, name="unnormalized_g")
        self.max_g = tf.identity(self.l2_normalize(self.current_unnormalized_goal, 1), name="g")

        self.local_random = tf.random_uniform(shape=[tf.shape(self.max_g)[0]], minval=0., maxval=1., dtype=tf.float32, name="rand_goals")
        random_goal_sampling = tf.distributions.Categorical(probs=[1/(self.config.nb_options + 1) for _ in range(self.config.nb_options + 1)])
        self.which_random_goal = random_goal_sampling.sample(tf.shape(self.max_g)[0])
        # self.random_g = tf.gather(self.goal_sr_clusters, self.which_random_goal, axis=1)
        indices_random_goal = tf.stack([tf.range(tf.shape(self.which_random_goal)[0]), self.which_random_goal], axis=1)
        self.goal_sr_clusters_plus = tf.map_fn(lambda goals: tf.concat([goals, tf.random_normal(shape=(1, self.goal_embedding_size))], 0), self.goal_sr_clusters)
        self.random_g = tf.gather_nd(self.goal_sr_clusters_plus, indices_random_goal)
        self.random_goal_cond = self.local_random > self.prob_of_random_goal
        self.g = tf.where(self.random_goal_cond, self.max_g, self.random_g, name="current_goal")
        # self.g = self.max_g
        cut_g = tf.stop_gradient(self.g)
        cut_g = tf.expand_dims(cut_g, 1)
        self.g_stack = tf.placeholder_with_default(shape=[None, None, self.goal_embedding_size],
                                                   input=tf.concat([self.prev_goals, cut_g], 1))
        self.last_c_g = self.g_stack[:, 1:]
        self.g_sum = tf.reduce_sum(self.g_stack, 1)

      with tf.variable_scope("option_manager_value_ext"):
        phi = tf.get_variable("phi", (self.goal_embedding_size, self.config.goal_projected_size),
                              initializer=normalized_columns_initializer(1.))
        self.goal_projected = tf.matmul(self.g_sum, phi)
        extrinsic_features = layers.fully_connected(tf.stop_gradient(self.fi_relu),
                                               num_outputs=self.goal_embedding_size,
                                               activation_fn=tf.nn.relu,
                                               scope="extrinsic_features")
        v_ext = layers.fully_connected(extrinsic_features,
                                               num_outputs=1,
                                               activation_fn=None,
                                               scope="v_ext")
        self.v_ext = tf.squeeze(v_ext, 1)

      with tf.variable_scope("option_worker_features"):
        intrinsic_features = layers.fully_connected(tf.stop_gradient(self.fi_relu),
                                                num_outputs=self.action_size * self.config.goal_projected_size,
                                                activation_fn=tf.nn.relu,
                                                scope="intrinsic_features")
        policy_features = tf.reshape(intrinsic_features, [-1, self.action_size,
                                                           self.config.goal_projected_size],
                                          name="policy_features")
        value_features = tf.identity(intrinsic_features, name="value_features")

      with tf.variable_scope("option_worker_value_mix"):
        v_mix = layers.fully_connected(tf.concat([value_features, self.goal_projected], 1),
                                              num_outputs=1,
                                              activation_fn=None,
                                              scope="v_mix")
        self.v_mix = tf.squeeze(v_mix, 1)

      with tf.variable_scope("option_worker_pi"):
        policy = tf.einsum('bj,bij->bi', self.goal_projected, policy_features)
        self.g_policy = tf.nn.softmax(policy, name="policy")

        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.g_policy))

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  def build_losses(self):
    """Get the probabilities for each action taken under the intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.g_policy, self.actions_placeholder)

    """Building losses"""
    with tf.name_scope('sf_loss'):
      sf_td_error = self.target_sf - self.sf
      self.sf_loss = tf.reduce_mean(self.config.sf_coef * tf.square(sf_td_error))

    with tf.name_scope('aux_loss'):
      """L2 loss for the next frame prediction"""
      aux_error = self.next_obs - self.target_next_obs
      self.aux_loss = tf.reduce_mean(self.config.aux_coef * tf.square(aux_error))

    with tf.name_scope('mix_critic_loss'):
      mix_td_error = self.target_mix_return - self.v_mix
      self.mix_critic_loss = tf.reduce_mean(0.5 * tf.square(mix_td_error))

    with tf.name_scope('goal_critic_loss'):
      td_error = self.target_return - self.v_ext
      self.critic_loss = tf.reduce_mean(0.5 * tf.square(td_error))

    with tf.name_scope('goal_loss'):
      cosine = self.cosine_similarity(self.target_goal, self.g, 1)
      self.cosine_sim = tf.reduce_mean(cosine)
      self.goal_loss = -tf.reduce_mean(cosine * tf.stop_gradient(td_error))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(self.g_policy * tf.log(self.g_policy + 1e-7))

    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(mix_td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.mix_critic_loss

  def l2_normalize(self, x, axis):
    x += 1e-12
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True))
    return x / tf.maximum(norm, 1e-12)
    # return tf.maximum(x, 1e-8) / tf.maximum(norm, 1e-8)
		# return x / tf.maximum(norm, 1e-8)

  def cosine_similarity(self, v1, v2, axis):
    # norm_v1 = tf.nn.l2_normalize(tf.cast(v1, tf.float64), axis)
    # norm_v2 = tf.nn.l2_normalize(tf.cast(v2, tf.float64), axis)
    norm_v1 = tf.nn.l2_normalize(tf.cast(v1 + 1e-12, tf.float64), axis)
    norm_v2 = tf.nn.l2_normalize(tf.cast(v2 + 1e-12, tf.float64), axis)
    # norm_v1 = self.l2_normalize(tf.cast(v1, tf.float64), axis)
    # norm_v2 = self.l2_normalize(tf.cast(v2, tf.float64), axis)
    sim = tf.matmul(
      norm_v1, norm_v2, transpose_b=True)
    return tf.cast(sim, tf.float32)

  def take_gradient(self, loss):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    gradients = tf.gradients(loss, local_vars)
    var_norms = tf.global_norm(local_vars)
    grads, grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm_value)
    apply_grads = self.network_optimizer.apply_gradients(zip(grads, global_vars))
    return grads, apply_grads

  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    self.grads_sf, self.apply_grads_sf = self.take_gradient(self.sf_loss)
    self.grads_aux, self.apply_grads_aux = self.take_gradient(self.aux_loss)
    self.grads_option, self.apply_grads_option = self.take_gradient(self.option_loss)
    self.grads_critic, self.apply_grads_critic = self.take_gradient(self.critic_loss)
    self.grads_goal, self.apply_grads_goal = self.take_gradient(self.goal_loss)

    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('SF_loss', self.sf_loss),
        gradient_summaries(zip(self.grads_sf, local_vars))])

    self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss),
                                                gradient_summaries(zip(self.grads_aux, local_vars))])

    self.merged_summary_option = tf.summary.merge(self.summaries_option +\
                       [tf.summary.scalar('Entropy_loss', self.entropy_loss),
                        tf.summary.scalar('Policy_loss', self.policy_loss),
                        tf.summary.scalar('Mix_critic_loss', self.mix_critic_loss),
                        gradient_summaries(zip(self.grads_option, local_vars))])
    self.merged_summary_critic = tf.summary.merge(self.summaries_critic +\
                                                  [tf.summary.scalar('Critic_loss', self.critic_loss),
                                                   gradient_summaries(zip(self.grads_critic, local_vars))])
    self.merged_summary_goal = tf.summary.merge(self.image_summaries_goal +
                                                [tf.summary.scalar('cosine_product', self.cosine_sim),
                                                  tf.summary.scalar('goal_loss', self.goal_loss),
                                                 gradient_summaries(zip(self.grads_goal, local_vars))])

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
      self.goal_clusters_path = os.path.join(cluster_model_path, "goal_clusters_{}.pkl".format(l))

      """If the path exists, load them. Otherwise initialize all goals with zeros"""
      if os.path.exists(self.goal_clusters_path):
        self.goal_clusters = np.load(self.goal_clusters_path)
        self.goals_init = True
      else:
        self.goal_clusters = OnlineCluster(self.config.max_clusters, self.config.nb_options, self.goal_embedding_size)#np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
        self.goals_init = False