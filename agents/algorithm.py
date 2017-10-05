
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections

from agents import memory
from agents import utility
import numpy as np

from agents.schedules import LinearSchedule, TFLinearSchedule
from agents.optimizers import huber_loss


_NetworkOutput = collections.namedtuple(
    'NetworkOutput', 'termination, q_val, options, state')

class AOCAlgorithm(object):

  def __init__(self, batch_env, step, is_training, should_log, config, sess):
    self._batch_env = batch_env
    self._step = step
    self._is_training = is_training
    self._should_log = should_log
    self.sess = sess

    self._config = config
    self._nb_options = config.nb_options
    self._action_size = self._batch_env._batch_env._envs[0]._action_space.n
    self._num_agents = self._config.num_agents

    self._exploration_options = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
                                               self._config.initial_random_action_prob)
    # self._exploration_policies = TFLinearSchedule(self._config.explore_steps, self._config.final_random_action_prob,
    #                                             self._config.initial_random_action_prob)

    self._memory_index = tf.Variable(0, False)
    use_gpu = self._config.use_gpu and utility.available_gpus()
    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
      self._network(
        tf.zeros_like(self._batch_env.observ)[:, None],
        tf.ones(len(self._batch_env)), reuse=None)
      cell = self._config.network(self._action_size)

      self._option_terminated = tf.Variable(
        np.zeros([self._num_agents], dtype=np.bool), False, name='option_terminated', dtype=tf.bool)
      self._frame_counter = tf.Variable(
        np.zeros([self._num_agents], dtype=np.int32), False, name='frame_counter', dtype=tf.int32)
      self._current_option = tf.Variable(
        np.zeros([self._num_agents], dtype=np.int32), False, name='current_option')
      self._random = tf.random_uniform(shape=[(self._config.num_agents)], minval=0., maxval=1., dtype=tf.float32)

      self._delib_cost = tf.Variable(np.asarray(self._num_agents * [self._config.delib_cost]),
                                      False, name="delib_cost", dtype=tf.float32)

      self._network_optimizer = self._config.network_optimizer(
        self._config.lr, name='network_optimizer')

    template = (
      self._batch_env.observ[0], self._batch_env.action[0], self._batch_env.action[0],
      self._batch_env.reward[0], tf.cast(self._option_terminated[0], tf.int32), self._batch_env.observ[0],
      tf.cast(self._option_terminated[0], tf.int32))
    self._memory = memory.EpisodeMemory(
      template, config.update_every, config.max_length, 'memory')

    with tf.variable_scope('aoc_temporary'):
      self._episodes = memory.EpisodeMemory(
        template, len(batch_env), config.max_length, 'episodes')
      self._last_state = utility.create_nested_vars(
        cell.zero_state(len(batch_env), tf.float32))
      self._last_action = tf.Variable(
        tf.zeros_like(self._batch_env.action), False, name='last_action')
      self._last_option = tf.Variable(
        tf.zeros_like(self._batch_env.action), False, name='last_option')
      self._last_option_terminated = tf.Variable(
        tf.zeros_like(self._batch_env.action), False, name='option_terminated')

  def begin_episode(self, agent_indices):
    with tf.name_scope('begin_episode/'):
      reset_state = utility.reinit_nested_vars(self._last_state, agent_indices)
      reset_buffer = self._episodes.clear(agent_indices)
      variables = [self._frame_counter, self._current_option]
      reset_internal_vars = self.reinit_vars(variables)
      with tf.control_dependencies([reset_state, reset_buffer, reset_internal_vars]):
        return tf.constant('')

  def reinit_vars(self, variables):
    if isinstance(variables, (tuple, list)):
      return tf.group(*[
        self.reinit_vars(variable) for variable in variables])
    return tf.group(variables.assign(tf.zeros_like(variables)),
      self._option_terminated.assign(np.asarray(self._num_agents * [True])),
      self._delib_cost.assign(np.asarray(self._num_agents * [self._config.delib_cost])))

  def perform(self, observ):
    with tf.name_scope('perform/'):
      # observ = self._observ_filter.transform(observ)
      network = self._network(
        observ[:, None], tf.ones(observ.shape[0]), self._last_state)

      next_options = self.get_policy_over_options(network)
      self._current_option = tf.where(self._option_terminated, next_options, self._current_option)
      # Choose at according to πθ (·|st)
      action = self.get_action(network)

      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
        tf.summary.histogram('current_action', action),
        tf.summary.histogram('current_option', self._current_option),
        tf.summary.histogram('last_option_terminated', tf.cast(self._option_terminated, tf.int32))]), str)

      increment_frame_counter = self._frame_counter.assign_add(tf.ones(self._num_agents, tf.int32))

      # update internal state
      with tf.control_dependencies([
          utility.assign_nested_vars(self._last_state, network.state),
          self._last_action.assign(tf.cast(action, tf.int32)),
          self._last_option.assign(self._current_option),
          self._last_option_terminated.assign(tf.cast(self._option_terminated, tf.int32)),
          increment_frame_counter,
      ]):

        return tf.cast(action, dtype=tf.int32), self._current_option, self._option_terminated, tf.identity(summary)

  def experience(self, observ, option, action, reward, done, nextob, option_terminated):
    with tf.name_scope('experience/'):
      return tf.cond(
        self._is_training,
        lambda: self._define_experience(observ, option, action, reward, done, nextob, option_terminated),
        lambda: (tf.ones_like(option_terminated, dtype=tf.bool), str()))

  def _define_experience(self, observ, option, action, reward, done, nextob, option_terminated):
    # if the current option ot terminates in st+1 then
    #     choose new ot+1 with -soft(µ(st+1)) => next_time
    network = self._network(
      observ[:, None], tf.ones(observ.shape[0]), self._last_state)
    network_next_obs = self._network(
      nextob[:, None], tf.ones(nextob.shape[0]), self._last_state)
    self._option_terminated = self.get_termination(network_next_obs)

    # Take action at in st, observe rt, st+1
    # new_rt ← rt + ct
    new_reward = reward - \
                 tf.cast(option_terminated, tf.float32) * self._delib_cost #* tf.cast(self._frame_counter > 1, dtype=tf.float32)

    batch = observ, option, action, new_reward, tf.cast(done, tf.int32), nextob, tf.cast(self._option_terminated, tf.int32)
    append = self._episodes.append(batch, tf.range(len(self._batch_env)))
    with tf.control_dependencies([append]):
      # pylint: disable=g-long-lambda
      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
        tf.summary.scalar('memory_size', self._memory_index),
        # tf.summary.image(observ),
        tf.summary.histogram('observ', observ),
        tf.summary.histogram('action', action),
        tf.summary.histogram('option', option),
        tf.summary.histogram('option_terminated', tf.cast(self._option_terminated, dtype=tf.int32)),
        tf.summary.scalar('new_reward', tf.reduce_mean(reward))]), str)
      return self._option_terminated, summary

  def end_episode(self, agent_indices):
    with tf.name_scope('end_episode/'):
      return tf.cond(
          self._is_training,
          lambda: self._define_end_episode(agent_indices), str)

  def _define_end_episode(self, agent_indices):
    episodes, length = self._episodes.data(agent_indices)
    space_left = self._config.update_every - self._memory_index
    use_episodes = tf.range(tf.minimum(
        tf.shape(agent_indices)[0], space_left))
    episodes = [tf.gather(elem, use_episodes) for elem in episodes]
    append = self._memory.replace(
        episodes, tf.gather(length, use_episodes),
        use_episodes + self._memory_index)
    with tf.control_dependencies([append]):
      inc_index = self._memory_index.assign_add(tf.shape(use_episodes)[0])
    with tf.control_dependencies([inc_index]):
      memory_full = self._memory_index >= self._config.update_every
      return tf.cond(memory_full, self._training, str)

  def _network(self, observ, length=None, state=None, reuse=True):
    with tf.variable_scope('network', reuse=reuse):
      observ = tf.convert_to_tensor(observ)
      use_gpu = self._config.use_gpu and utility.available_gpus()
      with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
        observ = tf.check_numerics(observ, 'observ')
        cell = self._config.network(self._batch_env._batch_env._envs[0]._action_space.n)
        (termination, q_val, options), state = tf.nn.dynamic_rnn(
            cell, observ, length, state, tf.float32, swap_memory=True)

      return _NetworkOutput(termination, q_val, options, state)

  def get_policy_over_options(self, network):
    self.probability_of_random_option = self._exploration_options.value(self._step)
    max_options = tf.cast(tf.argmax(network.q_val[:, 0,:], 1), dtype=tf.int32)
    exp_options = tf.random_uniform(shape=[self._num_agents], minval=0, maxval=self._config.nb_options,
                              dtype=tf.int32)
    options = tf.where(self._random > self.probability_of_random_option, max_options, exp_options)
    return options

  def get_action(self, network):
    current_option_option_one_hot = tf.one_hot(self._current_option, self._nb_options, name="options_one_hot")
    current_option_option_one_hot = current_option_option_one_hot[:, :, None]
    current_option_option_one_hot = tf.tile(current_option_option_one_hot, [1, 1, self._action_size])
    self.action_probabilities = tf.reduce_sum(tf.multiply(network.options[:, 0,:], current_option_option_one_hot),
                                      reduction_indices=1, name="P_a")
    policy = tf.multinomial(tf.log(self.action_probabilities), 1)[:, 0]
    return policy

  def get_termination(self, network):
    current_option_option_one_hot = tf.one_hot(self._current_option, self._nb_options, name="options_one_hot")
    termination_probabilities = tf.reduce_sum(tf.multiply(network.termination[:, 0,:], current_option_option_one_hot),
                  reduction_indices=1, name="P_term")
    terminated = termination_probabilities > self._random
    return terminated

  def _training(self):
    """Perform one training iterations of both policy and value baseline.

    Training on the episodes collected in the memory. Reset the memory
    afterwards. Always returns a summary string.

    Returns:
      Summary tensor.
    """
    with tf.name_scope('training'):
      assert_full = tf.assert_equal(
          self._memory_index, self._config.update_every)
      with tf.control_dependencies([assert_full]):
        data = self._memory.data()
      (observ, option, action, reward, done, nextob, option_terminated), length = data
      with tf.control_dependencies([tf.assert_greater(length, 0)]):
        length = tf.identity(length)

      network_summary = self._update_network(
        observ, option, action, reward, tf.cast(done, tf.bool), nextob, tf.cast(option_terminated, tf.bool), length)

      with tf.control_dependencies([network_summary]):
        clear_memory = tf.group(
            self._memory.clear(), self._memory_index.assign(0))
      with tf.control_dependencies([clear_memory]):
        weight_summary = utility.variable_summaries(
            tf.trainable_variables(), self._config.weight_summaries)
        return tf.summary.merge([network_summary, weight_summary])

  def _update_network(self, observ, option, action, reward, done, nextob, option_terminated, length):
    self.probability_of_random_option = self._exploration_options.value(self._step)
    with tf.name_scope('update_network'):
      # add delib if  option termination because it isn't part of V
      delib = self._delib_cost #* tf.cast(self._frame_counter > 1, dtype=np.float32)
      network = self._network(observ, length)
      network_next = self._network(nextob, length)
      # network_next = self._network(nextob, length)
      # raw_v = tf.reduce_sum(tf.multiply(self.get_V(network), tf.one_hot(length, self._config.max_length)), axis=1)
      v = self.get_V(network_next) - tf.tile(delib[:, None], [1, self._config.max_length])
      # q = tf.reduce_sum(tf.multiply(self.get_Q(network, option), tf.one_hot(length, self._config.max_length)), axis=1)
      q = self.get_Q(network_next, option)
      new_v = tf.where(option_terminated, v, q)
      G = tf.cast(tf.logical_not(done), dtype=tf.float32) * new_v

      real_length = utility.get_length_option(option_terminated, length)
      timestep = tf.tile(tf.range(reward.shape[1].value)[None, ...], [reward.shape[0].value, 1])
      mask = tf.cast(timestep < real_length[:, None], tf.float32)

      G = utility.discounted_return_n_step(reward, real_length, self._config.discount, G)

      # mean, variance = tf.nn.moments(advantage, axes=[0, 1], keep_dims=True)
      # advantage = (advantage - mean) / (tf.sqrt(variance) + 1e-8)

      G = tf.Print(G, [tf.reduce_mean(G)], 'Return G: ')

      # q_opt = self.get_Q(network, option)
      # v = self.get_V(network)
      intra_option_policy = self.get_intra_option_policy(network, option)
      responsible_outputs = self.get_responsible_outputs(intra_option_policy, action)
      o_termination = self.get_option_termination(network_next, option)

      with tf.name_scope('critic_loss'):
        td_error = tf.stop_gradient(G) - q
        critic_loss = tf.reduce_mean(self._mask(self._config.critic_coef * 0.5 * tf.square((td_error)), real_length))
      with tf.name_scope('termination_loss'):
        term_loss = tf.reduce_mean(mask * o_termination * (tf.stop_gradient(q) - tf.stop_gradient(v) +
                                                          tf.tile(delib[:, None],
                                                                  [1, self._config.max_length])))
      with tf.name_scope('entropy_loss'):
        entropy_loss = self._config.entropy_coef * tf.reduce_mean(mask * tf.reduce_sum(intra_option_policy *
                                                                                tf.log(intra_option_policy +
                                                                                       1e-7), axis=2))
      with tf.name_scope('policy_loss'):
        policy_loss = -tf.reduce_sum(mask * tf.log(responsible_outputs + 1e-7) * td_error)

      total_loss = policy_loss + entropy_loss + critic_loss + term_loss

      gradients, variables = (
        zip(*self._network_optimizer.compute_gradients(total_loss)))
      optimize = self._network_optimizer.apply_gradients(
        zip(gradients, variables))
      summary = tf.summary.merge([
        tf.summary.scalar('avg_critic_loss', critic_loss),
        tf.summary.scalar('avg_termination_loss', term_loss),
        tf.summary.scalar('avg_entropy_loss', entropy_loss),
        tf.summary.scalar('avg_policy_loss', policy_loss),
        tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
        utility.gradient_summaries(
          zip(gradients, variables))])
      with tf.control_dependencies([optimize]):
        print_loss = tf.Print(0, [total_loss], 'network loss: ')

      with tf.control_dependencies([total_loss, print_loss]):
        return summary

  def get_intra_option_policy(self, network, option):
    current_option_option_one_hot = tf.one_hot(option, self._nb_options, name="options_one_hot")
    current_option_option_one_hot = tf.tile(current_option_option_one_hot[..., None], [1, 1, 1, self._action_size])
    action_probabilities = tf.reduce_sum(tf.multiply(network.options, current_option_option_one_hot),
                                              reduction_indices=2, name="P_a")
    return action_probabilities

  def get_option_termination(self, network, current_option):
    current_option_option_one_hot = tf.one_hot(current_option, self._nb_options, name="options_one_hot")
    o_terminations = tf.reduce_sum(tf.multiply(network.termination, current_option_option_one_hot),
                             reduction_indices=2, name="O_Terminations")
    return o_terminations

  def get_responsible_outputs(self, policy, action):
    actions_onehot = tf.one_hot(action, self._action_size, dtype=tf.float32,
                                     name="Actions_Onehot")
    responsible_outputs = tf.reduce_sum(policy * actions_onehot, [2])
    return responsible_outputs

  def get_V(self, network):
    q_val = network.q_val
    v = tf.reduce_max(q_val, axis=2) * (1 - self.probability_of_random_option) +\
        self.probability_of_random_option * tf.reduce_mean(q_val, axis=2)
    return v

  def get_Q(self, network, current_option):
    current_option_option_one_hot = tf.one_hot(current_option, self._nb_options, name="options_one_hot")
    q_values = tf.reduce_sum(tf.multiply(network.q_val, current_option_option_one_hot),
                                              reduction_indices=2, name="Values_Q")
    return q_values

  def _mask(self, tensor, length):
    """Set padding elements of a batch of sequences to zero.

    Useful to then safely sum along the time dimension.

    Args:
      tensor: Tensor of sequences.
      length: Batch of sequence lengths.

    Returns:
      Masked sequences.
    """
    with tf.name_scope('mask'):
      range_ = tf.range(tensor.shape[1].value)
      mask = tf.cast(range_[None, :] < length[:, None], tf.float32)
      masked = tensor * mask
      return tf.check_numerics(masked, 'masked')