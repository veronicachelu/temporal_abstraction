# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for using reinforcement learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import re

import ruamel.yaml as yaml
import tensorflow as tf
import collections


def define_saver(exclude=None):
  """Create a saver for the variables we want to checkpoint.

  Args:
    exclude: List of regexes to match variable names to exclude.

  Returns:
    Saver object.
  """
  variables = []
  exclude = exclude or []
  exclude = [re.compile(regex) for regex in exclude]
  for variable in tf.global_variables():
    if any(regex.match(variable.name) for regex in exclude):
      continue
    variables.append(variable)
  saver = tf.train.Saver(variables, max_to_keep=10000)
  return saver


# def define_network(constructor, config, action_size):
#   """Constructor for the recurrent cell for the algorithm.
#
#   Args:
#     constructor: Callable returning the network as RNNCell.
#     config: Object providing configurations via attributes.
#     action_size: Integer indicating the amount of action dimensions.
#
#   Returns:
#     Created recurrent cell object.
#   """
#   mean_weights_initializer = (
#       tf.contrib.layers.variance_scaling_initializer(
#           factor=config.init_mean_factor))
#   logstd_initializer = tf.random_normal_initializer(
#       config.init_logstd, 1e-10)
#   network = constructor(
#       config.policy_layers, config.value_layers, action_size,
#       mean_weights_initializer=mean_weights_initializer,
#       logstd_initializer=logstd_initializer)
#   return network

def define_network(constructor, config, action_size):
  """Constructor for the recurrent cell for the algorithm.

  Args:
    constructor: Callable returning the network as RNNCell.
    config: Object providing configurations via attributes.
    action_size: Integer indicating the amount of action dimensions.

  Returns:
    Created recurrent cell object.
  """
  network = constructor(
      config.conv_layers, config.fc_layers, action_size, config.nb_options, config.num_agents)
  return network

def initialize_variables(sess, saver, logdir, checkpoint=None, resume=None):
  """Initialize or restore variables from a checkpoint if available.

  Args:
    sess: Session to initialize variables in.
    saver: Saver to restore variables.
    logdir: Directory to search for checkpoints.
    checkpoint: Specify what checkpoint name to use; defaults to most recent.
    resume: Whether to expect recovering a checkpoint or starting a new run.

  Raises:
    ValueError: If resume expected but no log directory specified.
    RuntimeError: If no resume expected but a checkpoint was found.
  """
  sess.run(tf.group(
      tf.local_variables_initializer(),
      tf.global_variables_initializer()))
  if resume and not (logdir or checkpoint):
    raise ValueError('Need to specify logdir to resume a checkpoint.')
  if logdir:
    state = tf.train.get_checkpoint_state(logdir)
    if checkpoint:
      checkpoint = os.path.join(logdir, checkpoint)
    if not checkpoint and state and state.model_checkpoint_path:
      checkpoint = state.model_checkpoint_path
    if checkpoint and resume is False:
      message = 'Found unexpected checkpoint when starting a new run.'
      raise RuntimeError(message)
    if checkpoint:
      saver.restore(sess, checkpoint)


def save_config(config, logdir=None):
  """Save a new configuration by name.

  If a logging directory is specified, is will be created and the configuration
  will be stored there. Otherwise, a log message will be printed.

  Args:
    config: Configuration object.
    logdir: Location for writing summaries and checkpoints if specified.

  Returns:
    Configuration object.
  """
  if logdir:
    with config.unlocked:
      config.logdir = logdir
    message = 'Start a new run and write summaries and checkpoints to {}.'
    tf.logging.info(message.format(config.logdir))
    tf.gfile.MakeDirs(config.logdir)
    config_path = os.path.join(config.logdir, 'config.yaml')
    with tf.gfile.FastGFile(config_path, 'w') as file_:
      yaml.dump(config, file_, default_flow_style=False)
  else:
    message = (
        'Start a new run without storing summaries and checkpoints since no '
        'logging directory was specified.')
    tf.logging.info(message)
  return config


def load_config(logdir):
  """Load a configuration from the log directory.

  Args:
    logdir: The logging directory containing the configuration file.

  Raises:
    IOError: The logging directory does not contain a configuration file.

  Returns:
    Configuration object.
  """
  config_path = logdir and os.path.join(logdir, 'config.yaml')
  if not config_path or not tf.gfile.Exists(config_path):
    message = (
        'Cannot resume an existing run since the logging directory does not '
        'contain a configuration file.')
    raise IOError(message)
  with tf.gfile.FastGFile(config_path, 'r') as file_:
    config = yaml.load(file_)
  message = 'Resume run and write summaries and checkpoints to {}.'
  tf.logging.info(message.format(config.logdir))
  return config


def set_up_logging():
  """Configure the TensorFlow logger."""
  tf.logging.set_verbosity(tf.logging.INFO)
  logging.getLogger('tensorflow').propagate = False


def gradient_summaries(grad_vars, groups=None, scope='gradients'):
  """Create histogram summaries of the gradient.

  Summaries can be grouped via regexes matching variables names.

  Args:
    grad_vars: List of (gradient, variable) tuples as returned by optimizers.
    groups: Mapping of name to regex for grouping summaries.
    scope: Name scope for this operation.

  Returns:
    Summary tensor.
  """

  # groups = groups or {r'all': r'.*'}
  # grouped = collections.defaultdict(list)
  summaries = []
  for grad, var in grad_vars:
    if grad is None:
      continue
    # for name, pattern in groups.items():
    #   if re.match(pattern, var.name):
    #     name = re.sub(pattern, name, var.name)
    #     grouped[name].append(grad)
  # for name in groups:
  #   if name not in grouped:
  #     tf.logging.warn("No variables matching '{}' group.".format(name))
  # summaries = []
  # for grads in grouped.items():
  #   grads = [tf.reshape(grad, [-1]) for grad in grads]
  #   grads = tf.concat(grads, 0)
    summaries.append(tf.summary.histogram(scope + '/' + var.name.replace(':', '_') + '_grad', grad))
    summaries.append(tf.summary.histogram(scope + '/' + var.name.replace(':', '_'), var))
  return tf.summary.merge(summaries)


def variable_summaries(vars_, groups=None, scope='weights'):
  """Create histogram summaries for the provided variables.

  Summaries can be grouped via regexes matching variables names.

  Args:
    vars_: List of variables to summarize.
    groups: Mapping of name to regex for grouping summaries.
    scope: Name scope for this operation.

  Returns:
    Summary tensor.
  """
  groups = groups or {r'all': r'.*'}
  grouped = collections.defaultdict(list)
  for var in vars_:
    for name, pattern in groups.items():
      if re.match(pattern, var.name):
        name = re.sub(pattern, name, var.name)
        grouped[name].append(var)
  for name in groups:
    if name not in grouped:
      tf.logging.warn("No variables matching '{}' group.".format(name))
  summaries = []
  for name, vars_ in grouped.items():
    vars_ = [tf.reshape(var, [-1]) for var in vars_]
    vars_ = tf.concat(vars_, 0)
    summaries.append(tf.summary.histogram(scope + '/' + name, vars_))
  return tf.summary.merge(summaries)


def huber_loss(x, delta=1.0):
  """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
  return tf.where(
    tf.abs(x) < delta,
    tf.square(x) * 0.5,
    delta * (tf.abs(x) - 0.5 * delta)
  )