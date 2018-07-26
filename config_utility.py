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

def initialize_variables(sess, loader, checkpoint=None, resume=None):
  """Initialize or restore variables from a checkpoint if available."""
  if resume:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint, "models"))
    print("Loading Model from {}".format(ckpt.model_checkpoint_path))
    loader.restore(sess, ckpt.model_checkpoint_path)
    sess.run(tf.local_variables_initializer())
  else:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


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