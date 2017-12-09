import threading
import matplotlib
matplotlib.use('Agg')
import datetime
import os
import gym
import tensorflow as tf
import tools
import utility
from tools import wrappers
import configs
from env_wrappers import _create_environment
from tensorflow.python.client import device_lib

def initialize_agents(config):
  global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
  envs = [_create_environment(config) for _ in range(config.num_agents)]
  action_size = envs[0].action_space.n
  nb_states = envs[0].nb_states

  if config.agent_type == "a3c":
    global_network = config.network("global", config, action_size, nb_states)

  if FLAGS.task == "matrix":
    agent = config.agent(envs[0], 0, global_step, config, FLAGS.task)
  elif FLAGS.task == "option":
    agent = config.agent(envs[0], 0, global_step, config, FLAGS.task)
  elif FLAGS.task == "play_option":
    agent = config.agent(envs[0], 0, global_step, config, FLAGS.task)
  else:
    if config.agent_type == "a3c":
      agents = [config.agent(envs[i], i, global_step, config, FLAGS.task) for i in range(config.num_agents)]
      return agents
    else:
      agent = config.agent(envs[0], 0, global_step, config, FLAGS.task)

  return agent

def start_agents(agents, config, coord, sess, saver):
  agent_threads = []
  if FLAGS.task == "matrix":
    thread = threading.Thread(target=(lambda: agents.build_matrix(sess, coord, saver)))
    thread.start()
    agent_threads.append(thread)
  elif FLAGS.task == "option_greedy":
    thread = threading.Thread(target=(lambda: agents.plot_options_greedy(sess, coord, saver)))
    thread.start()
    agent_threads.append(thread)
  elif FLAGS.task == "option":
    thread = threading.Thread(target=(lambda: agents.plot_options(sess, coord, saver)))
    thread.start()
    agent_threads.append(thread)
  elif FLAGS.task == "play_option":
    thread = threading.Thread(target=(lambda: agents.play_option(sess, coord, saver, FLAGS.option, FLAGS.sign)))
    thread.start()
    agent_threads.append(thread)
  else:
    if config.agent_type == "a3c":
      for agent in agents:
        thread = threading.Thread(target=(lambda: agent.play(sess, coord, saver)))
        thread.start()
        agent_threads.append(thread)
    else:
      thread = threading.Thread(target=(lambda: agents.play(sess, coord, saver)))
      thread.start()
      agent_threads.append(thread)

  return agent_threads

# def get_list_vars_load():
#   vars_list = []
#   if FLAGS.resume_option:
#     vars_list = tf.get_collection("variables")
#   else:
#     vars_list = tf.get_collection("variables")
#     option_vars = tf.get_collection("Q")
#     vars_list = [v for v in vars_list if v not in option_vars]
#
#   return vars_list


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def train(config, logdir):
  tf.reset_default_graph()
  sess = tf.Session()
  stage_logdir = os.path.join(logdir, "dif")
  tf.gfile.MakeDirs(stage_logdir)
  with sess:
    # with tf.device("/{}:0".format("gpu" if config.agent_type == "dqn" and len(get_available_gpus()) > 0 else "cpu")):
    with config.unlocked:
      config.logdir = logdir
      config.stage_logdir = stage_logdir
      config.network_optimizer = getattr(tf.train, config.network_optimizer)
      agents = initialize_agents(config)

    # variables_to_load = get_list_vars_load()
    exclude = None
    if not FLAGS.resume_option:
      exclude = (r'.*/Q/.*',)
    loader = utility.define_saver(exclude=exclude)
    saver = utility.define_saver()
    if FLAGS.resume:
      sess.run(tf.global_variables_initializer())
      ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.join(FLAGS.load_from, "dif"), "models"))
      print("Loading Model from {}".format(ckpt.model_checkpoint_path))
      loader.restore(sess, ckpt.model_checkpoint_path)
      sess.run(tf.local_variables_initializer())
    else:
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    agent_threads = start_agents(agents, config, coord, sess, saver)
    coord.join(agent_threads)

def recreate_directory_structure(logdir):
  if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)
  if not FLAGS.resume and FLAGS.train:
    tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)


def main(_):
  utility.set_up_logging()
  if not FLAGS.config:
    raise KeyError('You must specify a configuration.')
  if FLAGS.load_from:
    logdir = FLAGS.logdir = FLAGS.load_from
  else:
    if FLAGS.logdir and os.path.exists(FLAGS.logdir):
      run_number = [int(f.split("-")[0]) for f in os.listdir(FLAGS.logdir) if os.path.isdir(os.path.join(FLAGS.logdir, f)) and FLAGS.config in f]
      run_number = max(run_number) + 1 if len(run_number) > 0 else 0
    else:
      run_number = 0
    logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
      FLAGS.logdir, '{}-{}'.format(run_number, FLAGS.config)))
  try:
    config = utility.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)
  train(config, logdir)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string(
    'logdir', "./logdir",
    'Base directory to store logs.')
  tf.app.flags.DEFINE_string(
    'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
    'Sub directory to store logs.')
  tf.app.flags.DEFINE_string(
    'config', "dqn_sf_4rooms_onehot",
    'Configuration to execute.')
  tf.app.flags.DEFINE_boolean(
    'train', True,
    # 'train', False,
    'Training.')
  tf.app.flags.DEFINE_boolean(
    'resume', False,
    # 'resume', False,
    'Resume.')
  tf.app.flags.DEFINE_boolean(
    'resume_option', False,
    'Resume option.')
  # tf.app.flags.DEFINE_boolean(
  #   'show_training', False,
  #   'Show gym envs.')
  tf.app.flags.DEFINE_string(
    'task', "sf",
    'Task nature')
  tf.app.flags.DEFINE_string(
    # 'load_from', None,
    'load_from', "./logdir/1-dqn_sf_4rooms_fc2",
    'Load directory to load models from.')
  tf.app.flags.DEFINE_integer(
    'option', 0,
    'Option - eigenvector to train dqn on')
  tf.app.flags.DEFINE_string(
    'sign', "poz",
    'Sign of the option eigenvector')
  tf.app.run()
