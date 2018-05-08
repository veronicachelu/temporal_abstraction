from agents import SomAgent
from agents import TargetAgent
from agents import BehaviourAgent
from agents import EigenOCAgent
from agents import LinearSFAgent
from agents import EigenOCAgentDyn
from agents import DynSRAgent
from env_tools import GridWorld
import functools
from networks import SomNetwork
from networks import ExplorationNetwork
from networks import EignOCNetwork
from networks import LinearSFNetwork
from networks import DynSRNetwork
from networks import EignOCMontezumaNetwork

def default():
  num_agents = 8
  use_gpu = False

  weight_summaries = dict(
    all=r'.*')
  input_size = (13, 13)
  history_size = 3
  network_optimizer = 'AdamOptimizer'
  lr = 0.0001
  discount = 0.99
  sf_coef = 1
  aux_coef = 1
  entropy_coef = 0.01
  critic_coef = 1
  eigen_critic_coef = 1
  target_update_iter_aux = 1
  target_update_iter_sf = 30
  target_update_iter_option = 30

  goal_locations = [(11, 7), (5, 2)]#, (1, 10), (2, 2), (6, 2)]
  #goal_locations = [(1, 11), (3, 2)]

  move_goal_nb_of_ep = 500

  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")

  max_update_freq = 30
  min_update_freq = 5
  aux_update_freq = 1

  steps = -1  # 10M
  episodes = 1e6  # 1M

  final_random_option_prob = 0.1
  initial_random_option_prob = 0.1
  decrease_option_prob = False
  final_random_action_prob = 0.01
  explore_options_episodes = 2000

  nb_test_ep = 100
  max_length = 3000

  gradient_clip_norm_value = 40
  clip_option_grad_by_value = False
  clip_by_value = 5

  steps_summary_interval = 1000
  episode_summary_interval = 10
  steps_checkpoint_interval = 1000
  episode_checkpoint_interval = 10
  episode_eval_interval = 10

  logging = False
  evaluation = False
  multi_task = True

  return locals()

def linear_sf():
  locals().update(default())
  target_agent = LinearSFAgent
  num_agents = 8
  nb_options = 4
  network = LinearSFNetwork

  input_size = (13, 13)
  history_size = 3
  conv_layers = (5, 2, 32),
  fc_layers = 128,
  sf_layers = 128, 128
  lr = 1e-3
  sf_lr = 1e-3
  discount = 0.985
  entropy_coef = 1e-4 #0.01
  critic_coef = 0.5
  sf_coef = 1
  instant_r_coef = 1
  option_entropy_coef = 0.01
  auto_coef = 1

  steps = -1  # 1M
  explore_steps = 1e5
  delib_cost = 0
  margin_cost = 0
  gradient_clip_value = 40
  summary_interval = 10
  checkpoint_interval = 1
  eval_interval = 1
  policy_steps = 1e3
  sf_transition_matrix_steps = 300#e3
  sf_transition_options_steps = 400#e3
  sf_transition_matrix_size = 1e3

  return locals()

def dynamic_SR():
  locals().update(default())
  target_agent = DynSRAgent
  num_agents = 8
  network = DynSRNetwork

  input_size = (13, 13)
  history_size = 3
  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,
  lr = 1e-3
  discount = 0.985

  batch_size = 16
  memory_size = 500000
  observation_steps = 1000
  steps = 1e6   # 1M
  training_steps = 5e5
  summary_interval = 10
  checkpoint_interval = 10
  max_length = 1e20

  return locals()

def oc():
  locals().update(default())
  nb_options = 4
  target_agent = EigenOCAgent
  eigen = False
  network = EignOCNetwork

  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,

  batch_size = 16
  memory_size = 100000
  observation_steps = 16*4

  steps = -1  # 1M
  episodes = 1e6  # 1M
  eigen_exploration_steps = 16*4
  max_length = 1000
  max_length_eval = 1000
  include_primitive_options = True
  sr_matrix_size = 169
  sr_matrix = "static"
  goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2)]
  #goal_locations = [(1, 11), (3, 2), (6, 2), (1, 4), (1, 1), (8, 1), (2, 5), (11, 10)]
  move_goal_nb_of_ep = 1000
  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")

  return locals()

def eigenoc():
  locals().update(default())
  target_agent = EigenOCAgent
  nb_options = 4
  eigen = True
  network = EignOCNetwork

  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,
  sf_coef = 1
  batch_size = 16
  memory_size = 100000
  observation_steps = 16*4

  alpha_r = 0.75
  eigen_exploration_steps = 16*4
  max_length = 1000
  max_length_eval = 1000
  first_eigenoption = 1
  include_primitive_options = True
  sr_matrix_size = 169
  sr_matrix = "static"
  # goal_locations = [(11, 7), (5, 2)] #, (1, 10), (2, 2), (6, 2)]
  # goal_locations = [(1, 11), (3, 2)]
  goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2)]
  move_goal_nb_of_ep = 1000
  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")
  tau = 0.1
  eigen_approach = "SVD"
  return locals()

def eigenoc_dyn():
  locals().update(eigenoc())
  target_agent = EigenOCAgentDyn
  sf_matrix_size = 5000
  sr_matrix = "dynamic"
  eigen_approach = "SVD"
  goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2)]
  move_goal_nb_of_ep = 1000
  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")
  return locals()

def oc_dyn():
  locals().update(oc())
  target_agent = EigenOCAgentDyn
  sr_matrix = None
  goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2)]
  move_goal_nb_of_ep = 1000
  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")
  return locals()

def eigenoc_montezuma():
  locals().update(default())
  target_agent = EigenOCAgentDyn
  eigen = True
  network = EignOCMontezumaNetwork

  input_size = (84, 84)
  history_size = 4
  channel_size = 1
  conv_layers = (6, 2, 0, 64), (6, 2, 2, 64), (6, 2, 2, 64),
  upconv_layers = (6, 2, 2, 64), (6, 2, 2, 64), (6, 2, 0, 1)
  fc_layers = 1024, 2048
  sf_layers = 2048, 1024, 2048
  aux_fc_layers = 2048, 1024, 10*10*64
  aux_upconv_reshape = (10, 10, 64)

  env = "MontezumaRevenge-v0"
  batch_size = 32
  memory_size = 500000
  observation_steps = 16*4
  alpha_r = 0.75
  steps = -1  # 10M
  eigen_exploration_steps = 16*4
  episode_eval_interval = 100
  max_length_eval = 1000
  nb_test_ep = 1
  first_eigenoption = 1
  include_primitive_options = True
  sf_matrix_size = 50000
  sr_matrix = "dynamic"
  eigen_approach = "NN"
  multi_task = False

  return locals()

def som():
  locals().update(default())
  target_agent = SomAgent
  nb_options = 4
  num_agents = 12
  eigen = True
  network = SomNetwork

  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,

  batch_size = 16
  memory_size = 100000
  observation_steps = 16*4

  alpha_r = 0.75
  eigen_exploration_steps = 16*4
  max_length = 1000
  max_length_eval = 1000
  first_eigenoption = 1
  include_primitive_options = True
  sr_matrix_size = 128
  sr_matrix = "static"
  goal_locations = [(11, 7), (5, 2)]#, (1, 10), (2, 2), (6, 2)]
  #goal_locations = [(1, 11), (3, 2)]
  move_goal_nb_of_ep = 1000
  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")

  final_random_option_prob = 0.1
  initial_random_option_prob = 0.1
  decrease_option_prob = False

  reward_update_freq = 1
  target_update_iter_reward = 1
  tau = 0.1
  reward_coef = 1
  reward_i_coef = 1
  adam_epsilon = 1e-08
  plot_every = 10

  return locals()

def exploration():
  locals().update(default())
  target_agent = TargetAgent
  behaviour_agent = BehaviourAgent
  nb_options = 4
  num_agents = 12
  eigen = True
  network = ExplorationNetwork

  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,

  batch_size = 16
  memory_size = 100000
  observation_steps = 16*4

  alpha_r = 0.75
  eigen_exploration_steps = 16*4
  max_length = 1000
  max_length_eval = 1000
  first_eigenoption = 1
  include_primitive_options = True
  sr_matrix_size = 128
  sr_matrix = "static"
  goal_locations = [(11, 7), (5, 2)]#, (1, 10), (2, 2), (6, 2)]
  #goal_locations = [(1, 11), (3, 2)]
  move_goal_nb_of_ep = 500
  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")
  reward_update_freq = 1
  target_update_iter_reward = 1
  tau = 0.1
  reward_coef = 1
  reward_i_coef = 1
  adam_epsilon = 1e-08
  plot_every = 10
  # sf_update_freq = 1
  behaviour_update_freq = 1
  target_update_freq_sf = 1000

  final_random_option_prob = 0.1
  initial_random_option_prob = 0.1
  decrease_option_prob = False

  return locals()


def eigenoc_exploration():
  locals().update(eigenoc())
  behaviour_agent = BehaviourAgent
  target_update_iter_aux_behaviour = 1000
  target_update_iter_sf_behaviour = 1000
  behaviour_update_freq = 1

  return locals()