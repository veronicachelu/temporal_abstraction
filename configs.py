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

"""Example configurations using the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-variable

from agents import LinearSFAgent
from agents import DIFAgent
from agents import EigenOCAgent
from agents import EigenOCAgentDyn
from env_wrappers import GridWorld
from env_wrappers import Gridworld_NonMatching
import functools
import networks


def default():
  """Default configuration for PPO."""
  num_agents = 8
  eval_episodes = 1
  use_gpu = False
  max_length = 100
  return locals()

def linear_4rooms():
  locals().update(default())
  linear_sf_agent = LinearSFAgent
  num_agents = 8
  use_gpu = False
  nb_options = 4
  # Network
  network = networks.LinearSFNetwork
  weight_summaries = dict(
      all=r'.*',
      conv=r'.*/conv/.*',
      fc=r'.*/fc/.*',
      term=r'.*/option_term/.*',
      q_val=r'.*/q_val/.*',
      policy=r'.*/i_o_policies/.*')

  # conv_layers = (8, 4, 16), (4, 2, 32)
  input_size = (13, 13)
  history_size = 3
  conv_layers = (5, 2, 32),
  fc_layers = 128,
  sf_layers = 128, 128
  # Optimization
  network_optimizer = 'AdamOptimizer'
  # lr = 0.0007
  lr = 1e-3
  sf_lr = 1e-3
  discount = 0.985
  entropy_coef = 1e-4 #0.01
  critic_coef = 0.5
  sf_coef = 1
  instant_r_coef = 1
  option_entropy_coef = 0.01
  auto_coef = 1

  env = functools.partial(
    GridWorld, "./mdps/4rooms.mdp")
  max_update_freq = 30
  min_update_freq = 5
  steps = 1e6  # 1M
  explore_steps = 1e5
  final_random_action_prob = 0.1
  initial_random_action_prob = 1.0
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

def dif_4rooms():
  locals().update(default())
  dif_agent = DIFAgent
  num_agents = 8
  use_gpu = False
  nb_options = 4
  # Network
  network = networks.DIFNetwork
  weight_summaries = dict(
      all=r'.*',
      conv=r'.*/conv/.*',
      fc=r'.*/fc/.*',
      term=r'.*/option_term/.*',
      q_val=r'.*/q_val/.*',
      policy=r'.*/i_o_policies/.*')

  conv_layers = (5, 2, 64),
  input_size = (13, 13)
  history_size = 3
  fc_layers = 128,
  sf_layers = 128, 256, 128
  aux_fc_layers = 128,
  aux_deconv_layers = (4, 2, 1, 64), (4, 2, 0, 64), (3, 2, 0, 3)
  # Optimization
  network_optimizer = 'RMSPropOptimizer'
  # lr = 0.0007
  lr = 1e-4
  discount = 0.985
  entropy_coef = 1e-4
  critic_coef = 0.5
  sf_coef = 1
  instant_r_coef = 1
  option_entropy_coef = 0.01
  aux_coef = 1

  env = functools.partial(
    GridWorld, "./mdps/4rooms.mdp")
  max_update_freq = 30
  min_update_freq = 5
  steps = 1e6   # 1M
  training_steps = 5e5
  explore_steps = 1e5
  final_random_action_prob = 0.1
  gradient_clip_value = 40
  summary_interval = 10
  checkpoint_interval = 10
  eval_interval = 1
  policy_steps = 1e3
  # sf_transition_matrix_steps = 50000#e3
  # sf_transition_options_steps = 50000#e3
  sf_transition_matrix_size = 50000

  return locals()

def dif_4rooms_fc():
  locals().update(default())
  dif_agent = DIFAgent
  num_agents = 8
  use_gpu = False
  nb_options = 4
  network = networks.DIFNetwork_FC
  weight_summaries = dict(
      all=r'.*',
      conv=r'.*/conv/.*',
      fc=r'.*/fc/.*',
      term=r'.*/option_term/.*',
      q_val=r'.*/q_val/.*',
      policy=r'.*/i_o_policies/.*')

  input_size = (13, 13)
  history_size = 3
  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,
  network_optimizer = 'AdamOptimizer'
  lr = 1e-3
  discount = 0.985
  sf_coef = 1
  aux_coef = 1
  target_update_iter_aux = 1
  target_update_iter_sf = 30

  env = functools.partial(
    GridWorld, "./mdps/4rooms.mdp")
  max_update_freq = 30
  min_update_freq = 5
  batch_size = 16
  memory_size = 500000
  observation_steps = 1000
  aux_update_freq = 1
  steps = 1e6   # 1M
  training_steps = 5e5
  final_random_action_prob = 0.1
  initial_random_action_prob = 1.0
  gradient_clip_value = 40
  summary_interval = 10
  checkpoint_interval = 10
  max_length = 1e6

  return locals()

def oc():
  locals().update(default())
  dif_agent = EigenOCAgent
  num_agents = 12
  use_gpu = False
  nb_options = 4
  eigen = False
  # Network
  network = networks.EignOCNetwork
  weight_summaries = dict(
      all=r'.*')

  input_size = (13, 13)
  history_size = 3
  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,
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
  goal_locations = [(1, 11), (3, 2), (6, 2), (1, 4), (1, 1), (8, 1), (2, 5), (11, 10)]

  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")
  max_update_freq = 30
  min_update_freq = 5
  batch_size = 32
  memory_size = 100000
  observation_steps = 16*4
  aux_update_freq = 1
  steps = 1000000  # 1M
  episodes = 1e6  # 1M
  eigen_exploration_steps = 16*4
  # explore_steps = 1
  final_random_option_prob = 0.1
  final_random_action_prob = 0.01
  # initial_random_action_prob = 1.0
  gradient_clip_norm_value = 40
  steps_summary_interval = 1000
  episode_summary_interval = 1
  steps_checkpoint_interval = 1000
  episode_checkpoint_interval = 1
  episode_eval_interval = 10
  max_length = 1000
  max_length_eval = 1000
  clip_option_grad_by_value = False
  clip_by_value = 5
  nb_test_ep = 100
  include_primitive_options = True
  move_goal_nb_of_ep = 1000
  sr_matrix_size = 169

  return locals()

def eigenoc():
  locals().update(default())
  dif_agent = EigenOCAgent
  num_agents = 12
  use_gpu = False
  nb_options = 4
  eigen = True
  # Network
  network = networks.EignOCNetwork
  weight_summaries = dict(
      all=r'.*')

  input_size = (13, 13)
  history_size = 3
  fc_layers = 128,
  sf_layers = 128,
  aux_fc_layers = 507,
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
  goal_locations = [(1, 11), (3, 2), (6, 2), (1, 4), (1, 1), (8, 1), (2, 5), (11, 10)]

  env = functools.partial(
    GridWorld, goal_locations, "./mdps/4rooms.mdp")
  max_update_freq = 30
  min_update_freq = 5
  batch_size = 32
  memory_size = 100000
  observation_steps = 16*4
  aux_update_freq = 1
  alpha_r = 0.75
  steps = 1000000  # 1M
  episodes = 1e6  # 1M
  eigen_exploration_steps = 16*4
  # explore_steps = 1
  final_random_option_prob = 0.1
  final_random_action_prob = 0.01
  # initial_random_action_prob = 1.0
  gradient_clip_norm_value = 40
  steps_summary_interval = 1000
  episode_summary_interval = 1
  steps_checkpoint_interval = 1000
  episode_checkpoint_interval = 1
  episode_eval_interval = 10
  max_length = 1000
  max_length_eval = 1000
  clip_option_grad_by_value = False
  clip_by_value = 5
  nb_test_ep = 100
  # recompute_eigenvect_every = 1000
  # stop_recompute_eigenvect_every = 10000
  first_eigenoption = 1
  move_goal_nb_of_ep = 1000
  include_primitive_options = True
  sr_matrix_size = 169

  return locals()

def eigenoc_dyn():
  locals().update(eigenoc())
  dif_agent = EigenOCAgentDyn

  goal_locations = [(1, 11), (3, 2), (6, 2), (1, 4), (11, 10)]

  sf_matrix_size = 10000
  steps = 10000000  # 10M

  return locals()

def eigenoc_montezuma():
  locals().update(default())
  dif_agent = EigenOCAgentDyn
  num_agents = 12
  use_gpu = False
  nb_options = 8
  eigen = True
  # Network
  network = networks.EignOCMontezumaNetwork
  weight_summaries = dict(
      all=r'.*')

  input_size = (84, 84)
  history_size = 4
  channel_size = 1
  conv_layers = (6, 2, 0, 64), (6, 2, 2, 64), (6, 2, 2, 64),
  upconv_layers = (6, 2, 2, 64), (6, 2, 2, 64), (6, 2, 0, 1)
  fc_layers = 1024, 2048
  sf_layers = 2048, 1024, 2048
  aux_fc_layers = 2048, 1024, 10*10*64
  aux_upconv_reshape = (10, 10, 64)
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

  env = "MontezumaRevenge-v0"
  max_update_freq = 30
  min_update_freq = 5
  batch_size = 32
  memory_size = 500000
  observation_steps = 16*4
  aux_update_freq = 1
  alpha_r = 0.75
  steps = -1  # 10M
  episodes = 1e6  # 1M
  eigen_exploration_steps = 16*4
  # explore_steps = 1
  final_random_option_prob = 0.1
  final_random_action_prob = 0.01
  # initial_random_action_prob = 1.0
  gradient_clip_norm_value = 40
  steps_summary_interval = 1000
  episode_summary_interval = 1
  steps_checkpoint_interval = 1000
  episode_checkpoint_interval = 1
  episode_eval_interval = 100
  max_length_eval = 1000
  clip_option_grad_by_value = False
  clip_by_value = 5
  nb_test_ep = 1
  first_eigenoption = 1
  include_primitive_options = True
  sf_matrix_size = 50000

  return locals()

def oc_dyn():
  locals().update(oc())
  dif_agent = EigenOCAgentDyn

  goal_locations = [(1, 11), (3, 2), (6, 2), (1, 4), (11, 10)]

  sf_matrix_size = 10000
  steps = 10000000  # 10M

  return locals()
