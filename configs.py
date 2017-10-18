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

from agents import AOCAgent
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

def aoc():
  locals().update(default())
  agent = AOCAgent
  num_agents = 8
  use_gpu = False

  # Network
  network = networks.AOCNetwork
  weight_summaries = dict(
      all=r'.*',
      conv=r'.*/conv/.*',
      fc=r'.*/fc/.*',
      term=r'.*/option_term/.*',
      q_val=r'.*/q_val/.*',
      policy=r'.*/i_o_policies/.*')

  # conv_layers = (8, 4, 16), (4, 2, 32)
  input_size = (13,13)
  history_size = 3
  conv_layers = (5, 2, 32),
  deconv_layers = (4, 2, 0, 128), (4, 2, 1, 64), (4, 2, 0, 32), ()
  # fc_layers = 256,
  fc_layers = 128,
  sf_layers = 256, 128, 256
  # Optimization
  network_optimizer = 'AdamOptimizer'
  # lr = 0.0007
  lr = 1e-3

  # Losses
  discount = 0.985

  entropy_coef = 1e-4 #0.01
  critic_coef = 0.5

  # nb_options = 8
  nb_options = 4
  env = functools.partial(
    GridWorld, "./mdps/4rooms.mdp")
  # env = Gridworld_NonMatching
  max_update_freq = 30
  min_update_freq = 5
  steps = 1e6  # 1M
  explore_steps = 1
  final_random_action_prob = 0.1
  initial_random_action_prob = 1.0
  delib_cost = 0
  margin_cost = 0
  gradient_clip_value = 40
  summary_interval = 1
  checkpoint_interval = 1
  eval_interval = 100

  return locals()

def sf():
  locals().update(default())
  agent = AOCAgent
  num_agents = 8
  use_gpu = False

  # Network
  network = networks.AOCNetwork
  weight_summaries = dict(
      all=r'.*',
      conv=r'.*/conv/.*',
      fc=r'.*/fc/.*',
      term=r'.*/option_term/.*',
      q_val=r'.*/q_val/.*',
      policy=r'.*/i_o_policies/.*')

  # conv_layers = (8, 4, 16), (4, 2, 32)
  input_size = (13,13)
  history_size = 3
  conv_layers = (5, 2, 32),
  deconv_layers = (4, 2, 0, 128), (4, 2, 1, 64), (4, 2, 0, 32), ()
  # fc_layers = 256,
  fc_layers = 128,
  sf_layers = 256, 128, 256
  # Optimization
  network_optimizer = 'AdamOptimizer'
  # lr = 0.0007
  lr = 1e-3

  # Losses
  discount = 0.985

  entropy_coef = 1e-4 #0.01
  critic_coef = 0.5

  # nb_options = 8
  nb_options = 4
  env = functools.partial(
    GridWorld, "./mdps/toy.mdp")
  # env = Gridworld_NonMatching
  max_update_freq = 30
  min_update_freq = 5
  steps = 1e6  # 1M
  explore_steps = 1
  final_random_action_prob = 0.1
  initial_random_action_prob = 1.0
  delib_cost = 0
  margin_cost = 0
  gradient_clip_value = 40
  summary_interval = 1
  checkpoint_interval = 1
  eval_interval = 100

  return locals()

