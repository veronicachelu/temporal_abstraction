import threading
import datetime
import os
import gym
import tensorflow as tf
import tools
import config_utility
from env_tools import env_wrappers as wrappers
import configs

def _create_environment(config):
  if isinstance(config.env, str):
    env = gym.make(config.env)
  else:
    env = config.env()
  if config.max_length:
    env = wrappers.LimitDuration(env, config.max_length)
  if config.history_size == 3 or config.history_size == 1 :
    env = wrappers.FrameResize(env, config.input_size)
  else:
    env = wrappers.ActionRepeat(env, config.history_size)
    env = wrappers.FrameHistoryGrayscaleResize(env, config.input_size)

  # env = tools.wrappers.ClipAction(env)
  env = wrappers.ConvertTo32Bit(env)
  return env