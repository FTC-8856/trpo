
import argparse
import os
import pickle
from datetime import datetime

from gym import wrappers

from trpo.constants import env_name
from trpo.core import (init_gym, load_model, make_model, run_episode, save_model,
                  train)
from trpo.policy import Policy
from trpo.utils import Scaler
from trpo.value import NNValueFunction
import tempfile

parser = argparse.ArgumentParser(
    description='Will render a new episode using the saved model from the given folder.')
parser.add_argument('folder', metavar='F', type=str,
                    help='folder to load from')
args = parser.parse_args()

env, obs_dim, act_dim = init_gym(env_name)
obs_dim += 1
env = wrappers.Monitor(env, tempfile.mkstemp(), force=True)
val_func, policy, scaler = load_model(args.folder)
run_episode(env, policy, scaler, animate=True)
