
import argparse
import os
import pickle
from datetime import datetime

from gym import wrappers

from trpo.constants import env_name
from trpo.core import evaluate, init_gym, load_model, make_model, save_model, train
from trpo.policy import Policy
from trpo.utils import Scaler
from trpo.value import NNValueFunction
import tempfile

parser = argparse.ArgumentParser(
    description='Will evaluate the saved model from the given folder.')
parser.add_argument('folder', metavar='F', type=str,
                    help='folder to load from')
args = parser.parse_args()

env, obs_dim, act_dim = init_gym(env_name)
obs_dim += 1
env = wrappers.Monitor(env, env=wrappers.Monitor(
    env, tempfile.mkstemp(), force=True), force=True)
val_func, policy, scaler = load_model(args.folder)
evaluate(env, policy, scaler)
