
import argparse
import os
import pickle
from datetime import datetime

from gym import wrappers

from core import init_gym, load_model, make_model, run_episode, save_model, train
from policy import Policy
from utils import Scaler
from value import NNValueFunction
from constants import env_name

parser = argparse.ArgumentParser(
    description='Will render a new episode using the saved model from the given folder.')
parser.add_argument('folder', metavar='F', type=str,
                    help='folder to load from')
args = parser.parse_args()

env, obs_dim, act_dim = init_gym(env_name)
obs_dim += 1
now = datetime.now().strftime("%b-%d_%H:%M:%S")
aigym_path = os.path.join('/tmp', env_name, now)
env = wrappers.Monitor(env, aigym_path, force=True)
val_func, policy, scaler = load_model(args.folder)
run_episode(env, policy, scaler, animate=True)
