
import argparse
import os
import pickle
from datetime import datetime

from gym import wrappers

from core import (load_model, make_model, train, save_model, init_gym)
from policy import Policy
from utils import Scaler
from value import NNValueFunction
from constants import env_name

parser = argparse.ArgumentParser(
    description='Will either create a new model or load a saved model from the given folder. Model will be trained and saved in the given folder.')
parser.add_argument('folder', metavar='F', type=str,
                    help='folder to save to and optionally load from')
args = parser.parse_args()


env, obs_dim, act_dim = init_gym(env_name)
obs_dim += 1
now = datetime.now().strftime("%b-%d_%H:%M:%S")
aigym_path = os.path.join('/tmp', env_name, now)
env = wrappers.Monitor(env, aigym_path, force=True)
try:
    val_func, policy, scaler = load_model(args.folder)
except BaseException as e:
    print(e)
    val_func, policy, scaler = make_model(obs_dim, act_dim)
train(env, policy, scaler, val_func)
save_model(args.folder, val_func, policy, scaler)