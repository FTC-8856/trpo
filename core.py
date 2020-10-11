import argparse
import os
import pickle
import signal
from datetime import datetime

import gym
import numpy as np
import scipy.signal
from gym import wrappers

from constants import (batch_size, env_name, gamma, hid1_mult, init_logvar,
                       kl_targ, lam, num_episodes)
from policy import Policy
from utils import Scaler
from value import NNValueFunction


class GracefulKiller:

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate=False):
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = np.concatenate([obs, [step]])  # add time step feature
        obs = obs.astype(np.float32).reshape((1, -1))
        unscaled_obs.append(obs)
        # center and scale observations
        obs = np.float32((obs - offset) * scale)
        observes.append(obs)
        action = policy.sample(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action.flatten())
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, episodes):
    trajectories = []
    for _ in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(
            env, policy, scaler)
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)

    return trajectories


def discount(x, gamma):
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def make_model(obs_dim, act_dim):
    val_func = NNValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ,
                    hid1_mult, init_logvar)
    scaler = Scaler(obs_dim)
    return val_func, policy, scaler


def load_model(folder):
    with open(folder + 'val_func.pickle', 'rb') as f:
        val_func = pickle.dump(f)
    with open(folder + 'policy.pickle', 'rb') as f:
        policy = pickle.dump(f)
    with open(folder + 'scaler.pickle', 'rb') as f:
        scaler = pickle.dump(f)
    return val_func, policy, scaler


def save_model(folder, val_func, policy, scaler):
    with open(folder + 'val_func.pickle', 'wb') as f:
        pickle.dump(val_func, f, pickle.HIGHEST_PROTOCOL)
    with open(folder + 'policy.pickle', 'wb') as f:
        pickle.dump(policy, f, pickle.HIGHEST_PROTOCOL)
    with open(folder + 'scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)


def train(env, policy, scaler, val_func):
    episode = 0
    killer = GracefulKiller()
    while episode < num_episodes:
        trajectories = run_policy(
            env, policy, scaler, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)
        add_disc_sum_rew(trajectories, gamma)
        add_gae(trajectories, gamma, lam)
        observes, actions, advantages, disc_sum_rew = build_train_set(
            trajectories)
        policy.update(observes, actions, advantages)
        val_func.fit(observes, disc_sum_rew)
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False


def add_disc_sum_rew(trajectories, gamma):
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values.flatten()


def add_gae(trajectories, gamma, lam):
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew
