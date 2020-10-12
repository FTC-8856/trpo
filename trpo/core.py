import argparse
import json
import os
import signal
from datetime import datetime

import gym
import jsonpickle
import numpy as np
import scipy.signal
from gym import wrappers

from trpo.constants import (batch_size, env_name, gamma, init_logvar, kl_targ, lam,
                            num_episodes, policy_hid_list, valfunc_hid_list)
from trpo.policy import Policy
from trpo.utils import Scaler
from trpo.value import NNValueFunction


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
    scale[-1] = 1.0
    offset[-1] = 0.0
    while not done:
        if animate:
            env.render()
        obs = np.concatenate([obs, [step]])
        obs = obs.astype(np.float32).reshape((1, -1))
        unscaled_obs.append(obs)
        obs = np.float32((obs - offset) * scale)
        observes.append(obs)
        action = policy.sample(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action.flatten())
        rewards.append(reward)
        step += 1e-3

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


def make_model(env, obs_dim, act_dim):
    val_func = NNValueFunction(obs_dim, valfunc_hid_list)
    policy = Policy(obs_dim, act_dim, kl_targ, policy_hid_list)
    scaler = Scaler(obs_dim)
    run_policy(env, policy, scaler, episodes=5)
    return val_func, policy, scaler


def load_model(folder):
    with open(os.path.join(folder, 'val_func.json')) as f:
        val_func = jsonpickle.decode(json.load(f))
    with open(os.path.join(folder, 'policy.json')) as f:
        policy = jsonpickle.decode(json.load(f))
    with open(os.path.join(folder, 'scaler.json')) as f:
        scaler = jsonpickle.decode(json.load(f))
    return val_func, policy, scaler


def save_model(folder, val_func, policy, scaler):
    with open(os.path.join(folder, 'val_func.json')) as f:
        json.dump(jsonpickle.encode(val_func), f)
    with open(os.path.join(folder, 'policy.json')) as f:
        json.dump(jsonpickle.encode(policy), f)
    with open(os.path.join(folder, 'scaler.json')) as f:
        json.dump(jsonpickle.encode(scaler), f)


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
    policy.close_sess()
    val_func.close_sess()


def evaluate(env, policy, scaler, val_func):
    episode = 0
    total_reward = 0
    killer = GracefulKiller()
    while episode < 100:
        trajectories = run_policy(
            env, policy, scaler, episodes=batch_size)
        episode += len(trajectories)
        for trajectory in trajectories:
            total_reward += trajectory['rewards']
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
    policy.close_sess()
    val_func.close_sess()
    print('Average Reward: ' + total_reward/episode)


def add_disc_sum_rew(trajectories, gamma):
    for trajectory in trajectories:
        if gamma < 0.999:
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
        if gamma < 0.999:
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return observes, actions, advantages, disc_sum_rew
