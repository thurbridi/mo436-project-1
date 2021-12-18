import numpy as np
import random
import gym
import time
import itertools
from numpy.random.mtrand import gamma
import torch
from torch import nn
from torch._C import dtype
from torch.optim import Adam
from torch.distributions import Categorical


class ExperienceBuffer:
    def __init__(self, buffer_size, gamma, lambd):
        self.obs_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.value_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.return_buffer = np.zeros(buffer_size, dtype=np.float32)

        self.gamma, self.lambd = gamma, lambd

        self.max_size = buffer_size
        self.pointer, self.traj_start = 0, 0

    def push(self, obs, action, reward, value, logp):
        assert self.pointer < self.max_size
        self.obs_buffer[self.pointer] = obs
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logp_buffer[self.pointer] = logp

        self.pointer += 1

    def get_data(self):
        assert self.pointer == self.max_size

        self.pointer, self.traj_start = 0, 0

        self.adv_buffer = (self.adv_buffer -
                           self.adv_buffer.mean()) / (self.adv_buffer.std() + 1e-8)

        data = dict(
            obs=self.obs_buffer,
            action=self.action_buffer,
            reward=self.reward_buffer,
            value=self.value_buffer,
            logp=self.logp_buffer,
            adv=self.adv_buffer,
            returns=self.return_buffer,
        )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def compute_trajectory_values(self, last_value=0):
        trajectory = slice(self.traj_start, self.pointer)

        rewards = np.append(self.reward_buffer[trajectory], last_value)
        values = np.append(self.value_buffer[trajectory], last_value)

        # GAE-Lambda
        deltas = - values[:-1] + rewards[:-1] + values[1:] * self.gamma
        weights = (
            self.gamma * self.lambd) ** np.array(range(len(deltas)), dtype=np.float32)
        self.adv_buffer[trajectory] = np.array([np.sum(
            deltas[i:] * weights[:len(deltas)-i]) for i, _ in enumerate(deltas)], dtype=np.float32)

        # Rewards-to-go
        weights = self.gamma ** np.array(range(len(rewards)), dtype=np.float32)
        self.return_buffer[trajectory] = np.array(
            [np.sum(rewards[i:] * weights[:len(rewards)-i]) for i, _ in enumerate(rewards)], dtype=np.float32)[:-1]

        self.traj_start = self.pointer


def mlp(layer_sizes, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:
            layers.append(activation())
        else:
            layers.append(output_activation())

    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    def __init__(self, input_dim, n_actions, layer_sizes, activation=nn.Tanh):
        super(ActorNet, self).__init__()

        self.actor = mlp([input_dim] + list(layer_sizes) +
                         [n_actions], activation)

    def forward(self, obs, action=None):
        logits = self.actor(obs)
        pi = Categorical(logits=logits)

        logp_a = pi.log_prob(action) if (action is not None) else None

        return pi, logp_a

    def step(self, obs):
        with torch.no_grad():
            pi = Categorical(logits=self.actor(obs))
            action = pi.sample()

            logp_a = pi.log_prob(action)

        return action.cpu().numpy(), logp_a.cpu().numpy()


class CriticNet(nn.Module):
    def __init__(self, input_dim, layer_sizes, activation=nn.Tanh):
        super(CriticNet, self).__init__()

        self.critic = mlp([input_dim] + list(layer_sizes) + [1], activation)

    def forward(self, obs):
        value = torch.squeeze(self.critic(obs), -1)
        return value

    def step(self, obs):
        with torch.no_grad():
            value = torch.squeeze(self.critic(obs), -1)

        return value.cpu().numpy()


def PPO_clip(env: gym.Env, actor, critic, actor_lr, critic_lr, epochs, steps_per_epoch, gamma, lambd, clip_ratio):
    ep_returns = []
    ep_lens = []

    buffer = ExperienceBuffer(
        steps_per_epoch,
        gamma,
        lambd
    )

    policy_optimizer = Adam(actor.actor.parameters(), lr=actor_lr)
    value_optimizer = Adam(critic.critic.parameters(), lr=critic_lr)

    def compute_loss_actor(data):
        obs, action, adv, logp_old = data['obs'].cuda(), data['action'].cuda(), data['adv'].cuda(
        ), data['logp'].cuda()
        _, logp = actor(obs, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv

        return -(torch.min(ratio * adv, clip_adv)).mean()

    def compute_loss_critic(data):
        obs, ret = data['obs'].cuda(), data['returns'].cuda()
        value = critic(obs)
        return ((value - ret)**2).mean()

    def update():
        data = buffer.get_data()

        loss_actor_old = compute_loss_actor(data).item()
        loss_critic_old = compute_loss_critic(data).item()

        for i in range(10):
            policy_optimizer.zero_grad()
            loss_actor = compute_loss_actor(data)
            loss_actor.backward()
            policy_optimizer.step()

        for i in range(10):
            value_optimizer.zero_grad()
            loss_critic = compute_loss_critic(data)
            loss_critic.backward()
            value_optimizer.step()

        print(f'Delta loss actor: {loss_actor.item() - loss_actor_old:.4f}')
        print(f'Delta loss critic: {loss_critic.item() - loss_critic_old:.4f}')

    obs = env.reset()
    ep_ret, ep_len = 0, 0

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for t in range(steps_per_epoch):
            # env.render()
            action, logp = actor.step(
                torch.as_tensor([obs], dtype=torch.float32).cuda())
            value = critic.step(torch.as_tensor(
                [obs], dtype=torch.float32).cuda())

            next_obs, reward, done, _ = env.step(action.item())

            ep_ret += reward
            ep_len += 1

            buffer.push(obs, action, reward, value, logp)

            obs = next_obs

            epoch_ended = t == steps_per_epoch - 1

            if done or epoch_ended:
                if epoch_ended and not done:
                    print('Trajectory cut off by epoch')

                    value = critic.step(
                        torch.as_tensor([obs], dtype=torch.float32).cuda())
                else:
                    value = 0

                buffer.compute_trajectory_values(value)
                if done:
                    ep_returns.append(ep_ret)
                    ep_lens.append(ep_len)

                obs = env.reset()
                ep_ret, ep_len = 0, 0

        update()

    return ep_returns, ep_lens


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.3)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--lambd', type=float, default=0.97)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    np.random.seed(777)
    env = gym.make('FrozenLake-v1', is_slippery=False)

    obs_dim = env.observation_space.shape

    obs_dim = 1
    actions_dim = env.action_space.n

    hidden_layers = [64, 128, 64]

    actor = ActorNet(obs_dim, actions_dim, hidden_layers).cuda()
    critic = CriticNet(obs_dim, hidden_layers).cuda()

    start = time.time()
    ep_returns, ep_lens = PPO_clip(
        env,
        actor, critic,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        gamma=args.gamma,
        lambd=args.lambd,
        clip_ratio=args.clip_ratio,
    )
    end = time.time()

    import plotly.express as px
    fig = px.line(ep_returns)
    fig.show()

    print(f'Algorithm took: {end-start} seconds')
