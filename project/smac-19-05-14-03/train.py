from __future__ import absolute_import, division, print_function

import argparse
from types import SimpleNamespace as SN
from typing import Deque, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from IPython import embed
import random

from smac.env import StarCraft2Env

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


Q_MIN = -1e10


class Memory(Deque):

    def __init__(self, capacity):
        super().__init__(maxlen=capacity)

    def sample(self, n_samples):
        return random.sample(self, n_samples)
        

class Agent(object):
    def __init__(self, args, n_agents, input_shape, n_actions):
        self.args = args
        self.n_agents = n_agents
        self.model = AgentModel(args, input_shape, n_actions)
        self.hidden_state = None

    def init_hidden(self, batch_size):
        self.hidden_state = self.model.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def act(self, obs, batch_size, n_agents):
        obs = obs.view(batch_size * n_agents, -1)
        qs, self.hidden_state = self.model(obs, self.hidden_state)
        return qs.view(batch_size, n_agents, -1)


class AgentModel(nn.Module):
    """
    에이전트 신경망
    """
    def __init__(self, args, input_shape, n_outputs):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, n_outputs)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h       


class VDNMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, agent_qs, states):
        return torch.sum(agent_qs, dim=2, keepdim=True)


class QMixer(nn.Module):
    def __init__(self, args, n_agents, state_shape):
        super().__init__()

        self.args = args
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


def main(args):

    # 환경 초기화
    env = StarCraft2Env(
        map_name=args.map_name, 
        window_size_x=800, 
        window_size_y=600)
    env_info = env.get_env_info()

    obs_shape = env_info['obs_shape']
    state_shape = env_info['state_shape']
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    episode_limit = env_info['episode_limit']

    # 인공신경망
    agent = Agent(args, n_agents, obs_shape, n_actions)
    parameters = list(agent.model.parameters())
    target_agent = Agent(args, n_agents, obs_shape, n_actions)
    target_agent.model.load_state_dict(agent.model.state_dict())

    if args.mixer == 'vdn':
        mixer = VDNMixer()
        parameters += list(mixer.parameters())
        target_mixer = VDNMixer()
        target_mixer.load_state_dict(mixer.state_dict())
    elif args.mixer == 'qmix':
        mixer = QMixer(args, n_agents, state_shape)
        parameters += list(mixer.parameters())
        target_mixer = QMixer(args, n_agents, state_shape)
        target_mixer.load_state_dict(mixer.state_dict())
    else:
        mixer = None
    optimizer = optim.Adam(parameters, lr=args.lr)

    # 버퍼
    replay_memory = Memory(capacity=args.memory_capacity)

    # 학습 변수 초기값 설정
    epsilon = 1.0

    # tensorboard summary writer
    writer = SummaryWriter(args.outs)

    for e in range(args.max_episodes):
        env.reset()
        done = False
        score = 0
        # state_stack.clear()
        episode = Memory(capacity=episode_limit) 
        agent.init_hidden(batch_size=1)
        test_game = True if e % args.test_interval == 0 else False
        q_values = list()

        while not done:
            obs = env.get_obs()
            state = env.get_state()

            with torch.no_grad():
                obs_t = torch.tensor(obs).to(torch.float32)
                qs_t = agent.act(obs_t, 1, n_agents).view(n_agents, -1)
                
                random_actions = list()
                action_mask = list()
                for agent_id in range(n_agents):
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    random_action = np.random.choice(avail_actions_ind)
                    random_actions.append(random_action)
                    action_mask.append(avail_actions)
                action_mask_t = torch.Tensor(action_mask).float()

                masked_qs_t = action_mask_t * qs_t + (1. - action_mask_t) * Q_MIN

                qs_t, actions_t = masked_qs_t.max(dim=1)
                q_values.append(qs_t.numpy())
                best_actions = actions_t.numpy()
                
                if test_game:
                    actions = best_actions
                else:
                    actions = np.where(
                        np.random.random(n_agents) < epsilon,
                        random_actions,
                        best_actions
                    )

                # print(env.get_avail_agent_actions(0))

            reward, done, info = env.step(actions)
            score += reward

            # 데이터 저장
            state_action = SN(
                obs=obs, state=state, 
                action_mask=action_mask, actions=actions, 
                reward=reward, done=done, info=info)
            episode.append(state_action)

        # 게임 종료
        replay_memory.append(episode)
        print(f"게임결과: {e} = {score}, test: {test_game}, e: {epsilon}")
        if test_game:
            writer.add_scalar('score/test', score, e)
            writer.add_scalar('score/q_value', np.mean(q_values), e)
        else:
            writer.add_scalar('score/train', score, e)

        # 학습
        if len(replay_memory) > args.batch_size:
            obs_buffer = torch.zeros(episode_limit, args.batch_size * n_agents, obs_shape)
            state_buffer = torch.zeros(episode_limit, args.batch_size, state_shape)
            action_mask_buffer = torch.zeros(episode_limit, args.batch_size * n_agents, n_actions)
            actions_buffer = torch.zeros(episode_limit, args.batch_size * n_agents, 1).to(torch.long)
            rewards_buffer = torch.zeros(episode_limit, args.batch_size * n_agents, 1)
            state_mask_buffer = torch.zeros(episode_limit, args.batch_size * n_agents, 1)

            samples = replay_memory.sample(args.batch_size)

            for episode_id in range(args.batch_size):
                for t in range(len(samples[episode_id])):
                    state_buffer[t, episode_id, :] = torch.Tensor(samples[episode_id][t].state)
                    for agent_id in range(n_agents):
                        obs_buffer[t, episode_id * n_agents + agent_id, :] = torch.Tensor(samples[episode_id][t].obs[agent_id])
                        action_mask_buffer[t, episode_id * n_agents + agent_id, :] = torch.Tensor(samples[episode_id][t].action_mask[agent_id])
                        actions_buffer[t, episode_id * n_agents + agent_id] = int(samples[episode_id][t].actions[agent_id])
                        rewards_buffer[t, episode_id * n_agents + agent_id] = float(samples[episode_id][t].reward)
                        state_mask_buffer[t, episode_id * n_agents + agent_id] = 1. - float(samples[episode_id][t].done)
                        

            rewards_buffer = rewards_buffer.squeeze()
            state_mask_buffer = state_mask_buffer.squeeze()

            # target Q-value 계산
            with torch.no_grad():
                q2s = list()
                target_agent.init_hidden(batch_size=args.batch_size)
                for t in range(1, episode_limit):
                    outs = target_agent.act(obs_buffer[t], args.batch_size, n_agents)
                    outs = outs.view(args.batch_size * n_agents, -1)
                    action_mask = action_mask_buffer[t]
                    outs = action_mask * outs + (1. - action_mask) * Q_MIN
                    q2s.append(outs)
                q2s = torch.stack(q2s, dim=1)
                q2s = q2s.transpose_(0, 1)  # (119, 64, 14)
                q2max, _ = q2s.max(dim=2)

            # Q-value 계산
            q1s = list()
            agent.init_hidden(batch_size=args.batch_size)
            for t in range(episode_limit - 1):
                outs = agent.act(obs_buffer[t], args.batch_size, n_agents)
                outs = outs.view(args.batch_size * n_agents, -1)
                q1s.append(outs)
            q1s = torch.stack(q1s, dim=1)
            q1s = q1s.transpose_(0, 1)  # (119, 64, 14)
            q1s = torch.gather(q1s, dim=2, index=actions_buffer[:-1, :]).squeeze()

            parameters = list(agent.model.parameters())
            if mixer is not None:
                parameters += list(mixer.parameters())
                q1s = mixer(
                    q1s.view(-1, args.batch_size, n_agents), 
                    state_buffer[:-1, :, :])
                q2max = target_mixer(
                    q2max.view(-1, args.batch_size, n_agents), 
                    state_buffer[1:, :, :])

                q1s = q1s.squeeze()
                q2max = q2max.squeeze()
                rewards = rewards_buffer[:-1, :].view(-1, args.batch_size, n_agents)[:, :, 0]
                state_masks = state_mask_buffer[1:, :].view(-1, args.batch_size, n_agents)[:, :, 0]
            else:
                rewards = rewards_buffer[:-1, :]
                state_masks = state_mask_buffer[1:, :]

            target_qs = rewards + args.gamma * state_masks * q2max

            # 오차 계산
            q_loss = (target_qs.detach() - q1s) ** 2
            q_loss_mean = q_loss.mean()

            loss_dict = dict(polcy_loss=q_loss_mean.item())
            print(f'학습결과: {loss_dict}')
            writer.add_scalar('loss/q_loss', q_loss_mean.item(), e)

            # 최적화
            optimizer.zero_grad()
            q_loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()

            # epsilon 업데이트
            epsilon = max(args.min_epsilon, epsilon - args.epsilon_delta)
            writer.add_scalar('params/epsilon', epsilon, e)

            # target model 업데이트
            for target, param in zip(target_agent.model.parameters(), agent.model.parameters()):
                target.data.copy_(target.data * (1.0 - args.soft_tau) + param.data * args.soft_tau)

            if mixer is not None:
                for target, param in zip(target_mixer.parameters(), mixer.parameters()):
                    target.data.copy_(target.data * (1.0 - args.soft_tau) + param.data * args.soft_tau)

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outs', type=str, default='../smac_outs/')
    parser.add_argument('--map_name', type=str, default='8m')
    parser.add_argument('--memory_capacity', type=int, default=5000)
    parser.add_argument('--max_episodes', type=str, default=100000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--epsilon_delta', type=int, default=0.001)
    parser.add_argument('--min_epsilon', type=int, default=0.05)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--test_interval', type=int, default=10)
    parser.add_argument('--ddpn', action='store_true', default=False)
    parser.add_argument('--soft_tau', type=float, default=0.2)
    parser.add_argument('--mixer', choices=['none', 'vdn', 'qmix'], default='none')
    parser.add_argument('--mixing_embed_dim', type=int, default=32)
    args = parser.parse_args()

    main(args)