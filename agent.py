from collections import deque, namedtuple
from pathlib import Path
import random
import gym
from typing import Tuple
import numpy as np
import torch
from network import DQN

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "new_state"])

class ReplayBuffer:
    """ReplayBuffer используется для хранения предыдущего опыта, для того чтобы агент учился на нем
    
    """
    def __init__(self, device, capacity: int = int(1e6)) -> None:
        self.memory = deque([], maxlen=capacity)
        self.device = device
    
    def __len__(self) -> int:
        return len(self.memory)
    
    def append(self, states, actions, rewards, dones, next_states) -> None:
        """Добавить опыт в память
        
        """

        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.memory.append(Experience(state, action, reward, done, next_state))
    
    def sample(self, batch_size: int) -> Tuple:
        """Создать сэмпл

        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, dones, next_states = map(np.stack, zip(*batch))

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        return states_t, actions_t, rewards_t, dones_t, next_states_t


class Agent:
    def __init__(self, env: gym.Env, device, save_dir: Path = "checkpoints", checkpoint=None) -> None:
        self.env = env
        self.device = device
        self.save_dir = save_dir
        if not checkpoint:
            self.replay_buffer = ReplayBuffer(device)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.batch_size = 32
        self.net = DQN(self.state_shape, self.action_shape)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        if checkpoint:
            self.load(checkpoint)
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999977
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5
        self.gamma = 0.99 # Discount rate
        self.burnin = 5e4 # Min experiences before training
        self.learn_every = 3
        self.sync_every = 1e4 // 4

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-5)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def reset(self) -> None:
        self.state = self.env.reset()

    def act(self, state) -> int:
        """Используя нейросеть решаем какое действие следует выполнить используя epsilon-greedy политику 
        """
        # Агент исследует среду и выполняет случайное действие
        
        # Агент использует использует накопленный опыт для принятия решения о следующем действии
        obses_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.net(obses_t, model="main")

        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= self.exploration_rate:
                actions[i] = random.randint(0, self.action_shape - 1)
            

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step +=1

        return actions

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="main")[
            np.arange(0, self.batch_size), action
            ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="main")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
            ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_main(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.main.state_dict())
    
    def save(self) -> None: 
        save_path = (
            self.save_dir / f"model{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Model saved to {save_path} at step {self.curr_step}")
    
    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
    
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every !=0:
            return None, None
        
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.batch_size)

        td_est = self.td_estimate(states, actions)

        td_tgt = self.td_target(rewards, next_states, dones)

        loss = self.update_Q_main(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    

    

    # @torch.no_grad()
    # def play_step(self, net: torch.nn.Module) -> Tuple[float, bool]:
    #     """Выполняем одно взаимодействие между агентом и средой
    #     """

    #     action = self.get_action(net, self.exploration_rate)
    #     new_state, reward, done, info = self.env.step(action)
    #     exp = Experience(self.state, action, reward, done, new_state)
    #     self.replay_buffer.append(exp)

    #     self.state = new_state

    #     if done:
    #         return self.reset()

    #     return reward, done

