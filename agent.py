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
    def __init__(self, capacity: int = 200) -> None:
        self.memory = deque([], maxlen=capacity)
    
    def __len__(self) -> int:
        return len(self.memory)
    
    def append(self, state, action, reward, done, next_state) -> None:
        """Добавить опыт в память
        
        """
        state = torch.tensor(state.__array__())
        next_state = torch.tensor(next_state.__array__())
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        if torch.cuda.is_available():
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()
            next_state = next_state.cuda()

        self.memory.append(Experience(state, action, reward, done, next_state))
    
    def sample(self, batch_size: int) -> Tuple:
        """Создать сэмпл

        """
        indices = random.sample(self.memory, batch_size)
        states, actions, rewards, dones, next_states = map(torch.stack, zip(*indices))
        actions = actions.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()

        # if torch.cuda.is_available():
        #     states = states.cuda()
        #     actions = actions.cuda()
        #     rewards = rewards.cuda()
        #     dones = dones.cuda()
        #     next_states = next_states.cuda()
        
        return states, actions, rewards, dones, next_states


class Agent:
    def __init__(self, env: gym.Env, save_dir: Path = "checkpoints") -> None:
        self.env = env
        self.save_dir = save_dir
        self.replay_buffer = ReplayBuffer()
        self.state_shape = self.env.reset().shape
        self.action_shape = self.env.action_space.n
        self.batch_size = 32
        self.net = DQN(self.state_shape, self.action_shape)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e4
        self.gamma = 0.9 # Discount rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.02)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e4 # Min experiences before training
        self.learn_every = 3
        self.sync_every = 1e4

    def reset(self) -> None:
        self.state = self.env.reset()

    def act(self, state) -> int:
        """Используя нейросеть решаем какое действие следует выполнить используя epsilon-greedy политику 
        """
        # Агент исследует среду и выполняет случайное действие
        if np.random.random() < self.exploration_rate:
            action = self.env.action_space.sample()
        
        # Агент использует использует накопленный опыт для принятия решения о следующем действии
        else:
            state = torch.tensor(state.__array__())

            if torch.cuda.is_available():
                state = state.cuda()
            
            state = state.unsqueeze(0)
            action_values = self.net(state, model="main")
            action = torch.argmax(action_values, axis=1).item()
        
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step +=1

        return action

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

