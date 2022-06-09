from logger import MetricLogger
import datetime
import gym
import torch
from pathlib import Path
from agent import Agent, Experience
from env_wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from gym.wrappers import FrameStack

def main():

    env = gym.make("ALE/Breakout-v5", render_mode="human").unwrapped
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    save_dir.mkdir(parents=True)
    

    agent = Agent(env, save_dir)
    logger = MetricLogger(save_dir)

    episodes = 100000
    for e in range(episodes):
        state = env.reset()

        while True:
            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.append(state, action, reward, done, next_state)

            q, loss = agent.learn()

            logger.log_step(reward, loss, q)

            state = next_state

            if done:
                break
        env.close()
        
        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
    





if __name__ == "__main__":
    main()




