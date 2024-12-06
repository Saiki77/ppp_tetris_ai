from nes_py.wrappers import JoypadSpace
import gymnasium as gym
import gym_tetris
from gym_tetris.actions import MOVEMENT
import ale_py

gym.register_envs(ale_py)

# Initialise the environment
env = gym.make("ALE/Tetris-v5", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(5000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
