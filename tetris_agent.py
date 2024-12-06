
from nes_py.wrappers import JoypadSpace
import gymnasium as gym
from gym_tetris.actions import MOVEMENT
import ale_py
import numpy as np
from tensorflow import keras
from tensorflow.keras import utils
from tf_keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tf_keras.optimizers.legacy import Adam
from tf_keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
from rl.core import Processor
from PIL import Image


gym.register_envs(ale_py)

# Initialise the environment
env = gym.make("ALE/Tetris-ram-v5", render_mode="human",full_action_space=True)

nb_actions = env.action_space.n
IMG_SHAPE = (84, 84)
WINDOW_LENGTH = 12
input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(100):
    # this is where you would insert your policy
    action = env.action_space.sample() # random action at the moment 
   
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()

class ImageProcessor:
    def process_observations(self, observation):
        IMG_SHAPE = (84, 84)

        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE)
        img = img.convert("L")
        img = np.array(img)

        return img.astype('uint8')
    
    def process_state_batch(self, batch):
        processed_batch = batch / 255.0
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)

def build_model(nb_actions):
    input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])
    model = Sequential()

    model.add(Permute((2,3,1), input_shape=input_shape))

    model.add(Convolution2D(32, (8,8), strides=(4, 4), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, (8,8), strides=(4, 4), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, (8,8), strides=(4, 4), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    return model
    
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0,
                              value_min=0.1,
                              value_test=0.05,
                              nb_steps=1000) # 500000

model = build_model()

memory = SequentialMemory(limit=500000, window_length=WINDOW_LENGTH)

processor = ImageProcessor()

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               processor=processor,
               nb_steps_warmup=1000, # 500000 ??????
               gamma = 0.99, # ???
               target_model_update=1000,
               train_interval=12,
               delta_clip=1)

dqn.compile(Adam(learning_rate=0.25), metrics=['mae'])

checkpoint_filename = "AI DATA"

checkpoint_callback = ModelIntervalCheckpoint(checkpoint_filename, interval=100)#1000
