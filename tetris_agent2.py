import gym
import numpy as np
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
from rl.core import Processor
from PIL import Image

train = True

if train:
    env = gym.make("ALE/Tetris-v5")
else:
    env = gym.make("ALE/Tetris-v5", render_mode='human')

nb_actions = env.action_space.n

IMG_SHAPE = (84, 84)
WINDOW_LENGTH = 12
input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

class ImageProcessor(Processor):
    def process_observation(self, observation):
        IMG_SHAPE = (84, 84)

        # Check if observation is a tuple
        if isinstance(observation, tuple):
            observation = observation[0]  # Extract the first element (modify if necessary)

        # Ensure the observation is a NumPy array
        observation = np.array(observation)

        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE)
        img = img.convert("L")
        img = np.array(img)

        return img.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch / 255.0
        return processed_batch

    def process_reward(self, reward):               # Muss wahrscheinlich raus
        return np.clip(reward, -1.0, 1.0)


def build_the_model(input_shape, nb_actions=5):
    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (2, 2), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    return model


# Rest of the code remains the same
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0,
                              value_min=0.1,
                              value_test=0.05,
                              nb_steps=1000000)

model = build_the_model(input_shape, nb_actions)

memory = SequentialMemory(limit=500000, window_length=WINDOW_LENGTH)

processor = ImageProcessor()

checkpoint_filename = "ParametersAI.hf5"

checkpoint_callback = ModelIntervalCheckpoint(checkpoint_filename, interval=1000)

try:
    model.load_weights(checkpoint_filename)
    print("Model loaded")
except:
    print("No model loaded")

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    policy=policy,
    memory=memory,
    processor=processor,
    nb_steps_warmup=100000,
    gamma=.99,
    target_model_update=1000,
    train_interval=32,
    delta_clip=1)

dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

if train:
    metrics = dqn.fit(env, nb_steps=2000000, callbacks=[checkpoint_callback], log_interval=10000, visualize=False)
    dqn.test(env, nb_episodes=1, visualize=True)
    env.close()
    model.summary()
else:
    dqn.test(env, nb_episodes=1, visualize=True)
    env.close()
