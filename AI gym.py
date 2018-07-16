import gym
import tensorflow as tf
import numpy as np
import matplotlib as pl

import os
# TODO: Load an environment
env = gym.make("CartPole-v1")

print(env.observation_space)
print(env.action_space)

# TODO Make a random agent
games_to_play = 10

for i in range(games_to_play):
    # Reset the environment
    obs = env.reset()
    episode_rewards = 0
    done = False

    while not done:
        # Render the environment so we can watch
        env.render()

        # Choose a random action
        action = env.action_space.sample()

        # Take a step in the environment with the chosen action
        obs, reward, done, info = env.step(action)
        episode_rewards += reward

    # Print episode total rewards when done
    print(episode_rewards)

# Close the environment
#env.close()
#TODO Build the policy gradient neural network

class Agent:
    def__init__(self,num_action,state_size):
    initializer = tf.contrib.layers.xavier_initializer()

    self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
    hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
    hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)

    # Output of neural net
    out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)
    self.outputs = tf.nn.softmax(out)
    self.choice = tf.argmax(self.outputs, axis=1)
