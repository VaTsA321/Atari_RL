import gym
import tensorflow as tf
from wrappers import wrap_deepmind
from agent import Agent
import time

env = gym.make('Pong-v4')
env = wrap_deepmind(env, frame_stack=True, scale=True)
action_size = env.action_space.n

# Reset the graph
tf.reset_default_graph()

#Create our agent
agent = Agent(action_size)

count = 0
with tf.Session() as sess:
    total_test_rewards = []

    saver = tf.train.Saver()
    # Load the model
    saver.restore(sess, "./model.ckpt")

    for episode in range(1):
        total_rewards = 0

        state = env.reset()

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            count += 1

            action = agent.get_next_action(state, False, sess)

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            #print('obs = ', next_state)
            env.render()
            time.sleep(0.02)

            total_rewards += reward

            if done:
                break

            state = next_state

    env.close()
