import gym
import tensorflow as tf
import numpy as np
from wrappers import wrap_deepmind, make_atari
from agent import Agent
from collections import deque

def main():
    env = gym.make('Pong-v4')
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    action_size = env.action_space.n
    
    #High level hyperparameters
    pretrain_length=50000
    reward_stop = 0
    max_steps_per_episode = 10000
    train_freq = 4
    max_copy_steps = 10000

    #Reward tracking
    total_episode_reward = 0
    avg_reward_10 = -21 #average reward over last 10 episodes
    episode_rewards = []
    last_10_rewards = deque(maxlen=10)

    # Reset the graph
    tf.reset_default_graph()

    #Create our agent
    agent = Agent(action_size)

    # Saver will help us to save our model
    saver = tf.train.Saver()

    #Setup summaries for tensorboard
    tf.summary.scalar("Loss", agent.dqn.loss)
    tf.summary.scalar("Qmax", agent.dqn.Qmax)
    #tf.summary.tensor_summary("Q", agent.dqn.Q)

    #Reset the env
    state = env.reset()

    for i in range(pretrain_length):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        if done:
            next_state = np.zeros(state.shape)
            agent.add_to_buffer((state, action, reward, next_state, done))
            state = env.reset()
        else:
            agent.add_to_buffer((state, action, reward, next_state, done))
            state = next_state

    #Create agent
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("/output/tensorboard")
        write_op = tf.summary.merge_all()

        # Initialize the decay rate (that will use to reduce epsilon)
        total_steps = 0
        decay_step = 0
        copy_steps = 0
        episode = 0
        step = 0
        

        while avg_reward_10 < reward_stop:
            #Take action and add to mem
            while step < max_steps_per_episode:
                copy_steps += 1
                total_steps += 1

                action = agent.get_next_action(state=state, training=True, sess=sess)
                next_state, reward, done, _ = env.step(action)
                episode_rewards.append(reward)
                if done:
                    total_episode_reward = np.sum(episode_rewards)
                    last_10_rewards.append(total_episode_reward)
                    if episode > 9:
                        avg_reward_10 = np.sum(last_10_rewards) / 10
                    print('{{"metric": "avg_reward_per_episode", "value": {}}}'.format(avg_reward_10))
                    print('{{"metric": "episode_reward", "value": {}}}'.format(total_episode_reward))
                    print('{{"metric": "steps_per_episde", "value": {}}}'.format(step))
                    print('{{"metric": "explore_probability", "value": {}}}'.format(agent.epsilon))
                    print('episode = ', episode)
                    next_state = np.zeros(state.shape)
                    agent.add_to_buffer((state, action, reward, next_state, done))
                    state = env.reset()
                    step = max_steps_per_episode
                    episode_rewards = []
                else:
                    agent.add_to_buffer((state, action, reward, next_state, done))
                    state = next_state

                #Train with experience replay
                if total_steps % train_freq == 0:
                    agent.train(sess, writer, write_op, total_steps)

                if copy_steps > max_copy_steps:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = agent.copy_dqn2target()
                    sess.run(update_target)
                    copy_steps = 0
                    print("Model updated")

                step += 1

            #Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./model/model.ckpt")
                print("Model Saved")

            #Increment episode count
            episode += 1
            step = 0

            

if __name__ == '__main__':
    main()
