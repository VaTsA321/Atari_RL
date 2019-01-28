import tensorflow as tf
import numpy as np
from collections import deque

class DQN:
    def __init__(self, learning_rate, gamma, decay_rate, action_size, state_size, name = ''):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.action_size = action_size
        self.state_size = state_size

        ##Model
        with tf.variable_scope(name):
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions = tf.placeholder(tf.int32, [None, action_size], name='actions')
            one_hot_actions = tf.one_hot(self.actions, action_size)
            
            #Conv Layer 1
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                                          filters=32,
                                          kernel_size=[8,8],
                                          strides=(4,4),
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv1'
                                         )
            self.relu1 = tf.nn.relu(self.conv1, name='relu1')

            # Conv Layer 2
            self.conv2 = tf.layers.conv2d(inputs=self.relu1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=(2, 2),
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2'
                                         )
            self.relu2 = tf.nn.relu(self.conv2, name='relu2')

            # Conv Layer 3
            self.conv3 = tf.layers.conv2d(inputs=self.relu1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=(1, 1),
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv3'
                                         )
            self.relu3 = tf.nn.relu(self.conv2, name='relu3')

            #FC Layer 1
            self.flatten = tf.layers.flatten(self.relu3, name='flatten')
            self.fc1 = tf.layers.dense(inputs=self.flatten,
                                       units=512,
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="fc1")
            self.output = tf.layers.dense(inputs=self.fc1,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=action_size,
                                          activation=None,
                                          name='output'
                                         )

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]


class Agent:
    def __init__(self, 
        action_size,  
        state_size=[84,84,4],
        batch_size=32,
        train_frequency=4,
        discount_factor=0.99, 
        explore_probability=1,
        explore_probability_stop=0.1, 
        decay_rate=1e-4, 
        reward_stop=0, 
        learning_rate=0.00025 
    ):
        
        self.dqn = DQN(learning_rate, discount_factor, decay_rate, action_size, state_size, 'dqn')
        self.target_network = DQN(learning_rate, discount_factor, decay_rate, action_size, state_size, 'target_network')
        self.replay_buffer = Memory(1000000)
        self.epsilon = explore_probability

    def get_next_action(self, state, training, sess):
        if training:
            if np.random.random_sample() < self.epsilon:
                #Take random action
                return np_random.randint(0, action_size)
            else:
                #Take greedy action
                Qs = sess.run(self.dqn, feed_dict={self.dqn.inputs: state.reshape((1, *state.shape))})
                action = np.argmax(Qs)
                return action
        else:
            #Take greedy action
            Qs = sess.run(self.dqn, feed_dict={self.dqn.inputs: state.reshape((1, *state.shape))})
            action = np.argmax(Qs)
            return action

    def add_to_buffer(self, experience):
        self.replay_buffer.add(experience)

    def copy_dqn2target():
        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqn")
        
        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_network")

        op_holder = []
        
        # Update our target_network parameters with DQNNetwork parameters
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def train(self, sess, summary_writer, write_op, step):
        # Sample mini-batch from memory
        batch = self.replay_buffer.sample(batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        dones = np.array([each[4] for each in batch])

        next_Qs = sess.run(self.target_network.output, feed_dict={sel.dqn.inputs: next_states})
        targets = []

        for i in range(len(batch)):
            terminal = dones[i]
            if terminal:
                targets.append(rewards[i])
            else:
                target = rewards[i] + self.gamma * np.max(next_Qs[i])
                targets.append(target)

        loss, _ = sess.run([self.dqn.loss, self.dqn.optimizer],
                            feed_dict={self.dqn.inputs: states,
                                       self.dqn.target_Q: targets,
                                       self.dqn.actions: actions})

        summary = sess.run(write_op, feed_dict={self.dqn.inputs: states,
                                        self.dqn.target_Q: targets,
                                        self.dqn.actions: actions})
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

        

