import tensorflow as tf


# O# TODO Build the policy gradient neural network
class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

        # Neural net starts here

        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)

        # Output of neural net
        out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        # Training Procedure
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

        self.loss = tf. reduce_mean(cross_entropy * self.rewards)
        self.gradient = tf.gradients(self.loss, tf.trainable())
        # Create a placeholder list for gradients
        self.gradients_to_apply = []
        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        # Create the operation to update gradients with the gradients placeholder.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))