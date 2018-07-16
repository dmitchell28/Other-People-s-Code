import tensorflow as tf
'''
# TODO: Clear the tensorflow graph
tf.reset_default_graph()

test_constant = tf.constant(10.0, dtype=tf.float32)
add_one_operation = test_constant + 1
double_operation = input_data * 2

# TODO: Run a session
input_data = tf.placeholder(dtype=tf.float32, shape=None)
with tf.Session() as sess:
    print(sess.run(add_one_operation))
'''

# TODO: Create placeholders
tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 2])

double_operation = input_data * 2

# TODO: Run a session
with tf.Session() as sess:
    print(sess.run(double_operation, feed_dict={input_data:[[1,2],[3,4]]}))