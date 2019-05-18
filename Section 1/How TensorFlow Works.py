import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

my_tensor = tf.zeros([1,20])

sess.run(my_tensor)

my_var = tf.Variable(tf.zeros([1,20]))

sess.run(my_var.initializer)
sess.run(my_var)

row_dim = 2
col_dim = 3

zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))

sess.run(zero_var.initializer)
sess.run(ones_var.initializer)
print(sess.run(zero_var))
print(sess.run(ones_var))

zero_similar = tf.Variable(tf.zeros_like(zero_var))
ones_similar = tf.Variable(tf.ones_like(ones_var))
sess.run(ones_similar.initializer)
sess.run(zero_similar.initializer)
print(sess.run(ones_similar))
print(sess.run(zero_similar))

fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))
sess.run(fill_var.initializer)
print(sess.run(fill_var))

const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))
sess.run(const_var.initializer)
sess.run(const_fill_var.initializer)
print(sess.run(const_var))
print(sess.run(const_fill_var))

linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end
sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end
sess.run(linear_var.initializer)
sess.run(sequence_var.initializer)
print(sess.run(linear_var))
print(sess.run(sequence_var))

rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
runif_var = tf.random_uniform([row_dim, col_dim], minval=0, maxval=4)
print(sess.run(rnorm_var))
print(sess.run(runif_var))

ops.reset_default_graph()
sess = tf.Session()
my_var = tf.Variable(tf.zeros([1,20]))
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs", graph=sess.graph)
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)
