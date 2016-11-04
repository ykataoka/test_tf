# coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# example 1
const1 = tf.constant(2)
const2 = tf.constant(3)
add_op = tf.add(const1, const2)

with tf.Session() as sess:
    result = sess.run(add_op)
    print(result)
# results in 5


# example 2
const1 = tf.constant(2)
const2 = tf.constant(3)
add_op = tf.add(const1, const2)

print(add_op)
# results in Tensor("Add_1:0", shape=(), dtype=int32), no number
# the tensorflow's object is operation. Unless run it, the value is not returned.


# example 3
const1 = tf.constant(2)
const2 = tf.constant(3)
add_op = tf.add(const1, const2)
mul_op = tf.mul(add_op, const2)

with tf.Session() as sess:
    result1, result2 = sess.run([add_op, mul_op])
    print(result1, result2)
# this returns 5, 15


# example 4 - variable (node)
var1 = tf.Variable(0) # variable declaration
const2 = tf.constant(3)
add_op = tf.add(var1, const2)
update_var1 = tf.assign(var1, add_op) # substitution
mul_op = tf.mul(add_op, update_var1)
	
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables()) # initialization?
  print(sess.run([mul_op])) # [9]
  print(sess.run([mul_op])) # [36]
  print(sess.run([mul_op, mul_op])) # [81, 81]


# example 5 - placeholder (node without initialization)
var1 = tf.Variable(0)
holder2 = tf.placeholder(tf.int32) # type is declared but not initialized
add_op = tf.add(var1, holder2)
update_var1 = tf.assign(var1, add_op)
mul_op = tf.mul(add_op, update_var1)  

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    result = sess.run(mul_op, feed_dict={ # insert 5 here
        holder2: 5
    })
    print(result)


# example 6 - graph visualization
const1 = tf.constant(2)
const2 = tf.constant(3)
add_op = tf.add(const1, const2)
mul_op = tf.mul(add_op, const2)

with tf.Session() as sess:
    result, result2 = sess.run([mul_op, add_op])
    print(result)
    print(result2)
    tf.train.SummaryWriter('./', sess.graph)
    # this output file which can rady by tensorboard
    # $ tensorboard -logdir='./'
   
