import tensorflow as tf

'''
a = tf.add(3, 5)
print(a)

sess = tf.Session()
print(sess.run(a))
sess.close()

with tf.Session() as sess:
    print(sess.run(a))

'''

# Distributed Graph.
with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name = 'a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name = 'b')
    c = tf.multiply(a, b)
    #c = tf.math.multiply(a, b)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    print(sess.run(c))

