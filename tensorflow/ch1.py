import tensorflow as tf

mat1 = tf.constant([[3, 3]])
mat2 = tf.constant([[2], [2]])

print(mat1)
print(mat2)

product = tf.matmul(mat1, mat2)

sess = tf.Session()

result = sess.run(product)

print (result)

sess.close()


with tf.Session() as sess:
  result = sess.run(product)
  print(result)

