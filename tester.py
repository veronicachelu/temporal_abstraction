import tensorflow as tf

sess = tf.InteractiveSession()
option = tf.constant([0, 1, 3, 2, 4])
q_val = tf.constant([10, 11, 12])
nb_options = 3
condition = option >= nb_options
rez = tf.where(condition, q_val, tf.zeros_like(q_val))
print("dadas")
