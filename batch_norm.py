import tensorflow as tf
from tensorflow.python.framework import ops
#function batch normalization


def batch_normalization(x, n_out, train_phas, scope ='bn'):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE) as scope:
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
                # print('batch_ normalization.........................')
            mean, var = tf.cond(train_phas,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

