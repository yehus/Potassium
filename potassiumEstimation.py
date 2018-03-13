import tensorflow as tf
import numpy as np
import ops  # Ops is a file with operations. Currently only conv layer implementation
import batch_norm  # batch normalization

EPS = 1e-5
IMAGE_SIZE = 5000*12
# IMAGE_SIZE = 5000*6
SEED = 66478  # Set to None for random seed.
NUM_OUT = 1

# two defs for tf.summary.histogram
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
# https://www.tensorflow.org/get_started/tensorboard_histograms


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class PotassiumEstimation(object):
# class CellSegmentation(object):
    """
    Cell segmentation model class
    PotassiumEstimation model class
    """
    def __init__(self, input=None, labels=None, dims_in=None, dims_out=None, regularization_weight=None, name=None):
        """
        :param input: data set images
        :param labels: data set labels
        :param dims_in: list input image size, for example [64,64,1] (W,H,C)
        :param dims_out: list output image size, for example [64,64,1] (W,H,C)
        :param regularization_weight: L2 Norm reg weight
        :param name: model name, used for summary writer sub-names (Must be unique!)
        """
        self.input = input
        self.labels = labels
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.regularization_weight = regularization_weight
        self.base_name = name

    def model(self, train_phase):
        """
        Define the model - The network architecture
        :param train_phase: tf.bool with True for train and False for test
        """
        # Reshape the input for batchSize, dims_in[0] X dims_in[1] image, dims_in[2] channels
        x_image = tf.reshape(self.input, [-1, self.dims_in[0], self.dims_in[1], self.dims_in[2]],
                             name='x_input_reshaped')
        # Dump input image
        tf.summary.image(self.get_name('x_input'), x_image)
        # tf.image_summary(self.get_name('x_input'), x_image)
        print('model  -  x_image:', x_image)
        print('model  -  self.input:', self.input)
        print('model  -  self.dims_in:', self.dims_in)

        # Model convolutions
        kh = 1                           #averagepool
        kw = 1
        dh = 1
        dw = 1
        wind = [1,4,1,1]
        stride = [1,2,2,1]
        with tf.variable_scope('conv_1'):
            conv_1, reg1 , weights1, biases1 = ops.conv2d(x_image, output_dim=64, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_1")
            # conv_1, reg1 , weights1, biases1 = ops.conv2d(x_image, output_dim=64, k_h=1, k_w=12, d_h=dh, d_w=dw, name="conv_1")
            relu1 = tf.nn.relu(conv_1)
            max_pool1 = tf.nn.max_pool(relu1, wind, stride, 'SAME')
            print('if train_phase == True: ', train_phase)
            if train_phase == True:
                tf.get_variable_scope().reuse_variables()    # tf.get_variable_scope().reuse_variables
                tf.summary.histogram("weights1", weights1)   # tf.histogram_summary("weights1", weights1)
                tf.summary.histogram("biases1", biases1)     # tf.histogram_summary("biases1", biases1)
                tf.summary.merge_all()                      # tf.merge_all_summaries()
                print('if train_phase == True:  weights1.get_shape()',weights1.get_shape())

                # def histogram(conv):
                #     return tf.summary.histogram("hist_conv_bn", conv) #tf.histogram_summary("hist_conv_bn", conv)
                #
                # train_phase1 = tf.constant(train_phase, dtype=tf.bool)
                # conv_1 = batch_norm.batch_normalization(conv_1, n_out=128, train_phas=train_phase1, scope='bn1')
                # histogram(conv_1)
                # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyees')

        with tf.variable_scope('conv_2'):
            # conv_2, reg2, weights2, biases2 = ops.conv2d(relu1, output_dim=32, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_2")
            # conv_2, reg2, weights2, biases2 = ops.conv2d(relu1, output_dim=64, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_2")
            conv_2, reg2, weights2, biases2 = ops.conv2d(max_pool1, output_dim=64, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_2")
            # if train_phase == True:
            #     train_phase1 = tf.constant(train_phase, dtype=tf.bool)
            #     conv_2 = batch_norm.batch_normalization(conv_2, n_out=96, train_phas=train_phase1, scope='bn2')
            relu2 = tf.nn.relu(conv_2)
            max_pool2 = tf.nn.max_pool(relu2, wind, stride, 'SAME')

        with tf.variable_scope('conv_3'):
            # conv_3, reg3, weights3, biases3 = ops.conv2d(relu2, output_dim=16, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_3")
            # conv_3, reg3, weights3, biases3 = ops.conv2d(relu2, output_dim=64, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_3")
            conv_3, reg3, weights3, biases3 = ops.conv2d(max_pool2, output_dim=64, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_3")
            # if train_phase == True:
            #     train_phase1 = tf.constant(train_phase, dtype=tf.bool)
            #     conv_3 = batch_norm.batch_normalization(conv_3, n_out=64, train_phas=train_phase1, scope='bn3')
            relu3 = tf.nn.relu(conv_3)#+x_image)

        with tf.variable_scope('conv_4'):
            conv_4, reg4, weights4, biases4 = ops.conv2d(relu3, output_dim=64, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_4")
            # conv_4, reg4, weights4, biases4 = ops.conv2d(relu3, output_dim=32, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_4")
            # if train_phase == True:
            #     train_phase1 = tf.constant(train_phase, dtype=tf.bool)
            #     conv_4 = batch_norm.batch_normalization(conv_4, n_out=32, train_phas=train_phase1, scope='bn4')
            relu4 = tf.nn.relu(conv_4)

            # repool = tf.image.resize_bilinear(relu4, [5000, 12], align_corners=None, name=None)

        '''
        print('relu_shape')
        print(relu_shape)
        print('relu_3')
        print(relu_3)
        reshape = tf.reshape(
            relu_3,
            [relu_shape[0], relu_shape[1] * relu_shape[2] * relu_shape[3]])

        print('reshape')
        print(reshape)

        fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE * 64, NUM_CLASSES],
                                stddev=0.1,
                                seed=None,
                                dtype= tf.float32))
        print('!!11111!!')
        print(fc1_weights)

        fc1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype = tf.float32))
        predict = tf.matmul(reshape,fc1_weights)+fc1_biases
        '''
        with tf.variable_scope('conv_5'):
            conv_5, reg5, weights5, biases5 = ops.conv2d(relu4, output_dim=64, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_5")
            # conv_5, reg5, weights5, biases5 = ops.conv2d(repool, output_dim=1, k_h=kh, k_w=kw, d_h=dh, d_w=dw, name="conv_5")
            print ('!!conv_5', conv_5.get_shape().as_list())
            relu5 = tf.nn.relu(conv_5)
            # ###########################################
            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
        relu5_shape = relu5.get_shape().as_list()

        print ('!!relu5_shape', relu5_shape)

        reshape = tf.reshape(
            relu5,
            [relu5_shape[0], relu5_shape[1] * relu5_shape[2] * relu5_shape[3]])

        print('!!!!!!!!!reshape',reshape)
        print('!!!!!!!!!reshape.get_shape().as_list()',reshape.get_shape().as_list())

        # predict = tf.nn.avg_pool(relu5, ksize=[1,2,2,1], strides=[1, 1, 1, 1], padding='SAME')
        # # h = tf.nn.avg_pool(h, ksize=[1, 4, 4, 256], strides=[1, 1, 1, 1], padding='VALID')
        # predict = tf.reduce_mean(predict, axis=[1, 2])
        ##########################################################################
        # good
        reshape_shape = reshape.get_shape().as_list()

        with tf.variable_scope('fully_connected_1'):
            fc1_weights = tf.Variable(  # fully connected, depth 512.IMAGE_SIZE*2
                tf.truncated_normal([ reshape_shape[1], NUM_OUT],
                                    stddev=0.1,
                                    seed=SEED,
                                    dtype= tf.float32))


            fc1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_OUT], dtype = tf.float32))
            predict = tf.nn.relu(tf.matmul(reshape,fc1_weights)+fc1_biases)
            # predict = tf.nn.softmax(predict, name="softmax_tensor")
            # predict_shape = predict.get_shape().as_list()
            print('!!11111!!fc1_weights', fc1_weights)
            print('!!!!predict_shape after fc1 predict.get_shape().as_list()', predict.get_shape().as_list())

            # # Add a 50% dropout during training only. Dropout also scales
            # # activations such that no rescaling is needed at evaluation time.
            # if train_phase == True:
            #     predict = tf.nn.dropout(predict, 0.4, seed=SEED)
        ##########################################################################

            # with tf.variable_scope('fully_connected_2'):
        #     fc2_weights = tf.Variable(tf.truncated_normal([predict_shape[1], NUM_OUT ],
        #                                                   stddev=0.1,
        #                                                   seed=SEED,
        #                                                   dtype=tf.float32))
        #     fc2_biases = tf.Variable(tf.constant(
        #         0.1, shape=[NUM_OUT], dtype=tf.float32))
        #     predict = tf.nn.relu(tf.matmul(predict, fc2_weights) + fc2_biases)
        #
        #     print('!!11111!!predict after fc2', predict)
            ##########################################################################


        # # predict = tf.nn.relu(tf.matmul(predict,fc1_weights)+fc1_biases)
        # # predict = tf.nn.softmax(predict, name="softmax_tensor") #NUM_OUT

        ##########################################################################
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.

        # fc1_weights = tf.Variable(  # fully connected, depth 512.IMAGE_SIZE*2
        #     tf.truncated_normal([ reshape_shape[1], NUM_OUT],
        #                         stddev=0.1,
        #                         seed=SEED,
        #                         dtype= tf.float32))
        #
        # print('!!11111!!fc1_weights', fc1_weights)
        # fc1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_OUT], dtype = tf.float32))
        # predict = tf.nn.relu(tf.matmul(reshape,fc1_weights)+fc1_biases)
        # # predict = tf.nn.softmax(tf.matmul(reshape,fc1_weights)+fc1_biases, name="softmax_tensor")
        # predict_shape = predict.get_shape().as_list()
        #
        # fc2_weights = tf.Variable(tf.truncated_normal([2048, 1],
        #                                               stddev=0.1,
        #                                               seed=SEED,
        #                                               dtype=tf.float32))
        # fc2_biases = tf.Variable(tf.constant(
        #     0.1, shape=[2048], dtype=tf.float32))
        # predict = tf.matmul(predict, fc2_weights) + fc2_biases

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
#        if train_phase == True:
#            reshape = tf.nn.dropout(reshape, 0.5, seed=SEED)

        # predict = tf.matmul(reshape,fc1_weights)+fc1_biases

        # predict = tf.nn.relu(tf.matmul(reshape,fc1_weights)+fc1_biases)
        # predict = tf.nn.softmax(tf.matmul(reshape,fc1_weights)+fc1_biases, name="softmax_tensor")

        # hidden = tf.nn.relu(tf.matmul(reshape,fc1_weights)+fc1_biases)

        # fc2_weights = tf.Variable(tf.truncated_normal([2048, NUM_OUT],
        #                                               stddev=0.1,
        #                                               seed=SEED,
        #                                               dtype=tf.float32))
        # fc2_biases = tf.Variable(tf.constant(
        #     0.1, shape=[2048], dtype=tf.float32))
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
        # hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        # predict = tf.matmul(hidden, fc2_weights) + fc2_biases

        ##########################################################################

        # predict = x_image #tf.squeeze(relu5, 1) #relu5
        predict = tf.squeeze(predict, 1) #relu5

        reg = reg1+reg2+reg3+reg4+reg5
        print('model - predict', predict)
        print('model - conv_1', conv_1)
        print('model - conv_2', conv_2)
        print('model - conv_3', conv_3)
        print('model - conv_4', conv_4)
        print('model - conv_5', conv_5)
        print('model - relu1', relu1)
        print('model - relu2', relu2)
        print('model - relu3', relu3)
        print('model - relu4', relu4)
        print('model - relu5', relu5)
        print('model - reg', reg)

        #################################################
        # Reshape the output layer to a 1-dim Tensor to return predictions
        # predict = tf.squeeze(conv_5, 1)
        # reg = tf.squeeze(reg1+reg2+reg3+reg4+reg5, 1)
        # predictions = tf.squeeze(output_layer, 1)
        ################################################
        return predict, reg, conv_1

    def loss(self, predict, reg=None):
        """
        Return loss value
        :param predict: prediction from the model
        :param reg: regularization
        :return:
        """
        labels_image = tf.cast(self.labels, tf.float32, name='y_input_reshape')
        # labels_image = tf.reshape(tf.cast(self.labels, tf.float16), [-1, self.dims_out[0], self.dims_out[1], self.dims_out[2]], name='y_input_reshape')
        # labels_image1 = tf.reshape(tf.cast(self.labels, tf.float32), [-1, self.dims_out], name='y_input_reshape')
        print('loss - labels_image',labels_image)
        print('loss - predict',predict)
        # tf.summary.image(self.get_name('Labels'), labels_image)
        # tf.image_summary(self.get_name('Labels'), labels_image)

        # Reshape to flatten tensors
        # predict_reshaped = tf.cast(tf.contrib.layers.flatten(predict > 0), tf.float32)
        predict_reshaped = tf.cast(predict, tf.float32)
        #### predict_reshaped = tf.cast(tf.contrib.layers.flatten(predict), tf.float32)
        # predict_reshaped = predict
        # predict_reshaped = tf.contrib.layers.flatten(predict)
        labels = labels_image
        # labels = tf.reshape(tf.cast(self.labels, tf.float32), [-1, self.dims_out[0]], name='y_input_reshape')
        # labels = tf.contrib.layers.flatten(self.labels)

        # cross = tf.nn.sigmoid_cross_entropy_with_logits(predict_reshaped,labels ,  name = 'cross')
        # loss = tf.reduce_mean(cross, name = 'loss')
        ###########################################
        print('loss - labels',labels)
        print('loss - predict_reshaped',predict_reshaped)
        print('loss - self.labels',self.labels)

        # Calculate loss using mean squared error   (average_loss)
        # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predict_reshaped))))
        # loss = tf.reduce_mean(tf.pow(labels - predict_reshaped, 2))
        loss = tf.reduce_mean(tf.squared_difference(labels, tf.transpose(predict_reshaped)))
        print('loss - loss',loss)

        ##############################################################
        # # Define loss and optimizer, minimize the squared error
        # loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        #########################################################

        # n_samples = labels.shape[0]
        # loss = tf.reduce_sum(tf.pow(predict_reshaped - labels, 2)) / (2 * n_samples)

        # loss = tf.reduce_mean(tf.squared_difference(labels, predict_reshaped))
        # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predict_reshaped))))
        # loss = tf.reduce_sum(tf.pow(labels - predict_reshaped, 2)) / (2 * 60000)  # L2 loss
        # loss = tf.reduce_sum(tf.pow(labels - predict_reshaped, 2)) / (2 * labels.__sizeof__())  # L2 loss
        # loss = tf.reduce_mean(tf.squared_difference(labels, predict_reshaped))
        # loss = tf.losses.mean_squared_error(labels, predict_reshaped)

        # size_data = numpy.asarray([2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427,
        #                            1380, 1494, 1940, 2000, 1890, 4478, 1268, 2300,
        #                            1320, 1236, 2609, 3031, 1767, 1888, 1604, 1962,
        #                            3890, 1100, 1458, 2526, 2200, 2637, 1839, 1000,
        #                            2040, 3137, 1811, 1437, 1239, 2132, 4215, 2162,
        #                            1664, 2238, 2567, 1200, 852, 1852, 1203])
        # price_data = numpy.asarray([399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999,
        #                             212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900,
        #                             299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900,
        #                             573900, 249900, 464500, 469000, 475000, 299900, 349900, 169900,
        #                             314900, 579900, 285900, 249900, 229900, 345000, 549000, 287000,
        #                             368500, 329900, 314000, 299000, 179900, 299900, 239500])
        # # Test a data set
        # size_data_test = numpy.asarray([1600, 1494, 1236, 1100, 3137, 2238])
        # price_data_test = numpy.asarray([329900, 242500, 199900, 249900, 579900, 329900])
        # def normalize(array):
        # return (array - array.mean()) / array.std()
        # # Normalize a data set
        # size_data_n = normalize(size_data)
        # price_data_n = normalize(price_data)
        # size_data_test_n = normalize(size_data_test)
        # price_data_test_n = normalize(price_data_test)
        # # Display a plot
        # plt.plot(size_data, price_data, 'ro', label='Samples data')
        # plt.legend()
        # plt.draw()
        # samples_number = price_data_n.size
        # # Minimize squared errors
        # cost_function = tf.reduce_sum(tf.pow(model - Y, 2)) / (2 * samples_number)  # L2 loss
        #
        # samples_number = price_data_n.size
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)  # Gradient descent


        # The way to implement it in TF is tf.sqrt(tf.reduce_mean(tf.squared_difference(Y1, Y2)))
        # With the same result you can minimize just
        # tf.reduce_mean(tf.squared_difference(Y1, Y2)) or even
        # tf.reduce_sum(tf.squared_difference(Y1, Y2))
        # but because they have a smaller graph of operations, they will be optimized faster.
        ############################################
        tf.summary.scalar(self.get_name('loss without regularization'), loss)

        if reg is not None:
            tf.summary.scalar(self.get_name('regulariztion'), reg)
            # tf.scalar_summary(self.get_name('regulariztion'), reg)

            # Add the regularization term to the loss.
            loss += self.regularization_weight * reg
            tf.summary.scalar(self.get_name('loss+reg'), loss)
        
        return loss


    def training(self, s_loss, learning_rate):
        """
        :param s_loss:
        :param learning_rate:
        :return:
        """
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar(self.get_name(s_loss.op.name), s_loss)
        # tf.scalar_summary(self.get_name(s_loss.op.name), s_loss)

        # Here you can change to any solver you want

        # Create Adam optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(s_loss, global_step=global_step)

        # train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(s_loss, global_step=global_step)

        # RMSProp: Divide the learning rate(lr) for a weight(w) by a running average of the magnitudes of recent gradients for that weight.
        # The idea behind applying different optimizations like(Momentum, AdaGrad, RMSProp)
        # to the gradients is that after computing the gradients, you want
        # to do some processing on them  and thenapply these processed gradients.This is
        # for better learning.
        # So, in RMSProp, you just don't use a constant learning rate throughout.
        # It is instead dependent on the "running average of magnitudes of recent gradients"
##############################################################
        # # Define loss and optimizer, minimize the squared error
        # loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
#########################################################
        return train_op

    def evaluation(self, predict, labels):
        """
        Calcualte dice score
        :param predict: predict tensor
        :param labels: labels tensor
        :return: Dice score [0,1]
        """
        ######################################################
        # # Please do not change this function
        #
        # predict = tf.cast(tf.contrib.layers.flatten(predict > 0), tf.float32)
        # labels = tf.contrib.layers.flatten(self.labels)
        #
        # # Calculate dice score
        # intersection = tf.reduce_sum(predict * labels, keep_dims=True) + EPS
        # union = tf.reduce_sum(predict, keep_dims=True) + tf.reduce_sum(labels, keep_dims=True) + EPS
        # dice = (2 * intersection) / union
        #
        # # Return value and write summary
        # ret = dice[0,0]
        # tf.scalar_summary(self.get_name("Evaluation"), ret)
        ######################################################

        # In evaluation mode we will calculate evaluation metrics.
        # predict = tf.cast(tf.contrib.layers.flatten(predict > 0), tf.float32)

        # predict = tf.contrib.layers.flatten(predict)
        # labels = tf.contrib.layers.flatten(self.labels)
        labels = self.labels
        print('evaluate  -  predict.get_shape().as_list() - ', predict.get_shape().as_list())
        print('evaluate  -  labels.get_shape().as_list() - ', labels.get_shape().as_list())
        # delta = tf.reduce_mean(tf.pow(labels - predict, 2))
        # delta = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predict))))
        # delta = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, tf.transpose(predict)))))
        # rmse1 = predict

        # rmse = tf.reduce_mean(tf.squared_difference(labels, tf.transpose(predict)))
        # rmse = 1-tf.sqrt(tf.reduce_mean(tf.square(labels - tf.transpose(predict))))/tf.reduce_mean(labels)
        # rmse = 1-tf.reduce_mean(labels - tf.transpose(predict))/tf.reduce_mean(labels)
        # rmse = tf.reduce_mean(tf.squared_difference(labels, tf.transpose(predict)))
        rmse = tf.reduce_mean(tf.abs(labels -tf.transpose(predict)))   #
        print('evaluate  -  rmse - ',rmse)

        # delta = tf.reduce_sum(tf.pow(predict - labels, 2)) / (2 * predict.shape[0])
        # delta = tf.reduce_mean(tf.subtract(labels, predict))#predict) #
        # predict = tf.cast(tf.contrib.layers.flatten(predict > 0), tf.float32)
        # labels = tf.transpose(tf.cast(self.labels, tf.float32, name='y_input_reshape'))

        # labels = tf.cast(self.labels, tf.float32, name='y_input_reshape')
        # labels = tf.reshape(tf.cast(self.labels, tf.float32), [-1, self.dims_out[0]], name='y_input_reshape')
        # labels = tf.contrib.layers.flatten(self.labels)

        # Calculate root mean+
        #
        #  squared error
        # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predict_reshaped))))

        # rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, predict)))
        # rmse = tf.clip_by_value(tf.sqrt(0.5*tf.reduce_mean(tf.square(tf.subtract(labels, predict)))), 1e-10, 1)
        # rmse = predict#labels#-predict

        # delta = tf.reduce_mean(labels-predict)

        # Return value and write summary
        # ret = rmse # delta
###        # tf.scalar_summary(self.get_name("Evaluation"), ret)

        #####################################################
        tf.summary.tensor_summary(self.get_name("Evaluation"), rmse)
        return rmse #, rmse1

    def get_name(self, name):
        """
        Get full name with prefix name
        """
        return "%s_%s" % (self.base_name, name)
