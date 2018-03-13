
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

__author__ = 'assafarbelle'

class DataSets(object):
    """
    Wrapper to CSVSegReader for train, test, val
    """

    def __init__(self, filenames, base_folder='.', image_size=(5000, 12, 1), num_threads=4,
                 capacity=5000, min_after_dequeue=1000, num_epochs=None):

    # def __init__(self, filenames, base_folder='.', image_size=(64,64,1), num_threads=4,
    #              capacity=5000, min_after_dequeue=1000, num_epochs=None):
        data = {}
        for file in filenames:
            # data[file] = CSVPngReader([os.path.join(base_folder, "%s.csv" % file)], base_folder, image_size, num_threads, capacity, min_after_dequeue, num_epochs)
            data[file] = CSVPngReader([os.path.join('/home/yehu/Desktop/new/nonPHIData/', "%s.csv" % file)], base_folder, image_size, num_threads, capacity, min_after_dequeue, num_epochs)
            # data[file] = CSVSegReader([os.path.join(base_folder, "%s.csv" % file)], base_folder, image_size, num_threads, capacity, min_after_dequeue, num_epochs)
        
        self.data = data

class CSVPngReader(object):  #CSVSegReader(object):

    def __init__(self, filenames, base_folder='.', image_size=(5000,12,1), num_threads=4,
                 capacity=5000, min_after_dequeue=1000, num_epochs=None):
        """
        CSVSegReader is a class that reads csv files containing paths to input image and segmentation image and outputs
        batchs of correspoding image inputs and segmentation inputs.
         The inputs to the class are:

            filenames: a list of csv files filename
            num_epochs: the number of epochs - how many times to go over the data
            image_size: a tuple containing the image size in Y and X dimensions
            num_threads: number of threads for prefetch
            capacity: capacity of the shuffle queue, the larger capacity results in better mixing
            min_after_dequeue: the minimum example in the queue after a dequeue op. ensures good mixing
        """
        self.reader = tf.TextLineReader(skip_header_lines=0)
        self.input_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        self.image_size = image_size
        self.batch_size = None
        self.num_threads = num_threads
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.base_folder = base_folder

    def _get_image(self):
        _, records = self.reader.read(self.input_queue)
        print('records',records)
        # print('self.input_queue',self.input_queue)
        # print('records',records)

        file_names = tf.decode_csv(records, [tf.constant([],tf.string), tf.constant([],tf.float32)], field_delim=None, name=None)
        # file_names = tf.decode_csv(records, [tf.constant([],tf.string), tf.constant([],tf.string)], field_delim=None, name=None)

        # im_raw = tf.read_file(self.base_folder+file_names[0])
        # seg_raw =tf.read_file(self.base_folder+file_names[1])

        ecg_raw = tf.read_file(self.base_folder+file_names[0])
        k_raw = file_names[1]
        print('CSVPngReader   ecg_raw.get_shape().as_list()',ecg_raw.get_shape().as_list())

        ecg_image = tf.reshape(tf.cast(tf.image.decode_png(
                        ecg_raw, channels=1, dtype=tf.uint16),
                        tf.float32, ), self.image_size, name='input_image')
        potassium = k_raw
        # potassium = tf.reshape(
        #                 tf.cast(tf.image.decode_png(
        #                                             k_raw,
        #                                             channels=1, dtype=tf.uint8),
        #                 tf.float32,), self.image_size, name='input_seg')
        print('!!before!!ecg_image.get_shape().as_list()', ecg_image.get_shape().as_list())
        # ecg_image = (ecg_image-10000)/1000
        # ecg_image = (ecg_image[:,6:,:]-10000)/1000
        print('!!after!!ecg_image.get_shape().as_list()', ecg_image.get_shape().as_list())
        return ecg_image, potassium

    def get_batch(self, batch_size=1):

        self.batch_size = batch_size

        ecg_image, potassium = self._get_image()
        image_batch, k_batch = tf.train.shuffle_batch([ecg_image, potassium], batch_size=self.batch_size,
                                                        num_threads=self.num_threads,
                                                        capacity=self.capacity,
                                                        min_after_dequeue=self.min_after_dequeue)
        return image_batch, k_batch

        ######################################
        #
        # sess = tf.Session()
        #
        # TYPE = np.float64
        #
        # N = 1000000
        # # data = np.random.normal(0, 1, N).astype(TYPE)
        # Truncate data to make it harder
        # data = data[(data > -1) & (data < 5)]
        ##################################################