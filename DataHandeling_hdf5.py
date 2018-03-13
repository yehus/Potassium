
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# from random import shuffle
# import glob
# import numpy
# from random import shuffle
#############################################


from __future__ import print_function
from math import ceil
import pandas
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import tftables
import tensorflow as tf
from random import shuffle

seed=2018
np.random.seed(seed)

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'dataset.hdf5'  # address to where you want to save the hdf5 file
AllData_path = 'ActivDataFilteredCliped/'
# AllData_path = 'ActivDataFiltered/'
# AllData_path = 'ActivData/'

dataseton = 1
if dataseton == 0:
    # dataframe = pandas.read_csv("train.csv", delim_whitespace=True, header=None)
    dataframe = pandas.read_csv("train.csv", header=None)
    print('dataframe: ', dataframe)
    train_addrs = AllData_path + dataframe[0]
    train_labels = dataframe[1]
    print('train_addrs:', train_addrs)
    print('len train_addrs:', len(train_addrs))
    print('train_labels:', train_labels)
    print('lentrain_labels:', len(train_labels))

    dataframe = pandas.read_csv("val.csv", header=None)
    val_addrs = AllData_path + dataframe[0]
    val_labels = dataframe[1]
    print('val_addrs:', val_addrs)
    print('len val_addrs:', len(val_addrs))
    print('val_labels:', val_labels)
    print('len val_labels:', len(val_labels))

    dataframe = pandas.read_csv("test.csv", header=None)
    test_addrs = AllData_path + dataframe[0]
    test_labels = dataframe[1]
    print('test_addrs:', test_addrs)
    print('len test_addrs:', len(test_addrs))
    print('test_labels:', test_labels)
    print('len test_labels:', len(test_labels))

    '''To store images,
    we should define an array for each of train, validation and test sets
    with the shape of (number of data, image_height, image_width, image_depth) in Tensorflow order or
    (number of data, image_height, image_width, image_depth) in Theano order.
    For labels we also need an array for each of train, validation and test sets
    with the shape of (number of data).
    Finally, we calculate the pixel-wise mean of the train set
    and save it in an array with the shape of (1, image_height, image_width, image_depth).
    Note that you always should determine the type of data (dtype)
    when you want to create an array for it.
    '''
    ###############################################

    # Create a HDF5 file

    '''
    tables:In tables we can use create_earray which create an empty array
    (number of data=0)and we can append data to it later.
    For labels, it is more convenient here to use create_array
    as it lets us to write the lables when we are creating the array.
    To set the dtype of an array, you can use tables dtype such as tables.UInt8Atom() for uint8.
    The first attribute of create_earray and create_array methods is the data group
    (we create the arrays in root group) which lets you to manage your data
    by creating different data groups.
    You can consider groups as somethings like folders in your HDF5 file.

    h5py: in h5py we create an array using create_dataset.
    Note that we should determine the exact size of array when you are defining it.
    We can use the create_dataset for labels as well and immediately put the labels on it.
    You can set the dtype of an array directly using numpy dypes.'''

    ###############################################

    #####################################################

    # h5py

    train_shape = (len(train_addrs), 5000, 12, 1)
    val_shape = (len(val_addrs), 5000, 12, 1)
    test_shape = (len(test_addrs), 5000, 12, 1)

    print('train_shape:', train_shape)
    print('val_shape:', val_shape)
    print('test_shape:', test_shape)

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(hdf5_path, mode='w')

    hdf5_file.create_dataset("train_img", train_shape, np.uint16)
    hdf5_file.create_dataset("val_img", val_shape, np.uint16)
    hdf5_file.create_dataset("test_img", test_shape, np.uint16)

    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

    hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.float32)
    hdf5_file["train_labels"][...] = train_labels
    hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.float32)
    hdf5_file["val_labels"][...] = val_labels
    hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.float32)
    hdf5_file["test_labels"][...] = test_labels

    print('hdf5_file:', hdf5_file)
    print('len  hdf5_file:', len(hdf5_file))

    # Now, it's time to read images one by one, apply preprocessing (only resize in our code) and then save it.

    ###############################################

    # h5py

    # a numpy array to save the mean of the images
    mean = np.zeros(train_shape[1:], np.float32)

    # loop over train addresses
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if i % 100 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, len(train_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]

        # print('train_addrs' , train_addrs)
        # print('addr' , addr)
        # img = addr
        img = cv2.imread(addr, cv2.IMREAD_UNCHANGED)
        # img = cv2.imread(addr, -1)
        # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # np.reshape(img,(1,5000,12,1))
        # print(img, addr)
        img.resize(img.shape[0], img.shape[1], 1)
        # add any image pre-processing here
        # print('train' , i)
        # print('shape  img' , np.shape(img))

        # if the data order is Theano, axis orders should change
        # if data_order == 'th':
        #     img = np.rollaxis(img, 2)

        # save the image and calculate the mean so far
        hdf5_file["train_img"][i, ...] = img[None]
        # hdf5_file["train_img"][i, ...] = img[None]
        mean += img / float(len(train_labels))

    # loop over validation addresses
    for i in range(len(val_addrs)):
        # print how many images are saved every 1000 images
        if i % 100 == 0 and i > 1:
            print('Validation data: {}/{}'.format(i, len(val_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = val_addrs[i]

        # print('addr', addr)
        # img = addr
        img = cv2.imread(addr, cv2.IMREAD_UNCHANGED)
        # img = cv2.imread(addr, -1)
        # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.resize((img.shape[0], img.shape[1], 1))
        # print('val' , i)

        # add any image pre-processing here

        # if the data order is Theano, axis orders should change
        # if data_order == 'th':
        #     img = np.rollaxis(img, 2)

        # save the image
        hdf5_file["val_img"][i, ...] = img[None]

    # loop over test addresses
    for i in range(len(test_addrs)):
        # print how many images are saved every 1000 images
        if i % 100 == 0 and i > 1:
            print('Test data: {}/{}'.format(i, len(test_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = test_addrs[i]

        # print('addr', addr)
        # img = addr
        img = cv2.imread(addr, cv2.IMREAD_UNCHANGED)
        # img = cv2.imread(addr, -1)
        # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.resize((img.shape[0], img.shape[1], 1))
        # print('test' , i)

        # add any image pre-processing here

        # if the data order is Theano, axis orders should change
        # if data_order == 'th':
        #     img = np.rollaxis(img, 2)

        # save the image
        hdf5_file["test_img"][i, ...] = img[None]

    # save the mean and close the hdf5 file
    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()

###############################################

'''
Read the HDF5 file

It's time to check if the data is saved properly in the HDF5 file.
To do so, we load the data in batchs of an arbitrary size and plot the first image
of the first 5 batchs.
We also check the label of each image.
We define a variable, subtract_mean,
which indicates if we want to subtract mean of the training set before showing the image.
In tables we access each array calling its name after its data group
(like this hdf5_file.group.arrayname).
You can index it like a numpy array.
However, in h5py we access an array using its name like a dictionary name
(hdf5_file["arrayname""]).
In either case, you have access to the shape of the array through .shape like a numpy array.
'''

# import h5py
# import numpy as np

# hdf5_path = 'Cat vs Dog/dataset.hdf5'
subtract_mean = False

# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")

# subtract the training mean
if subtract_mean:
    mm = hdf5_file["train_mean"][0, ...]
    mm = mm[np.newaxis, ...]

# Total number of samples
data_num = hdf5_file["train_img"].shape[0]

print('hdf5_file', hdf5_file)
print('data_num', data_num)
print('hdf5_file["train_img"]', hdf5_file["train_img"])

# ################################################
#
# '''
# Now we create a list of batches indeces and shuffle it.
# Now, we loop over batches and read all images in each batch at once.
#
# '''
# batch_size = 10
# # nb_class=2
# #
# # from random import shuffle
# # from math import ceil
# # import matplotlib.pyplot as plt
#
# # create list of batches to shuffle the data
# batches_list = list(range(int(ceil(float(data_num) / batch_size))))
# # shuffle(batches_list)
# print('batches_list:', batches_list)
# print('batches_list.shape:', len(batches_list))
#
# # loop over batches
# # for counter, value in enumerate(some_list):
# #     print(counter, value)
# # for counter, value in enumerate(some_list):
# for n, i in enumerate(batches_list):
#     i_s = i * batch_size  # index of the first image in this batch
#     i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
#
#     # read batch images and remove training mean
#     images = hdf5_file["train_img"][i_s:i_e, ...]
#     if subtract_mean:
#         images -= mm
#
#     # read labels and convert to one hot encoding
#     labels = hdf5_file["train_labels"][i_s:i_e]
#     print('labels:', labels)
#     print('n, i:', n, i)
#     # print('labels:', labels)
#     # labels_one_hot = np.zeros((batch_size, nb_class))
#     # labels_one_hot[np.arange(batch_size), labels] = 1
#
#     print('n+1, len(batches_list):', n + 1, '/', len(batches_list))
#     print('labels[0], labels_one_hot[0, :]:', labels[0])  # , labels_one_hot[0, :])
#     print('images[0]:', images[0])
#     print('len(images[0]):', images[0].shape)
#
#     im0 = images[0]#(images[0] - 10000) / 1000
#     print('im0    :', im0)
#     lb0 = labels[0]
#     leads_im = im0[:, :, 0]
#     print('leads_im    :', leads_im)
#     print('mean im0    :', np.max(leads_im, axis=0))
#     n_leads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#     n_subplots = 12
#     pltfig1 = n
#     plt.figure(pltfig1)  # , figsize=(20, 20))
#     # plt.title("All filters ECG, origin leads 'upper'")
#     n_columns = 2
#     n_rows = 6
#     for k in range(n_subplots):
#         plt.subplot(n_rows, n_columns, k + 1)
#         plt.title('ECG, Lead ' + str(n_leads[k]) + ', potassium=' + str(lb0))  # +'-'+train_addrs[i_s])
#         plt.plot(leads_im[:, n_leads[k] - 1])  # , 'r')  # plotting t, a separately
#         # plt.plot(leads_im[0:2500, n_leads[k] - 1])  # , 'r')  # plotting t, a separately
#         # addr = train_addrs[i]
#         # plt.colorbar(LeadsIm, orientation='horizontal')
#     plt.savefig('Leads 1-12 ECG' + str(n) + ', potassium=' + str(lb0) + '.png')
#
#     # plt.imshow(im0[:,:,0])
#     # plt.imshow(images[0][:,:,0])
#     # plt.show()
#
#     if n == 5:  # break after 5 batches
#         break
#
#         # hdf5_file.close()
#
#         ############################################
#         #
#         #
#         #
#         #
#         #
#         #
#         #########################################
# # Quick   start
# # An example of accessing a table in a HDF5 file.
#
# import tftables
# import tensorflow as tf
# with tf.device('/cpu:0'):
#     # This function preprocesses the batches before they
#     # are loaded into the internal queue.
#     # You can cast data, or do one-hot transforms.
#     # If the dataset is a table, this function is required.
#     def input_transform(tbl_batch):
#         labels = tbl_batch['label']
#         data = tbl_batch['data']
#
#         truth = tf.to_float(tf.one_hot(labels, num_labels, 1, 0))
#         data_float = tf.to_float(data)
#
#         return truth, data_float
#
#     # Open the HDF5 file and create a loader for a dataset.
#     # The batch_size defines the length (in the outer dimension)
#     # of the elements (batches) returned by the reader.
#     # Takes a function as input that pre-processes the data.
#     loader = tftables.load_dataset(filename='path/to/h5_file.h5',
#                                    dataset_path='/internal/h5/path',
#                                    input_transform=input_transform,
#                                    batch_size=20)
#
# # To get the data, we dequeue it from the loader.
# # Tensorflow tensors are returned in the same order as input_transformation
# truth_batch, data_batch = loader.dequeue()
#
# # The placeholder can then be used in your network
# result = my_network(truth_batch, data_batch)
#
# with tf.Session() as sess:
#
#     # This context manager starts and stops the internal threads and
#     # processes used to read the data from disk and store it in the queue.
#     with loader.begin(sess):
#         for _ in range(num_iterations):
#             sess.run(result)
# #######################################################
'''
If the dataset is an array instead of a table.Then input_transform can
be omitted if no pre - processing is required.
If only a single pass through the dataset is desired, then
you should pass cyclic = False to load_dataset.
A slightly more involved example showing how to access multiple datasets in one
HDF5 file, as well as the full API.
'''
# reader = tftables.open_file(filename='path/to/h5_file',
#                             batch_size = 20)
reader = tftables.open_file(filename='/home/yehu/Desktop/new/nonPHIData/dataset.hdf5',
                            batch_size=10)
print('reader    :', reader)
'''
# Accessing a single array
# Suppose you only want to read a single array from your HDF5 file.
# Doing this is quite straight-forward.
# Start by getting a tensorflow placeholder for your batch from reader.
# X_train = hdf5_file["train_img"]
# Y_train = hdf5_file["train_labels"]
#'''

array_batch_placeholder = reader.get_batch(
    path = '/train_labels',#''/h5/path',  # This is the path to your array inside the HDF5 file.
    cyclic = True,      # In cyclic access, when the reader gets to the end of the
                        # array, it will wrap back to the beginning and continue.
    ordered = False     # The reader will not require the rows of the array to be
                        # returned in the same order as on disk.
)
print('array_batch_placeholder    :', array_batch_placeholder)

# You can transform the batch however you like now.
# For example, casting it to floats.
array_batch_float = tf.to_float(array_batch_placeholder)

# The data can now be fed into your network
result = my_network(array_batch_float)

with tf.Session() as sess:
    # The feed method provides a generator that returns
    # feed_dict's containing batches from your HDF5 file.
    for i, feed_dict in enumerate(reader.feed()):
        sess.run(result, feed_dict=feed_dict)
        if i >= N:
            break

# Finally, the reader should be closed.
reader.close()
# Note that be default, the ordered argument to get_batch is set to True.
# If you require the rows of the array to be returned in the same order as they are on disk,
# then you should leave it as ordered = True.
# However, this may result in a performance penalty.
# In machine learning, rows of a dataset often represent independent examples, or data points.
# Thus their ordering is not important.

###############################################################
###############################################################

# # For tables and compound data types, a dictionary is returned.
# table_batch_dict = reader.get_batch('/internal/h5_path/to/table')
# # The keys for the dictionary are taken from the column names of the table.
# # The values of the dictionary are the corresponding placeholders for the batch.
# col_A_pl, col_B_pl = table_batch_dict['col_A'], table_batch_dict['col_B']
#
# # You can access multiple datasets within the HDF5 file.
# # They all share the same batchsize, and are fed into your
# # graph simultaneously.
# labels_batch = reader.get_batch('/my_label_array')
# truth_batch = tf.one_hot(labels_batch, 2, 1, 0)
#
# # This class creates a Tensorflow FIFOQueue and populates it with data from the reader.
# loader = tftables.FIFOQueueLoader(reader, size=2,
# # The inputs are placeholders (or graphs derived thereof) from the reader.
#     inputs=[col_A_pl, col_B_pl, truth_batch])
# # Batches are taken out of the queue using a dequeue operation.
# dequeue_op = loader.dequeue()
#
# # The dequeued data can then be used in your network.
# result = my_network(dequeue_op)
#
# with tf.Session() as sess:
#     # The queue loader needs to be started inside your session
#     loader.start(sess)
#
#     # Then simply run your operation, data will be streamed
#     # out of the HDF5 file and into your graph!
#     for _ in range(N):
#         sess.run(result)
#
#     # Finally, the queue should be stopped.
#     loader.stop(sess)
# reader.close()
# #######################################################
# #######################################################
# #######################################################
# # read images
# X_train = hdf5_file["train_img"]
# Y_train = hdf5_file["train_labels"]
#
# X_val = hdf5_file["val_img"]
# Y_val = hdf5_file["val_labels"]
#
# X_test = hdf5_file["test_img"]
# Y_test = hdf5_file["test_labels"]
#
# print('Split train: ',X_train[1], len(X_train))
# print('Split valid: ',X_val[1], len(X_val))
# print('Split holdout: ',X_test[1], len(X_test))
# # hdf5_file.close()
# ############################################
