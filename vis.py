import tensorflow as tf
#from tensorflow.python.framework import ops
##############################
import numpy as np
import matplotlib.pyplot as plt
import math
##############################
'''

git add aaa.py
git commit -a -m "messag"
git push

Now we define a couple functions that will allow us to visualize the network.
The first gets the activations at a given layer for a given input image.
The second plots those activations in a grid.
'''

def getactivations(conv1, image, sess):

    x = tf.placeholder(tf.float32, [None, 60000], name="x-in")
    x = tf.reshape(x, [-1, 5000, 12, 1])

    im = (image.eval(session=sess)-10000)/1000
    # plt.ion()
    # conv1 = conv[0, 0:64, 0:64, 0]
    print("vis - conv1:", conv1)
    print("vis - image:", image)

    #image2 = np.reshape(image, (1, 4096), order='F')
    #    np.reshape(image, (1, 4096))
    #units = sess.run(conv1,feed_dict={x:np.reshape(stimuli,[1,4096],order='F'),keep_prob:1.0})

    units = sess.run(conv1, feed_dict={x: im})
    print ('vis - units:', units[0, :, :, 0])
    print ('vis - type(units):', type(units))
    print ('vis - units.shape:', units.shape)

    # A = np.random.rand(5, 5)
    # plt.figure(1)
    # plt.imshow(A, interpolation='nearest')
    # plt.grid(True)
    print('vis - im:', im)
    print('vis - im.shape', im.shape)
    print('vis - type(im)', type(im))
#######################################
    # from numpy import *
    # import math
    # import matplotlib.pyplot as plt

    # t = linspace(0, 2 * math.pi, 400)
    # a = sin(t)
    # b = cos(t)
    # c = a + b
    #
    # plt.plot(t, a, 'r')  # plotting t, a separately
    # plt.plot(t, b, 'b')  # plotting t, b separately
    # plt.plot(t, c, 'g')  # plotting t, c separately
#########################################

    LeadsIm0 = im[0, :, :, 0]
    conv10 = units[0, :, :, 0]

    plotECG(LeadsIm0, conv10, NLeads=[1,2,3], NSubPlots = 3, pltfig1=1)
    plt.savefig('Leads 1-3 ECG and conv1.png')
    plotECG(LeadsIm0, conv10, NLeads=[4,5,6], NSubPlots = 3, pltfig1=2)
    plt.savefig('Leads 4-6 ECG and conv1.png')
    plotECG(LeadsIm0, conv10, NLeads=[7,8,9], NSubPlots = 3, pltfig1=3)
    plt.savefig('Leads 7-9 ECG and conv1.png')
    plotECG(LeadsIm0, conv10, NLeads=[10,11,12], NSubPlots = 3, pltfig1=4)
    plt.savefig('Leads 10-12 ECG and conv1.png')

    # plt.figure(1)
    # plt.plot(LeadsIm0[:, 0])
    # # plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
    # plt.xlabel('Samples')
    # plt.ylabel('ECG')
    # # plt.show()
    # # With a simple chart under our belts, now we can opt to output the chart to a file instead of displaying it (or both if desired), by using the .savefig() method.

    plot5000(LeadsIm0, conv10, NLeads=[1,2,3], NSubPlots = 3, pltfig1=5)#(LeadsIm0, 'ECG', 5)
    plt.savefig('12 Leads ECG and conv1 image.png')#ECG 12 Leads.png')
    # plot5000(LeadsIm0, conv10, NLeads=[4,5,6], NSubPlots = 3, pltfig1=6)#(conv10, 'Conv1-Filter1', 6)
    # plt.savefig('Leads 4-6 ECG and conv1 image.png')
    # plot5000(LeadsIm0, conv10, NLeads=[7,8,9], NSubPlots = 3, pltfig1=7)#(LeadsIm0, 'ECG', 5)
    # plt.savefig('Leads 7-9 ECG and conv1 image.png')#ECG 12 Leads.png')
    # plot5000(LeadsIm0, conv10, NLeads=[10,11,12], NSubPlots = 3, pltfig1=8)#(conv10, 'Conv1-Filter1', 6)
    # plt.savefig('Leads 10-12 ECG and conv1 image.png')

    # plt.figure(3)
    plotnnfilter(units, 9)
    plt.savefig('All filters in Conv1 12 Leads.png')
    plt.show()
    # plt.close('all')


def plotnnfilter(units, pltfig2):
    filters = 6#units.shape[3]
    plt.figure(pltfig2, figsize=(6,6))
    plt.title("All filters ECG, origin leads 'upper'")
    n_columns = 1#6
    n_rows = 6#math.ceil(filters / n_columns)# + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.figure(pltfig2, figsize=(20, 20))
        plt.imshow(np.transpose(units[0, 0:1000, :,i]), origin='lower',interpolation="nearest", cmap="gray")
        # plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
# def plotnnfilter(units, pltfig2):
#     filters = units.shape[3]
#     plt.figure(pltfig2, figsize=(6,6))
#     plt.title("All filters ECG, origin leads 'upper'")
#     n_columns = 6
#     n_rows = math.ceil(filters / n_columns)# + 1
#     for i in range(filters):
#         plt.subplot(n_rows, n_columns, i+1)
#         plt.title('Filter ' + str(i))
#         plt.figure(pltfig2, figsize=(20, 20))
#         plt.imshow(np.transpose(units[0, 0:300, :,i]), origin='lower',interpolation="nearest", cmap="gray")
#         # plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
################################################


def plotECG(LeadsIm, conv1, NLeads, NSubPlots, pltfig1):

    plt.figure(pltfig1, figsize=(20, 20))
    # plt.title("All filters ECG, origin leads 'upper'")
    n_columns = 1 #2
    n_rows = 6#math.ceil(NSubPlots / n_columns)# + 1

    # for i in range(NSubPlots):not good scale
    #     plt.subplot(n_rows, n_columns, i+1)
    #     plt.title('ECG, Lead ' + str(NLeads[i])+ 'ECG-Red, (Conv1-mean)-Blue')
    #     plt.plot(LeadsIm[0:5000, NLeads[i] - 1], 'r')  # plotting t, a separately
    #     # plt.subplot(n_rows, n_columns, 2 * (i + 1))
    #     # plt.title('conv1, Lead ' + str(NLeads[i]))
    #     plt.plot(conv1[0:5000, NLeads[i] - 1]-np.mean(conv1[0:5000, NLeads[i] - 1]), 'b')  # plotting t, a separately

    for i in range(NSubPlots):
        plt.subplot(n_rows, n_columns, 2*i+1)
        plt.title('ECG, Lead ' + str(NLeads[i]))
        plt.plot(LeadsIm[0:5000, NLeads[i]-1], 'r')  # plotting t, a separately


        plt.subplot(n_rows, n_columns, 2*(i+1))
        plt.title('conv1, Lead ' + str(NLeads[i]))
        plt.plot(conv1[0:5000, NLeads[i]-1], 'b')  # plotting t, a separately
    # plt.colorbar(LeadsIm, orientation='horizontal')
  # for i in range(NSubPlots-6):
  #       plt.subplot(n_rows, n_columns, 2*i+1)
  #       plt.title('ECG, Lead ' + str(NLeads[i]))
  #       plt.plot(LeadsIm[0:2500, NLeads[i]-1], 'r')  # plotting t, a separately
  #
  #
  #       plt.subplot(n_rows, n_columns, 2*(i+1))
  #       plt.title('conv1, Lead ' + str(NLeads[i]))
  #       plt.plot(conv1[0:2500, NLeads[i]-1], 'b')  # plotting t, a separately
  #   # plt.colorbar(LeadsIm, orientation='horizontal')
        ############################################


# def plot5000(LeadsIm, title1, pltfig1):
def plot5000(LeadsIm, conv1, NLeads, NSubPlots, pltfig1):
    plt.figure(pltfig1, figsize=(20, 20))
    n_rows = 6
    n_columns = 1 #2
    for i in range(NSubPlots):
        plt.subplot(n_rows, n_columns, 2 * i + 1)
        plt.title('ECG, 12 Leads, bits interval# ' + str(i*(1000)) + ' : ' + str((i+1)*1000-1))
        plt.imshow(np.transpose(LeadsIm[i*(1000):((i+1)*1000-1), :]), origin='lower', interpolation="nearest", cmap="gray")
        # plt.plot(LeadsIm[0:5000, NLeads[i] - 1], 'r')  # plotting t, a separately

        plt.subplot(n_rows, n_columns, 2 * (i + 1))
        plt.title('conv1, 12 Leads, bits interval# ' + str(i*(1000)) + ' : ' + str((i+1)*1000-1))
        plt.imshow(np.transpose(conv1[i*(1000):((i+1)*1000-1), :]), origin='lower', interpolation="nearest", cmap="gray")
        # plt.plot(conv1[0:5000, NLeads[i] - 1], 'b')  # plotting t, a separately
    # for i in range(n_rows):
    #     plt.subplot(n_rows, n_columns, i+1)
    #     plt.title(title1 + '12 Leads, bits interval# ' + str(i*(300)) + ' : ' + str((i+1)*300-1))
    #     plt.imshow(np.transpose(LeadsIm[i*(300):((i+1)*300-1), :]), origin='lower', interpolation="nearest", cmap="gray")
    # # plt.colorbar(LeadsIm, orientation='horizontal')
        ############################################
# import matplotlib
# import matplotlib.pyplot as plt

# plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
# plt.xlabel('Months')
# plt.ylabel('Books Read')
# plt.show()
# #With a simple chart under our belts, now we can opt to output the chart to a file instead of displaying it (or both if desired), by using the .savefig() method.
# plt.savefig('books_read.png')

 ##############################################
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['savefig.facecolor'] = "0.8"
#
# def example_plot(ax, fontsize=12):
#     ax.plot([1, 2])
#
#     ax.locator_params(nbins=3)
#     ax.set_xlabel('x-label', fontsize=fontsize)
#     ax.set_ylabel('y-label', fontsize=fontsize)
#     ax.set_title('Title', fontsize=fontsize)
#
# plt.close('all')
# fig, ax = plt.subplots()
# example_plot(ax, fontsize=24)
# #########################################

    #
    # plt.subplot(8, 1, 1)
    # plt.title(title1 + ", intervals of 300 bits, leads 'upper'")
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[0:300, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.subplot(8, 1, 2)
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[301:600, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.subplot(8, 1, 3)
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[601:900, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.subplot(8, 1, 4)
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[901:1200, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.subplot(8, 1, 5)
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[1201:1500, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.subplot(8, 1, 6)
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[1501:1800, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.subplot(8, 1, 7)
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[1801:2100, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.subplot(8, 1, 8)
    # plt.imshow(np.transpose(np.squeeze(LeadsIm[2101:2400, :])), origin='lower', interpolation="nearest", cmap="gray")
    # plt.figure(pltfig1, figsize=(20, 20))

################################################
# """
# You can specify whether images should be plotted with the array origin
# x[0,0] in the upper left or upper right by using the origin parameter.
# You can also control the default be setting image.origin in your
# matplotlibrc file; see http://matplotlib.org/matplotlibrc
# """
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(120)
# x.shape = (10, 12)
#
# interp = 'bilinear'
# fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))
# axs[0].set_title('blue should be up')
# axs[0].imshow(x, origin='upper', interpolation=interp)
#
# axs[1].set_title('blue should be down')
# axs[1].imshow(x, origin='lower', interpolation=interp)
# plt.show()

########################################