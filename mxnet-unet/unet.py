import os
os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

import mxnet as mx
from mxnet import ndarray as F
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from unetdataiter import UnetDataIter
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=(1, 2, 3))
    return mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.sum(y_true, axis=(1, 2, 3)) + mx.sym.sum(y_pred, axis=(1, 2, 3)) + 1.))

def dice_coef_loss(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=1, )
    return -mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.broadcast_add(mx.sym.sum(y_true, axis=1), mx.sym.sum(y_pred, axis=1)) + 1.))

def build_unet(batch_size, input_width, input_height, train=True):
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='softmax_label')

    # encode
    # 256x256
    conv1 = mx.sym.Convolution(data, num_filter=64, kernel=(3,3), pad=(1,1), name='conv1_1')
    conv1 = mx.sym.BatchNorm(conv1, name='bn1_1')
    conv1 = mx.sym.Activation(conv1, act_type='relu', name='relu1_1')
    conv1 = mx.sym.Convolution(conv1, num_filter=64, kernel=(3,3), pad=(1,1), name='conv1_2')
    conv1 = mx.sym.BatchNorm(conv1, name='bn1_2')
    conv1 = mx.sym.Activation(conv1, act_type='relu', name='relu1_2')
    pool1 = mx.sym.Pooling(conv1, kernel=(2,2), stride=(2, 2), pool_type='max', name='pool1')
    # 128x128
    conv2 = mx.sym.Convolution(pool1, num_filter=128, kernel=(3,3), pad=(1,1), name='conv2_1')
    conv2 = mx.sym.BatchNorm(conv2, name='bn2_1')
    conv2 = mx.sym.Activation(conv2, act_type='relu', name='relu2_1')
    conv2 = mx.sym.Convolution(conv2, num_filter=128, kernel=(3,3), pad=(1,1), name='conv2_2')
    conv2 = mx.sym.BatchNorm(conv2, name='bn2_2')
    conv2 = mx.sym.Activation(conv2, act_type='relu', name='relu2_2')
    pool2 = mx.sym.Pooling(conv2, kernel=(2,2), stride=(2, 2), pool_type='max', name='pool2')
    # 64x64
    conv3 = mx.sym.Convolution(pool2, num_filter=256, kernel=(3,3), pad=(1,1), name='conv3_1')
    conv3 = mx.sym.BatchNorm(conv3, name='bn3_1')
    conv3 = mx.sym.Activation(conv3, act_type='relu', name='relu3_1')
    conv3 = mx.sym.Convolution(conv3, num_filter=256, kernel=(3,3), pad=(1,1), name='conv3_2')
    conv3 = mx.sym.BatchNorm(conv3, name='bn3_2')
    conv3 = mx.sym.Activation(conv3, act_type='relu', name='relu3_2')
    pool3 = mx.sym.Pooling(conv3, kernel=(2,2), stride=(2, 2), pool_type='max', name='pool3')
    # 32x32
    conv4 = mx.sym.Convolution(pool3, num_filter=256, kernel=(3,3), pad=(1,1), name='conv4_1')
    conv4 = mx.sym.BatchNorm(conv4, name='bn4_1')
    conv4 = mx.sym.Activation(conv4, act_type='relu', name='relu4_1')
    conv4 = mx.sym.Convolution(conv4, num_filter=256, kernel=(3,3), pad=(1,1), name='conv4_2')
    conv4 = mx.sym.BatchNorm(conv4, name='bn4_2')
    conv4 = mx.sym.Activation(conv4, act_type='relu', name='relu4_2')
    pool4 = mx.sym.Pooling(conv4, kernel=(2,2), stride=(2, 2), pool_type='max', name='pool4')
    # 16x16
    conv5 = mx.sym.Convolution(pool4, num_filter=256, kernel=(3,3), pad=(1,1), name='conv5_1')
    conv5 = mx.sym.BatchNorm(conv5, name='bn5_1')
    conv5 = mx.sym.Activation(conv5, act_type='relu', name='relu5_1')
    conv5 = mx.sym.Convolution(conv5, num_filter=256, kernel=(3,3), pad=(1,1), name='conv5_2')
    conv5 = mx.sym.BatchNorm(conv5, name='bn5_2')
    conv5 = mx.sym.Activation(conv5, act_type='relu', name='relu5_2')
    pool5 = mx.sym.Pooling(conv5, kernel=(2,2), stride=(2, 2), pool_type='max', name='pool5')
    # 8x8

    # decode
    trans_conv6 = mx.sym.Deconvolution(pool5, num_filter=256, kernel=(2,2), stride=(2,2), no_bias=True, name='trans_conv6')
    up6 = mx.sym.concat(*[trans_conv6, conv5], dim=1, name='concat6')
    conv6 = mx.sym.Convolution(up6, num_filter=256, kernel=(3,3), pad=(1,1), name='conv6_1')
    conv6 = mx.sym.BatchNorm(conv6, name='bn6_1')
    conv6 = mx.sym.Activation(conv6, act_type='relu', name='relu6_1')
    conv6 = mx.sym.Convolution(conv6, num_filter=256, kernel=(3,3), pad=(1,1), name='conv6_2')
    conv6 = mx.sym.BatchNorm(conv6, name='bn6_2')
    conv6 = mx.sym.Activation(conv6, act_type='relu', name='relu6_2')

    trans_conv7 = mx.sym.Deconvolution(conv6, num_filter=256, kernel=(2,2), stride=(2,2), no_bias=True, name='trans_conv7')
    up7 = mx.sym.concat(*[trans_conv7, conv4], dim=1, name='concat7')
    conv7 = mx.sym.Convolution(up7, num_filter=256, kernel=(3,3), pad=(1,1), name='conv7_1')
    conv7 = mx.sym.BatchNorm(conv7, name='bn7_1')
    conv7 = mx.sym.Activation(conv7, act_type='relu', name='relu7_1')
    conv7 = mx.sym.Convolution(conv7, num_filter=256, kernel=(3,3), pad=(1,1), name='conv7_2')
    conv7 = mx.sym.BatchNorm(conv7, name='bn7_2')
    conv7 = mx.sym.Activation(conv7, act_type='relu', name='relu7_2')

    trans_conv8 = mx.sym.Deconvolution(conv7, num_filter=256, kernel=(2,2), stride=(2,2), no_bias=True, name='trans_conv8')
    up8 = mx.sym.concat(*[trans_conv8, conv3], dim=1, name='concat8')
    conv8 = mx.sym.Convolution(up8, num_filter=256, kernel=(3,3), pad=(1,1), name='conv8_1')
    conv8 = mx.sym.BatchNorm(conv8, name='bn8_1')
    conv8 = mx.sym.Activation(conv8, act_type='relu', name='relu8_1')
    conv8 = mx.sym.Convolution(conv8, num_filter=256, kernel=(3,3), pad=(1,1), name='conv8_2')
    conv8 = mx.sym.BatchNorm(conv8, name='bn8_2')
    conv8 = mx.sym.Activation(conv8, act_type='relu', name='relu8_2')

    trans_conv9 = mx.sym.Deconvolution(conv8, num_filter=128, kernel=(2,2), stride=(2,2), no_bias=True, name='trans_conv9')
    up9 = mx.sym.concat(*[trans_conv9, conv2], dim=1, name='concat9')
    conv9 = mx.sym.Convolution(up9, num_filter=128, kernel=(3,3), pad=(1,1), name='conv9_1')
    conv9 = mx.sym.BatchNorm(conv9, name='bn9_1')
    conv9 = mx.sym.Activation(conv9, act_type='relu', name='relu9_1')
    conv9 = mx.sym.Convolution(conv9, num_filter=128, kernel=(3,3), pad=(1,1), name='conv9_2')
    conv9 = mx.sym.BatchNorm(conv9, name='bn9_2')
    conv9 = mx.sym.Activation(conv9, act_type='relu', name='relu9_2')

    trans_conv10 = mx.sym.Deconvolution(conv9, num_filter=64, kernel=(2,2), stride=(2,2), no_bias=True, name='trans_conv10')
    up10 = mx.sym.concat(*[trans_conv10, conv1], dim=1, name='concat10')
    conv10 = mx.sym.Convolution(up10, num_filter=64, kernel=(3,3), pad=(1,1), name='conv10_1')
    conv10 = mx.sym.BatchNorm(conv10, name='bn10_1')
    conv10 = mx.sym.Activation(conv10, act_type='relu', name='relu10_1')
    conv10 = mx.sym.Convolution(conv10, num_filter=64, kernel=(3,3), pad=(1,1), name='conv10_2')
    conv10 = mx.sym.BatchNorm(conv10, name='bn10_2')
    conv10 = mx.sym.Activation(conv10, act_type='relu', name='relu10_2')

    ###
    conv11 = mx.sym.Convolution(conv10, num_filter=2, kernel=(1,1), name='conv11_1')
    conv11 = mx.sym.sigmoid(conv11, name='softmax')

    net = mx.sym.Reshape(conv11, (batch_size, 2, input_width*input_height))

    if train:
        loss = mx.sym.MakeLoss(dice_coef_loss(label, net), normalization='batch')
        mask_output = mx.sym.BlockGrad(conv11, 'mask')
        out = mx.sym.Group([loss, mask_output])
    else:
        # mask_output = mx.sym.BlockGrad(conv11, 'mask')
        out = mx.sym.Group([conv11])

    return out


# https://blog.csdn.net/JianJuly/article/details/81105436
