
#!/usr/bin/env python
# coding=utf8

import os
import sys
import random
import cv2
import mxnet as mx
import numpy as np
from mxnet.io import DataIter, DataBatch

sys.path.append('../')

def get_batch(items, root_path, nClasses, height, width):
    x = []
    y = []
    for item in items:
        flipped = False
        cropped = False
        image_path = root_path + item.split(' ')[0]
        label_path = root_path + item.split(' ')[-1].strip()
        if image_path.find('_flipped') >= 0:
          image_path = image_path.replace('_flipped', '')
          flipped = True
        elif image_path.find('_cropped') >= 0:
          image_path = image_path.replace('_cropped', '')
          cropped = True
        img = cv2.imread(image_path, 1)
        label_img = cv2.imread(label_path, 1)
        if cropped:
          tmp_height = img.shape[0]
          img = img[:,tmp_height//5:tmp_height*4//5]
          tmp_height = label_img.shape[0]
          label_img = label_img[:,tmp_height//5:tmp_height*4//5]
        im = np.zeros((height, width, 3), dtype='uint8')
        im[:, :, :] = 128
        lim = np.zeros((height, width, 3), dtype='uint8')

        if img.shape[0] >= img.shape[1]:
            scale = img.shape[0] / height
            new_width = int(img.shape[1] / scale)
            diff = (width - new_width) // 2

            img = cv2.resize(img, (new_width, height))
            label_img = cv2.resize(label_img, (new_width, height))

            im[:, diff:diff + new_width, :] = img
            lim[:, diff:diff + new_width, :] = label_img
        else:
            scale = img.shape[1] / width
            new_height = int(img.shape[0] / scale)
            diff = (height - new_height) // 2

            img = cv2.resize(img, (width, new_height))
            label_img = cv2.resize(label_img, (width, new_height))
            im[diff:diff + new_height, :, :] = img
            lim[diff:diff + new_height, :, :] = label_img
        if flipped:
          im = cv2.flip(im, 1)
          lim = cv2.flip(lim, 1)
        lim = lim[:, :, 0]
        seg_labels = np.zeros((height, width, nClasses))
        for c in range(nClasses):
            seg_labels[:, :, c] = (lim == c).astype(int)
        im = np.float32(im) / 255.0
        seg_labels = np.reshape(seg_labels, (width * height, nClasses))
        x.append(im.transpose((2,0,1)))
        y.append(seg_labels.transpose((1,0)))

    return mx.nd.array(x), mx.nd.array(y)

class UnetDataIter(mx.io.DataIter):
    def __init__(self, root_path, path_file, batch_size, n_classes, input_width, input_height, train=True):
        f = open(path_file, 'r')
        self.items = f.readlines()
        f.close()

        self._provide_data = [['data', (batch_size, 3, input_width, input_height)]]
        self._provide_label = [['softmax_label', (batch_size, n_classes, input_width*input_height)]]

        self.root_path = root_path
        self.batch_size = batch_size
        self.num_batches = len(self.items) // batch_size
        self.n_classes = n_classes
        self.input_height = input_height
        self.input_width = input_width
        self.train = train

        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

        self.shuffled_items = []
        index = [n for n in range(len(self.items))]

        if self.train:
            random.shuffle(index)

        for i in range(len(self.items)):
            self.shuffled_items.append(self.items[index[i]])

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch == 0:
            print("")

        print("\r\033[k"+("Training " if self.train else "Validating ")+str(self.cur_batch)+"/"+str(self.num_batches), end=' ')

        if self.cur_batch < self.num_batches:
            data, label = get_batch(self.shuffled_items[self.cur_batch * self.batch_size:(self.cur_batch + 1) * self.batch_size], self.root_path, self.n_classes, self.input_height, self.input_width)
            self.cur_batch += 1

            return mx.io.DataBatch([data], [label])
        else:
            raise StopIteration

if __name__ =='__main__':
    root_path = '/datasets/'
    train_file = './data/seg_train.txt'
    val_file = './data/seg_test.txt'
    batch_size = 16
    n_classes = 2
    img_width = 256
    img_height = 256

    trainiter = UnetDataIter(root_path, train_file, batch_size, n_classes, img_width, img_height, True)

    while True:
        trainiter.next()
