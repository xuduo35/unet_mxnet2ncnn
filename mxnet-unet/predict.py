import os
os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
import sys
import cv2
import mxnet as mx
from mxnet import ndarray as F
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from unetdataiter import UnetDataIter
import matplotlib.pyplot as plt
from unet import build_unet

def post_process_mask(label, img_cols, img_rows, n_classes, p=0.5):
    pr = label.reshape(n_classes, img_cols, img_rows).transpose([1,2,0]).argmax(axis=2)
    return (pr*255).asnumpy()

def load_image(img, width, height):
    im = np.zeros((height, width, 3), dtype='uint8')
    im[:, :, :] = 128

    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / height
        new_width = int(img.shape[1] / scale)
        diff = (width - new_width) // 2
        img = cv2.resize(img, (new_width, height))

        im[:, diff:diff + new_width, :] = img
    else:
        scale = img.shape[1] / width
        new_height = int(img.shape[0] / scale)
        diff = (height - new_height) // 2

        img = cv2.resize(img, (width, new_height))
        im[diff:diff + new_height, :, :] = img

    im = np.float32(im) / 255.0

    return [im.transpose((2,0,1))]

def main():
    batch_size = 16
    n_classes = 2
    img_width = 256
    img_height = 256
    #img_width = 96
    #img_height = 96

    ctx = [mx.gpu(0)]

    # sym, arg_params, aux_params = mx.model.load_checkpoint('unet_person_segmentation', 20)
    # unet_sym = build_unet(batch_size, img_width, img_height, False)
    # unet = mx.mod.Module(symbol=unet_sym, context=ctx, label_names=None)

    sym, arg_params, aux_params = mx.model.load_checkpoint('./models/unet_person_segmentation', 100)
    unet = mx.mod.Module(symbol=sym, context=ctx, label_names=None)

    unet.bind(for_training=False, data_shapes=[['data', (batch_size, 3, img_width, img_height)]], label_shapes=unet._label_shapes)
    unet.set_params(arg_params, aux_params, allow_missing=True)

    testimg = cv2.imread(sys.argv[1], 1)
    img = load_image(testimg, img_width, img_height)
    unet.predict(mx.io.NDArrayIter(data=[img]))

    outputs = unet.get_outputs()[0]
    cv2.imshow('test', testimg)
    cv2.imshow('mask', post_process_mask(outputs[0], img_width, img_height, n_classes))
    cv2.waitKey()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("illegal parameters")
        sys.exit(0)

    main()
