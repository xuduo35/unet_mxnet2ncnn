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

def main():
    batch_size = 16
    n_classes = 2
    img_width = 256
    img_height = 256
    # img_width = 96
    # img_height = 96

    ctx = [mx.gpu(0)]

    sym, arg_params, aux_params = mx.model.load_checkpoint('best_unet_person_segmentation', 0)

    unet_sym = build_unet(batch_size, img_width, img_height, False)
    unet = mx.mod.Module(symbol=unet_sym, context=ctx, label_names=None)
    unet.bind(for_training=False, data_shapes=[['data', (batch_size, 3, img_width, img_height)]], label_shapes=unet._label_shapes)
    unet.set_params(arg_params, aux_params, allow_missing=True)

    unet.save_checkpoint('unet_person_segmentation', 0)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("illegal parameters")
        sys.exit(0)

    main()
