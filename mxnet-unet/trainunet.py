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
from unet import build_unet

def main():
    root_path = '../datasets/'
    train_file = './data/seg_train.txt'
    val_file = './data/seg_test.txt'
    batch_size = 16
    n_classes = 2
    img_width = 256
    img_height = 256
    #img_width = 96
    #img_height = 96

    train_iter = UnetDataIter(root_path, train_file, batch_size, n_classes, img_width, img_height, True)
    val_iter = UnetDataIter(root_path, val_file, batch_size, n_classes, img_width, img_height, False)

    ctx = [mx.gpu(0)]

    unet_sym = build_unet(batch_size, img_width, img_height)
    unet = mx.mod.Module(unet_sym, context=ctx, data_names=('data',), label_names=('softmax_label',))
    unet.bind(data_shapes=[['data', (batch_size, 3, img_width, img_height)]], label_shapes=[['softmax_label', (batch_size, n_classes, img_width*img_height)]])
    unet.init_params(mx.initializer.Xavier(magnitude=6))

    unet.init_optimizer(optimizer = 'adam',
                                   optimizer_params=(
                                       ('learning_rate', 1E-4),
                                       ('beta1', 0.9),
                                       ('beta2', 0.99)
                                  ))

    if os.path.exists("./best_unet_person_segmentation-0000.params"):
        print("Loading for best_unet_person_segmentation ...")
        unet_sym, arg_params, aux_params = mx.model.load_checkpoint('best_unet_person_segmentation', 0)
        unet.set_params(arg_params, aux_params, allow_missing=True)
    else:
        print("New training ...")

    # unet.fit(train_iter,  # train data
    #               eval_data=val_iter,  # validation data
    #               #optimizer='sgd',  # use SGD to train
    #               #optimizer_params={'learning_rate':0.1},  # use fixed learning rate
    #               eval_metric='acc',  # report accuracy during training
    #               batch_end_callback = mx.callback.Speedometer(batch_size, 1), # output progress for each 100 data batches
    #               num_epoch=10)  # train for at most 10 dataset passes

    epochs = 50
    smoothing_constant = .01
    curr_losses = []
    moving_losses = []
    i = 0
    best_val_loss = np.inf
    for e in range(epochs):
        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter.reset()
                break
            unet.forward_backward(batch)
            loss = unet.get_outputs()[0]
            unet.update()
            curr_loss = F.mean(loss).asscalar()
            curr_losses.append(curr_loss)
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                                   else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
            moving_losses.append(moving_loss)
            i += 1
        val_losses = []
        for batch in val_iter:
            unet.forward(batch)
            loss = unet.get_outputs()[0]
            val_losses.append(F.mean(loss).asscalar())
        val_iter.reset()
        val_loss = np.mean(val_losses)

        if e > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            unet.save_checkpoint('best_unet_person_segmentation', 0)
            print("Best model at Epoch %i" %(e+1))

        print("\nEpoch %i: Moving Training Loss %0.5f, Validation Loss %0.5f" % (e, moving_loss, val_loss))

if __name__ =='__main__':
    main()


