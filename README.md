# unet_mxnet2ncnn 

Train mxnet unet and run it on ncnn.

Dataset from https://github.com/TianzhongSong/Person-Segmentation-Keras

Some code from: https://github.com/milesial/Pytorch-UNet/tree/master/unet

Blog link: https://blog.csdn.net/xiexiecn/article/details/83029787

How to use:

1. Get dataset and put it in the some directory. Adjust files under mxnet-unet/data based on dataset location.

2. cd mxnet-unet; python3 trainunet.py

3. Get inference model by running: python3 train2infer.py

4. Use ncnn tool 'mxnet-ncnn' to get ncnn model

   Need to patch tools/mxnet/mxnet2ncnn.cpp for ncnn. Set num_group to 1 in if for "Deconvolution" op. But I think it will fix also if designate it in mxnet network.

5. Run ncnn-unet.

   Install OpenCV first.

   cd ncnn-unet

   Edit CMakefiles to fit your enviroment.

   mkdir build

   cmake ..; make

   ./unet your_jpg_file

