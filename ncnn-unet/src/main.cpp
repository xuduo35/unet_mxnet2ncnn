#include "net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define INPUT_WIDTH     256
#define INPUT_HEIGHT    256

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("illegal parameters!");
        exit(0);
    }

    ncnn::Net Unet;

    Unet.load_param("../models/ncnn.param");
    Unet.load_model("../models/ncnn.bin");

    cv::Scalar value = Scalar(0,0,0);
    cv::Mat src;
    cv::Mat tmp;
    src = cv::imread(argv[1]);

    if (src.size().width > src.size().height) {
        int top = (src.size().width - src.size().height) / 2;
        int bottom = (src.size().width - src.size().height) - top;
        cv::copyMakeBorder(src, tmp, top, bottom, 0, 0, BORDER_CONSTANT, value);
    } else {
        int left = (src.size().height - src.size().width) / 2;
        int right = (src.size().height - src.size().width) - left;
        cv::copyMakeBorder(src, tmp, 0, 0, left, right, BORDER_CONSTANT, value);
    }

    cv::Mat tmp1;
    cv::resize(tmp, tmp1, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), CV_INTER_CUBIC);

    cv::Mat image;
    tmp1.convertTo(image, CV_32FC3, 1/255.0);

    std::cout << "image element type "<< image.type() << " " << image.cols << " " << image.rows << std::endl;

    // std::cout << src.cols << " " << src.rows << " " << image.cols << " " << image.rows << std::endl;
    //cv::imshow("test", image);
    //cv::waitKey();

    //ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);

    // cv32fc3 的布局是 hwc ncnn的Mat布局是 chw 需要调整排布
    float *srcdata = (float*)image.data;
    float *data = new float[INPUT_WIDTH*INPUT_HEIGHT*3];
    for (int i = 0; i < INPUT_HEIGHT; i++)
       for (int j = 0; j < INPUT_WIDTH; j++)
           for (int k = 0; k < 3; k++) {
              data[k*INPUT_HEIGHT*INPUT_WIDTH + i*INPUT_WIDTH + j] = srcdata[i*INPUT_WIDTH*3 + j*3 + k];
           }
    ncnn::Mat in(image.rows*image.cols*3, data);
    in = in.reshape(256, 256, 3);

    //ncnn::Mat in;

    //resize_bilinear(ncnn_img, in, INPUT_WIDTH, INPUT_HEIGHT);

    ncnn::Extractor ex = Unet.create_extractor();

    ex.set_light_mode(true);
    //sex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat mask;
    //ex.extract("relu5_2_splitncnn_0", mask);
    //ex.extract("trans_conv6", mask);
    ex.extract("conv11_1", mask);
    //ex.extract("pool5", mask);

    std::cout << "whc " << mask.w << " " << mask.h << " " << mask.c << std::endl;
#if 1
    cv::Mat cv_img = cv::Mat::zeros(INPUT_WIDTH,INPUT_HEIGHT,CV_8UC1);
//    mask.to_pixels(cv_img.data, ncnn::Mat::PIXEL_GRAY);

    {
    float *srcdata = (float*)mask.data;
    unsigned char *data = cv_img.data;

    for (int i = 0; i < mask.h; i++)
       for (int j = 0; j < mask.w; j++) {
         float tmp = srcdata[0*mask.w*mask.h+i*mask.w+j];
         int maxk = 0;
         for (int k = 0; k < mask.c; k++) {
           if (tmp < srcdata[k*mask.w*mask.h+i*mask.w+j]) {
             tmp = srcdata[k*mask.w*mask.h+i*mask.w+j];
             maxk = k;
           }
           //std::cout << srcdata[k*mask.w*mask.h+i*mask.w+j] << std::endl;
         }
         data[i*INPUT_WIDTH + j] = maxk;
       }
    }
    
    cv_img *= 255;
    cv::imshow("test", cv_img);
    cv::waitKey();
#endif
    return 0;
}
