# Deformable Convolutional Networks
## Requirements: software
```
pip install Cython
pip install opencv-python==3.2.0.6
pip install easydict==1.6

g++ version > 4.9
install g++ 4.9.2

sudo yum install libmpc-devel mpfr-devel gmp-devel
cd ~/Downloads
curl ftp://ftp.mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-4.9.2/gcc-4.9.2.tar.bz2 -O
tar xvfj gcc-4.9.2.tar.bz2

cd gcc-4.9.2
./configure --disable-multilib --enable-languages=c,c++
make -j32
make install

check: g++ -v

```
if you are update g++4.9.2 from old version,you must check your  dynamic library:

strings /usr/lib/libstdc++.so.6 | grep CXXABI

the results like:
```
CXXABI_1.3
CXXABI_1.3.1
CXXABI_1.3.2
CXXABI_1.3.3
```
lacking CXXABI_1.3.8, this is because we donot update the  dynamic library

step1:

```
find / -name "libstdc++.so.*"
```
you will find 
```
/usr/local/src/gcc-6.3.0/gcc-build-6.3.0/prev-i686-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6.0.22
/usr/local/src/gcc-6.3.0/gcc-build-6.3.0/prev-i686-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6
/usr/local/src/gcc-6.3.0/gcc-build-6.3.0/stage1-i686-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6.0.22
/usr/local/src/gcc-6.3.0/gcc-build-6.3.0/stage1-i686-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6
/usr/local/src/gcc-6.3.0/gcc-build-6.3.0/i686-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6.0.22
/usr/local/src/gcc-6.3.0/gcc-build-6.3.0/i686-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6
/usr/local/lib/libstdc++.so.6.0.22
/usr/local/lib/libstdc++.so.6.0.22-gdb.py
/usr/local/lib/libstdc++.so.6
/usr/lib/libstdc++.so.6.0.13
/usr/lib/libstdc++.so.6
……
```
note that: the path /usr/local/src/gcc-6.3.0/gcc-build-6.3.0/i686-pc-linux-gnu replace by your path

step 2:
```
cp /usr/local/src/gcc-6.3.0/gcc-build-6.3.0/i686-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6.0.22 /usr/lib/
```
step 3:
```
cd /usr/lib
rm -rf libstdc++.so.6
ln -s libstdc++.so.6.0.22 libstdc++.so.6
```

```
strings /usr/lib/libstdc++.so.6 | grep 'CXXABI'

CXXABI_1.3
CXXABI_1.3.1
CXXABI_1.3.2
CXXABI_1.3.3
CXXABI_1.3.4
CXXABI_1.3.5
CXXABI_1.3.6
CXXABI_1.3.7
CXXABI_1.3.8
CXXABI_1.3.9
CXXABI_1.3.10
CXXABI_TM_1
CXXABI_FLOAT128
```
if you are runing in x64
```
strings /usr/lib64/libstdc++.so.6 | grep GLIBC
the results:
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_FORCE_NEW
GLIBCXX_DEBUG_MESSAGE_LENGTH
```

```
find / -name "libstdc++.so*"

/home/gcc-5.2.0/gcc-temp/stage1-x86_64-unknown-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so
/home/gcc-5.2.0/gcc-temp/stage1-x86_64-unknown-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6
/home/gcc-5.2.0/gcc-temp/stage1-x86_64-unknown-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6.0.21  //the new
……



cp /home/gcc-5.2.0/gcc-temp/stage1-x86_64-unknown-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6.0.21 /usr/lib64


cd /usr/lib64
rm -rf libstdc++.so.6
ln -s libstdc++.so.6.0.21 libstdc++.so.6
strings /usr/lib64/libstdc++.so.6 | grep GLIBC

the results:

GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBC_2.3
GLIBC_2.2.5
GLIBC_2.3.2
GLIBCXX_FORCE_NEW
GLIBCXX_DEBUG_MESSAGE_LENGTH

```





## Installation

1. Clone the Deformable ConvNets repository, and we'll call the directory that you cloned Deformable-ConvNets as ${DCN_ROOT}.
```
git clone https://github.com/msracver/Deformable-ConvNets.git
```

2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

	3.1 Clone MXNet and checkout to [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) by
	
	```
	git clone  https://github.com/dmlc/mxnet.git --recursive && --checkout v0.9.5
	```
	3.2 Copy operators in `$(DCN_ROOT)/rfcn/operator_cxx` or `$(DCN_ROOT)/faster_rcnn/operator_cxx` to `$(YOUR_MXNET_FOLDER)/src/operator/contrib` by
	```
	cp -r $(DCN_ROOT)/rfcn/operator_cxx/* $(MXNET_ROOT)/src/operator/contrib/
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j16 USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	3.4 Install the MXNet Python binding by
	
	***Note: If you will actively switch between different versions of MXNet, please follow 3.5 instead of 3.4***
	```
	cd python
	sudo python setup.py install
	```
	3.5 For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)`, and modify `MXNET_VERSION` in `./experiments/rfcn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.

4. For Deeplab, we use the argumented VOC 2012 dataset. The argumented annotations are provided by [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) dataset. For convenience, we provide the converted PNG annotations and the lists of train/val images, please download them from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMRhVImMI1jRrsxDg).

## Demo & Deformable Model

We provide trained deformable convnet models, including the deformable R-FCN & Faster R-CNN models trained on COCO trainval, and the deformable DeepLab model trained on CityScapes train.

1. To use the demo with our pre-trained deformable models, please download manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMSjehIcCgAhvEAHw), and put it under folder `model/`.

	Make sure it looks like this:
	```
	./model/rfcn_dcn_coco-0000.params
	./model/rfcn_coco-0000.params
	./model/rcnn_dcn_coco-0000.params
	./model/rcnn_coco-0000.params
	./model/deeplab_dcn_cityscapes-0000.params
	./model/deeplab_cityscapes-0000.params
	./model/deform_conv-0000.params
	./model/deform_psroi-0000.params
	```
2. To run the R-FCN demo, run
	```
	python ./rfcn/demo.py
	```
	By default it will run Deformable R-FCN and gives several prediction results, to run R-FCN, use
	```
	python ./rfcn/demo.py --rfcn_only
	```
3. To run the DeepLab demo, run
	```
	python ./deeplab/demo.py
	```
	By default it will run Deformable Deeplab and gives several prediction results, to run DeepLab, use
	```
	python ./deeplab/demo.py --deeplab_only
	```
4. To visualize the offset of deformable convolution and deformable psroipooling, run
	```
	python ./rfcn/deform_conv_demo.py
	python ./rfcn/defrom_psroi_demo.py
	```


## Preparation for Training & Testing

For R-FCN/Faster R-CNN\:
1. Please download COCO and VOC 2007+2012 datasets, and make sure it looks like this:

	```
	./data/coco/
	./data/VOCdevkit/VOC2007/
	./data/VOCdevkit/VOC2012/
	```

2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	```

For DeepLab\:
1. Please download Cityscapes and VOC 2012 datasets and make sure it looks like this:

	```
	./data/cityscapes/
	./data/VOCdevkit/VOC2012/
	```
2. Please download argumented VOC 2012 annotations/image lists, and put the argumented annotations and the argumented train/val lists into:

	```
	./data/VOCdevkit/VOC2012/SegmentationClass/
	./data/VOCdevkit/VOC2012/ImageSets/Main/
	```
   , Respectively.
   
2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	```
## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/rfcn/cfgs`, `./experiments/faster_rcnn/cfgs` and `./experiments/deeplab/cfgs/`.
2. Eight config files have been provided so far, namely, R-FCN for COCO/VOC, Deformable R-FCN for COCO/VOC, Faster R-CNN(2fc) for COCO/VOC, Deformable Faster R-CNN(2fc) for COCO/VOC, Deeplab for Cityscapes/VOC and Deformable Deeplab for Cityscapes/VOC, respectively. We use 8 and 4 GPUs to train models on COCO and on VOC for R-FCN, respectively. For deeplab, we use 4 GPUs for all experiments.

3. To perform experiments, run the python scripts with the corresponding config file as input. For example, to train and test deformable convnets on COCO with ResNet-v1-101, use the following command
    ```
    python experiments\rfcn\rfcn_end2end_train_test.py --cfg experiments\rfcn\cfgs\resnet_v1_101_coco_trainval_rfcn_dcn_end2end_ohem.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `output/rfcn_dcn_coco/`.
4. Please find more details in config files and in our code.

## Misc.

Code has been tested under:

- Ubuntu 14.04 with a Maxwell Titan X GPU and Intel Xeon CPU E5-2620 v2 @ 2.10GHz
- Windows Server 2012 R2 with 8 K40 GPUs and Intel Xeon CPU E5-2650 v2 @ 2.60GHz
- Windows Server 2012 R2 with 4 Pascal Titan X GPUs and Intel Xeon CPU E5-2650 v4 @ 2.30GHz

## FAQ

Q: It says `AttributeError: 'module' object has no attribute 'DeformableConvolution'`.

A: This is because either
 - you forget to copy the operators to your MXNet folder
 - or you copy to the wrong path
 - or you forget to re-compile
 - or you install the wrong MXNet

    Please print `mxnet.__path__` to make sure you use correct MXNet

<br/><br/>
Q: I encounter `segment fault` at the beginning.

A: A compatibility issue has been identified between MXNet and opencv-python 3.0+. We suggest that you always `import cv2` first before `import mxnet` in the entry script. 

<br/><br/>
Q: I find the training speed becomes slower when training for a long time.

A: It has been identified that MXNet on Windows has this problem. So we recommend to run this program on Linux. You could also stop it and resume the training process to regain the training speed if you encounter this problem.

<br/><br/>
Q: Can you share your caffe implementation?

A: Due to several reasons (code is based on a old, internal Caffe, port to public Caffe needs extra work, time limit, etc.). We do not plan to release our Caffe code. Since current MXNet convolution implementation is very similar to Caffe (almost the same), it is easy to port to Caffe by yourself, the core CUDA code could be kept unchanged. Anyone who wish to do it is welcome to make a pull request.
