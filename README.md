
# Real-Time detection of piano using Neural Networks

## Summary
Real-time Piano Detection - a paradigm for working with Pytorch->ONNX->Max/MSP.

As an example we show the ability to develop a low-latency detection algorithm for acoustic piano.
This paradigm is an open source and can be easily used by artists and researchers to combine NN models
in music and art applications.

## Method

The example provides interface to Max/MSP by using two external objects. The workflow is as follow:

1. Generate a training set by using the external object traindyn~.mxo. This will create the dataset and label files.
2. Train a model using Pytorch.
3. Use the trained model with the external inferdyn~.mxo for inference. This is done by interfacing to ONNX-realtime c++ library. 


## Instructions 

### Setting up the OONX: 
1. First download the realtime dynamic libraries from onnx-realtime microsoft:     https://github.com/microsoft/onnxruntime/releases/tag/v1.1.0 

2. Now you need to create the dynamic lib which will be used within max object, this is done by compiling the connx/main.cpp file.
- There is one precompiled lib that can be used, called mainlib.so.
- but if you want to compile from source or you use other version than onnxruntime v1.1.0.
    For example, in terminal go to the path connx/main.cpp  (ensure the correct path to ONNX, and the correct   
    version):

    g++ -shared -o libmain.so main.cpp -IPATH/TO/ONNX/onnxruntime-osx-x64-1.1.0/include/ -L"PATH/TO/ONNX/onnxruntime-osx-x64-     1.1.0/lib" -lonnxruntime.1.1.0 -lonnxruntime -std=c++14 -Wl,-rpath,"@executable_path/PATH/TO/ONNX/onnxruntime-osx-x64-    
    1.1.0/lib"


### Setting up in max: 
3. Open the maxsource/training_set.maxhelp in max and add the external object path. Follow the instructions in the patch.
(OSX, but source code avialable to build in windows)
4. Load piano vst or input piano and use inference with exsited ONNX model (see /model_pytorch/trainedmodels), or create a new training set.

### Train a new model in PyTorch to Max:
5. Follow the python notebook to train and export a new ONNX model then load it in inferdyn~ object in max.


