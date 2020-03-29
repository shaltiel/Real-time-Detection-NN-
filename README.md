
# Real-Time detection of piano using Neural Networks

## Summary
Real-time Piano Detection - a paradigm for working with Pytorch->ONNX->Max/MSP.

As an example we show the ability to develop a simple detection of real-time and low-latency model for piano notes and the classification of instrumets.
This paradigm is an open source that can be easily used by artists and researchers to combine NN models
for music and art applications.

## Method
The example is implementation as external objects in Max/MSP, and the workflow is as follow:

1. Generate a training set using the external object traindyn~.mxo
2. Train using Pytorch.
3. Use the trained model with the external inferdyn~.mxo for inference.

## Instructions 

### Setting up the connx: 
1. First download the realtime dynamic libraries from onnx-realtime microsoft:     https://github.com/microsoft/onnxruntime/releases/tag/v1.1.0 

2. Now you use  or create the dynamic lib that will be used within max object, this is in the connx/main.cpp file.
- There is one precompiled that can be used called mainlib.so.
- but if you want to compile from source or you use a different version  than onnxruntime v1.1.0.
    If compile, for example in terminal go to the path connx/main.cpp  (with the correct path to ONNX, and the correct   
    version):

    g++ -shared -o libmain.so main.cpp -IPATH/TO/ONNX/onnxruntime-osx-x64-1.1.0/include/ -L"PATH/TO/ONNX/onnxruntime-osx-x64-     1.1.0/lib" -lonnxruntime.1.1.0 -lonnxruntime -std=c++14 -Wl,-rpath,"@executable_path/PATH/TO/ONNX/onnxruntime-osx-x64-    
    1.1.0/lib"


### Setting up in max: 
3. Open the maxsource/training_set.maxhelp in max and add the external object path. Follow the instructions in the patch.
(OSX, but source code avialable to build in windows)
4. Load piano vst or input piano and use inference with exsited ONNX model (see /model_pytorch/trainedmodels), or create a new training set.

### Train a new model in PyTorch to Max:
5. Follow the python notebook to train and export a new ONNX model then load it in inferdyn~ object in max.


