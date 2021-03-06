//
//
//  cnnoxt max pytorch model
//
//  Created by se on 14/01/2020.
//  Copyright © 2020 FLEX. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <sstream>

extern "C"
float * test (float * testf);
extern "C"
float * inference (float *,float *,int,long);

extern "C"
const char * initial(const char*);


Ort::Env*  env_global;
std::stringstream meta;
Ort::SessionOptions * psession_options;

struct session_max
{
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    const char* model_path;
//    Ort::Session *session;
    size_t input_tensor_size;
    int64_t size_label;
}sei;

std::vector<session_max> static sessions;

const char * initial(const char* model_path) {
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    Ort::Env static env(ORT_LOGGING_LEVEL_WARNING, "max");
    meta.str("");
    env_global = &env;
    // initialize session options if needed
    Ort::SessionOptions static session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    Ort::Session session(*env_global, model_path, session_options);
    
    session_max sei;
    psession_options = &session_options;

    Ort::AllocatorWithDefaultOptions allocator;
    meta<<sessions.size();
    meta<<" NN, Number of inputs:";
    
    
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    
    size_t input_tensor_size;
    int64_t size_label;
    
    
    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    input_node_names.resize(num_input_nodes);
    meta<<num_input_nodes<<"\n";
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // input/output node names
        char* input_name = session.GetInputName(i, allocator);
        input_node_names[i] = input_name;
        meta<<"Input " << i << ": name= " << input_name<< "\n";
        
        char* output_name = session.GetOutputName(i, allocator);
        meta<< "Output " << i << ": name= " << output_name<< "\n";
        output_node_names = {output_name};
        
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
    }

    input_tensor_size = input_node_dims.back();
    // create input tensor object from data values
    std::vector<float> input_tensor_values(input_tensor_size);

    for (unsigned int i = 0; i < input_tensor_size; i++)
            input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
//    assert(input_tensor.IsTensor());
////
    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    size_label = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape().back();
   
    meta<<"Label size: "<<size_label<<"\t"<<"Input size: "<<input_tensor_size;

    std::string s = meta.str();
    char* ps = new char[s.length() + 1];
    std::copy(s.c_str(), s.c_str() + s.length() + 1, ps);
    
    
    sei.input_tensor_size=input_tensor_size;
    sei.input_node_dims =input_node_dims;
    sei.input_node_names =input_node_names;
    sei.size_label =size_label;
    sei.output_node_names=output_node_names;
    sei.model_path=model_path;
    sessions.push_back(sei);
    
    return  ps;
    
    //*************************************************************************
}


extern "C"
float * inference (float * input_max,float * output_max, int input_size, long si)
{
    Ort::Session session(*env_global, sessions[si].model_path, *psession_options);
    
    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_max, sessions[si].input_tensor_size, sessions[si].input_node_dims.data(), sessions[si].input_node_dims.size());
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, sessions[si].input_node_names.data(), &input_tensor, 1, sessions[si].output_node_names.data(), 1);
    assert(output_tensors.front().IsTensor());
//
//    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    for (int i = 0; i < sessions[si].size_label; i++)
        output_max[i]=floatarr[i];
    return output_max;
    
}




extern "C"
float * test (float * testf)
{
    return testf;
}





//int main(int argc, const char * argv[]) {
//    //*************************************************************************
//    // initialize  enviroment...one enviroment per process
//    // enviroment maintains thread pools and other state info
//    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//
//    // initialize session options if needed
//    Ort::SessionOptions session_options;
//    session_options.SetIntraOpNumThreads(1);
//
//    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
//    // session (we also need to include cuda_provider_factory.h above which defines it)
//    // #include "cuda_provider_factory.h"
//    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);
//
//    // Sets graph optimization level
//    // Available levels are
//    // ORT_DISABLE_ALL -> To disable all optimizations
//    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
//    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
//    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
//    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//
//    //*************************************************************************
//    // create session and load model into memory
//    // using squeezenet version 1.3
//    // URL = https://github.com/onnx/models/tree/master/squeezenet
//#ifdef _WIN32
//    const wchar_t* model_path = L"/Users/shalti/Desktop/pianoset/alexnet.onnx";
//#else
//    const char* model_path = "/Users/shalti/Desktop/pianoset/alexnet.onnx";
//#endif
//
//    printf("Using Onnxruntime C++ API\n");
//    Ort::Session session(env, model_path, session_options);
//
//    //*************************************************************************
//    // print model input layer (node names, types, shape etc.)
//    Ort::AllocatorWithDefaultOptions allocator;
//
//    // print number of model input nodes
//    size_t num_input_nodes = session.GetInputCount();
//    std::vector<const char*> input_node_names(num_input_nodes);
//    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
//    // Otherwise need vector<vector<>>
//
//    printf("Number of inputs = %zu\n", num_input_nodes);
//    // iterate over all input nodes
//    for (int i = 0; i < num_input_nodes; i++) {
//        // print input node names
//        char* input_name = session.GetInputName(i, allocator);
//
//        printf("Input %d : name=%s\n", i, input_name);
//        input_node_names[i] = input_name;
//
//        char* output_name = session.GetOutputName(i, allocator);
//        printf("output %d : name=%s\n", i, output_name);
//
//        // print input node types
//        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
//        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//
//        ONNXTensorElementDataType type = tensor_info.GetElementType();
//        printf("Input %d : type=%d\n", i, type);
//
//        // print input shapes/dims
//        input_node_dims = tensor_info.GetShape();
//        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
//        for (int j = 0; j < input_node_dims.size(); j++)
//            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
//    }
//
//    // Results should be...
//    // Number of inputs = 1
//    // Input 0 : name = data_0
//    // Input 0 : type = 1
//    // Input 0 : num_dims = 4
//    // Input 0 : dim 0 = 1
//    // Input 0 : dim 1 = 3
//    // Input 0 : dim 2 = 224
//    // Input 0 : dim 3 = 224
//
//    //*************************************************************************
//    // Similar operations to get output node information.
//    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
//    // OrtSessionGetOutputTypeInfo() as shown above.
//
//    //*************************************************************************
//    // Score the model using sample data, and inspect values
//
//    size_t input_tensor_size = 1024 * 2 * 1;  // simplify ... using known dim values to calculate size
//    // use OrtGetTensorShapeElementCount() to get official size!
//
//    std::vector<float> input_tensor_values(input_tensor_size);
//    std::vector<const char*> output_node_names = {"7"};
//
//    // initialize input data with values in [0.0, 1.0]
//    for (unsigned int i = 0; i < input_tensor_size; i++)
//        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
//
//    // create input tensor object from data values
//    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 3);
//    assert(input_tensor.IsTensor());
//
//    // score model & input tensor, get back output tensor
//    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
//    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
//
//    // Get pointer to output tensor float values
//    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
////assert(abs(floatarr[0] - 0.000045) < 1e-6);
//
//    // score the model, and print scores for first 5 classes
//    for (int i = 0; i < 88; i++)
//        printf("Score for class [%d] =  %f\n", i, floatarr[i]);
//
//    // Results should be as below...
//    // Score for class[0] = 0.000045
//    // Score for class[1] = 0.003846
//    // Score for class[2] = 0.000125
//    // Score for class[3] = 0.001180
//    // Score for class[4] = 0.001317
//    printf("Done!\n");
//    return 0;
//}
