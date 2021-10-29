#pragma once

#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/ImageProcess.hpp>

class L2NormModel{
    private:
        std::shared_ptr<MNN::Interpreter> net = nullptr;
        const char* modelPath;
        MNN::ScheduleConfig config;
        MNN::Session *session = nullptr;
        MNN::BackendConfig backendConfig;
        /*MNN 后端配置*/  
        MNNForwardType forward = MNN_FORWARD_CPU;
        int threads = 1;
        int precision = 0;
        int power = 0;
        int memory = 0;
    public:
        L2NormModel();
        // //初始化模型
        L2NormModel(const char* modelPath);
        // float* Inference(float* input,int realDim);
        std::shared_ptr<MNN::Tensor> Inference(std::shared_ptr<MNN::Tensor> inputTensor,int realDim);
        ~L2NormModel();
};