#pragma once

#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/ImageProcess.hpp>

class SuperGlue{
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

        std::string indices0OutputName = "indices0";
        std::string indices1OutputName = "indices1";
        std::string mscores0OutputName = "mscores0";
        std::string mscores1OutputName = "mscores1";

        std::string desc0InputTensor = "desc0";
        std::string desc1InputTensor = "desc1";
        std::string kpts0InputTensor = "kpts0";
        std::string kpts1InputTensor = "kpts1";
        std::string scores0InputTensor = "scores0";
        std::string scores1InputTensor = "scores1";
        
    public:

        SuperGlue();
        // //初始化模型
        SuperGlue(const char* modelPath);
        void Inference(std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc1,
                                               std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts1,
                                               std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores1,
                                               int realDim0,int realDim1);
        //对应python代码中的matches0
        std::shared_ptr<MNN::Tensor> GetIndices0Value();
        //对应python代码中的matches1
        std::shared_ptr<MNN::Tensor> GetIndices1Value();
        //对应python代码中的mscores0
        std::shared_ptr<MNN::Tensor> GetScores0Value();
        //对应python代码中的mscores1
        std::shared_ptr<MNN::Tensor> GetScores1Value();
        ~SuperGlue();
};