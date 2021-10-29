#include "superglue.hpp"
#include <iostream>
#include <fstream>

using namespace std;

SuperGlue::SuperGlue(){
    std::cout<<"L2Norm Not Creat !!!"<<std::endl;
}

SuperGlue::~SuperGlue(){}

SuperGlue::SuperGlue(const char* modelPath){
    this->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath));
    this->backendConfig.precision = (MNN::BackendConfig::PrecisionMode) this->precision;
    this->backendConfig.power = (MNN::BackendConfig::PowerMode) this->power;
    this->backendConfig.memory = (MNN::BackendConfig::MemoryMode) this->memory;
    this->config.backendConfig = & this->backendConfig;
    this->config.type = this->forward;
    this->session = this->net->createSession(this->config);
    std::cout<<"SuperPointNet Creat Done !!!"<<std::endl;
}


void SuperGlue::Inference(std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc1,
                                               std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts1,
                                               std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores1,
                                               int realDim0,int realDim1){
    // std::string desc0InputTensor = "desc0";
    auto SuperGlueSessionInTensorDesc0 = this->net->getSessionInput(this->session,this->desc0InputTensor.c_str());
    this->net->resizeTensor(SuperGlueSessionInTensorDesc0, {1, 256, realDim0});

    // std::string desc1InputTensor = "desc1";
    auto SuperGlueSessionInTensorDesc1 = this->net->getSessionInput(this->session, this->desc1InputTensor.c_str());
    this->net->resizeTensor(SuperGlueSessionInTensorDesc1, {1, 256, realDim1});

    // std::string kpts0InputTensor = "kpts0";
    auto SuperGlueSessionInTensorkpts0 = this->net->getSessionInput(this->session, this->kpts0InputTensor.c_str());
    this->net->resizeTensor(SuperGlueSessionInTensorkpts0, {1,realDim0,2});

    // std::string kpts1InputTensor = "kpts1";
    auto SuperGlueSessionInTensorkpts1 = this->net->getSessionInput(this->session, this->kpts1InputTensor.c_str());
    this->net->resizeTensor(SuperGlueSessionInTensorkpts1, {1,realDim1,2});

    // std::string scores0InputTensor = "scores0";
    auto SuperGlueSessionInTensorscores0 = this->net->getSessionInput(this->session, this->scores0InputTensor.c_str());
    this->net->resizeTensor(SuperGlueSessionInTensorscores0, {1,realDim0});

    // std::string scores1InputTensor = "scores1";
    auto SuperGlueSessionInTensorscores1 = this->net->getSessionInput(this->session, this->scores1InputTensor.c_str());
    this->net->resizeTensor(SuperGlueSessionInTensorscores1, {1,realDim1});
    
    this->net->resizeSession(this->session); 

    auto SuperGlueDesc0ResizeinTensor = std::shared_ptr<MNN::Tensor> (new MNN::Tensor(SuperGlueSessionInTensorDesc0, MNN::Tensor::CAFFE));
    ::memcpy(SuperGlueDesc0ResizeinTensor->host<float>(), SuperGlueinTensorDesc0->host<float>(), SuperGlueDesc0ResizeinTensor->elementSize() * 4);
    SuperGlueSessionInTensorDesc0->copyFromHostTensor(SuperGlueDesc0ResizeinTensor.get());

    auto SuperGlueDesc1ResizeinTensor = std::shared_ptr<MNN::Tensor> (new MNN::Tensor(SuperGlueSessionInTensorDesc1, MNN::Tensor::CAFFE));
    ::memcpy(SuperGlueDesc1ResizeinTensor->host<float>(), SuperGlueinTensorDesc1->host<float>(), SuperGlueDesc1ResizeinTensor->elementSize() * 4);
    SuperGlueSessionInTensorDesc1->copyFromHostTensor(SuperGlueDesc1ResizeinTensor.get());

    auto SuperGluekpts0ResizeinTensor = std::shared_ptr<MNN::Tensor> (new MNN::Tensor(SuperGlueSessionInTensorkpts0, MNN::Tensor::CAFFE));
    ::memcpy(SuperGluekpts0ResizeinTensor->host<float>(), SuperGlueinTensorkpts0->host<float>(), SuperGluekpts0ResizeinTensor->elementSize() * 4);
    SuperGlueSessionInTensorkpts0->copyFromHostTensor(SuperGluekpts0ResizeinTensor.get());

    auto SuperGluekpts1ResizeinTensor = std::shared_ptr<MNN::Tensor> (new MNN::Tensor(SuperGlueSessionInTensorkpts1, MNN::Tensor::CAFFE));
    ::memcpy(SuperGluekpts1ResizeinTensor->host<float>(), SuperGlueinTensorkpts1->host<float>(), SuperGluekpts1ResizeinTensor->elementSize() * 4);
    SuperGlueSessionInTensorkpts1->copyFromHostTensor(SuperGluekpts1ResizeinTensor.get());

    auto SuperGluescores0ResizeinTensor = std::shared_ptr<MNN::Tensor> (new MNN::Tensor(SuperGlueSessionInTensorscores0, MNN::Tensor::CAFFE));
    ::memcpy(SuperGluescores0ResizeinTensor->host<float>(), SuperGlueinTensorscores0->host<float>(), SuperGluescores0ResizeinTensor->elementSize() * 4);
    SuperGlueSessionInTensorscores0->copyFromHostTensor(SuperGluescores0ResizeinTensor.get());

    auto SuperGluescores1ResizeinTensor = std::shared_ptr<MNN::Tensor> (new MNN::Tensor(SuperGlueSessionInTensorscores1, MNN::Tensor::CAFFE));
    ::memcpy(SuperGluescores1ResizeinTensor->host<float>(), SuperGlueinTensorscores1->host<float>(), SuperGluescores1ResizeinTensor->elementSize() * 4);
    SuperGlueSessionInTensorscores1->copyFromHostTensor(SuperGluescores1ResizeinTensor.get());
    this->net->runSession(this->session);

}

std::shared_ptr<MNN::Tensor> SuperGlue::GetIndices0Value(){
    auto output= this->net->getSessionOutput(this->session, this->indices0OutputName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

std::shared_ptr<MNN::Tensor> SuperGlue::GetIndices1Value(){
    auto output= this->net->getSessionOutput(this->session, this->indices1OutputName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

std::shared_ptr<MNN::Tensor> SuperGlue::GetScores0Value(){
    auto output= this->net->getSessionOutput(this->session, this->mscores0OutputName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

std::shared_ptr<MNN::Tensor> SuperGlue::GetScores1Value(){
    auto output= this->net->getSessionOutput(this->session, this->mscores1OutputName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}
