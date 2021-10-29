#include "l2norm.hpp"

L2NormModel::L2NormModel(){
    std::cout<<"L2Norm Not Creat !!!"<<std::endl;
}

L2NormModel::L2NormModel(const char* modelPath){
    this->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath));
    this->backendConfig.precision = (MNN::BackendConfig::PrecisionMode) this->precision;
    this->backendConfig.power = (MNN::BackendConfig::PowerMode) this->power;
    this->backendConfig.memory = (MNN::BackendConfig::MemoryMode) this->memory;
    this->config.backendConfig = & this->backendConfig;
    this->config.type = this->forward;
    this->session = this->net->createSession(this->config);
    std::cout<<"L2Norm Creat Done !!!"<<std::endl;
}

std::shared_ptr<MNN::Tensor> L2NormModel::Inference(std::shared_ptr<MNN::Tensor> inputTensor,int realDim){
    auto L2inTensor = this->net->getSessionInput(this->session, NULL);
    this->net->resizeTensor(L2inTensor, {1, 256, realDim});
    this->net->resizeSession(this->session); 
    auto L2ResizeTensor = std::shared_ptr<MNN::Tensor>(new MNN::Tensor(L2inTensor, MNN::Tensor::CAFFE));
    ::memcpy(L2ResizeTensor->host<float>(), inputTensor->host<float>(), L2ResizeTensor->elementSize() * 4);
    L2inTensor->copyFromHostTensor(L2ResizeTensor.get());
    std::string L2NormTensor = "normoutput";
    this->net->runSession(this->session);
    auto L2Output= this->net->getSessionOutput(this->session, L2NormTensor.c_str());
    auto L2OutputTensor = std::shared_ptr<MNN::Tensor>(new MNN::Tensor(L2Output, MNN::Tensor::CAFFE));
    L2Output->copyToHostTensor(L2OutputTensor.get());
    // delete L2ResizeTensor;
    return L2OutputTensor;
}

L2NormModel::~L2NormModel(){}