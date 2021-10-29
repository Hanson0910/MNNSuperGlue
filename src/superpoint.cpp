#include "superpoint.hpp"
#include "common.hpp"
#include <fstream>
SuperPoint::SuperPoint(){};
SuperPoint::~SuperPoint(){};

SuperPoint::SuperPoint(const char* modelPath){
    this->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath));
    this->backendConfig.precision = (MNN::BackendConfig::PrecisionMode) this->precision;
    this->backendConfig.power = (MNN::BackendConfig::PowerMode) this->power;
    this->backendConfig.memory = (MNN::BackendConfig::MemoryMode) this->memory;
    this->config.backendConfig = & this->backendConfig;
    this->config.type = this->forward;
    this->session = this->net->createSession(this->config);
    std::cout<<"SuperPointNet Creat Done !!!"<<std::endl;
}

void SuperPoint::Mat2Tensor(const cv::Mat& image){
    cv::Mat preImage = image.clone();
    preImage.convertTo(preImage,CV_32FC3,1/255.);
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(preImage, bgrChannels);
    std::vector<float> chwImage;
    for (auto i = 0; i < bgrChannels.size(); i++)
    {  
        //HWC->CHW
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, preImage.cols * preImage.rows));
        chwImage.insert(chwImage.end(), data.begin(), data.end());
    }
    auto inTensor = net->getSessionInput(session, NULL);
    //std::cout<<"Hiii: "<<inTensor->shape()[0]<<" "<<inTensor->shape()[1]<<" "<<inTensor->shape()[2]<<" "<<inTensor->shape()[3]<<std::endl;
	auto nchwTensor = shared_ptr<MNN::Tensor> (new MNN::Tensor(inTensor, MNN::Tensor::CAFFE));
    ::memcpy(nchwTensor->host<float>(), chwImage.data(), nchwTensor->elementSize() * 4);
    inTensor->copyFromHostTensor(nchwTensor.get());
}

void SuperPoint::Inference(const cv::Mat& image){
     Mat2Tensor(image);
     this->net->runSession(this->session);
 }

shared_ptr<MNN::Tensor> SuperPoint::GetScoresValue(){
    auto output= this->net->getSessionOutput(this->session, this->scoresOutName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

shared_ptr<MNN::Tensor> SuperPoint::GetKeypointsValue(){
    auto output= this->net->getSessionOutput(this->session, this->keypointsOutName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

shared_ptr<MNN::Tensor> SuperPoint::GetDescriptorsValue(){
    auto keypoints = this->GetKeypointsValue();
    auto scores =  this->GetScoresValue();
    int realDim = this->GetRealDim(scores->host<float>());
    this->realDim = realDim;
    auto output= this->net->getSessionOutput(this->session, this->descriptorsOutName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    std::vector<int> inputDims{1,256,realDim};
    auto descFinalTensor = (shared_ptr<MNN::Tensor>) MNN::Tensor::create<float>(inputDims, NULL, MNN::Tensor::CAFFE);
    auto descFinalOutput = sample_descriptors(keypoints->host<float>(),outputTensor->host<float>(),outputTensor->shape(),realDim,8);
    ::memcpy(descFinalTensor->host<float>(),descFinalOutput.get(),descFinalTensor->elementSize() * 4);
    return descFinalTensor;
}

int SuperPoint::GetRealDim(float* scores){
    return getRealDim(scores);
}