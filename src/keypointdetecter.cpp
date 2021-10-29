#include <iostream>
#include "common.hpp"
#include "keypointdetecter.hpp"
using namespace std;
using namespace cv;

KeyPointsDetecter::KeyPointsDetecter(){}

KeyPointsDetecter::~KeyPointsDetecter(){}

KeyPointsDetecter::KeyPointsDetecter(const char* superpoint_model_name,const char* l2_model_name,
                  const char* superglue_model_name){
    this->superGlue = std::shared_ptr<SuperGlue> (new SuperGlue(superglue_model_name));
    this->l2Norm = std::shared_ptr<L2NormModel> (new L2NormModel(l2_model_name));
    this->superPoint =  std::shared_ptr<SuperPoint> (new SuperPoint(superpoint_model_name));
}

void KeyPointsDetecter::InderenceSpuerPoint(const cv::Mat& image){
    this->superPoint->Inference(image);
}

std::shared_ptr<MNN::Tensor> KeyPointsDetecter::GetSpuerPointKeypoints(){
    return this->superPoint->GetKeypointsValue();
}

std::shared_ptr<MNN::Tensor> KeyPointsDetecter::GetSpuerPointScores(){
    return this->superPoint->GetScoresValue();
}

std::shared_ptr<MNN::Tensor> KeyPointsDetecter::GetSpuerPointDesc(){
    auto output = superPoint->GetDescriptorsValue();
    return this->l2Norm->Inference(output,this->superPoint->realDim);
}

void KeyPointsDetecter::InderenceSpuerGlue(std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc1,
                                                                    std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts1,
                                                                    std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores1,
                                                                    int readDim0,int readDim1){                                               
                                                           
    this->superGlue->Inference(SuperGlueinTensorDesc0,SuperGlueinTensorDesc1,
                               SuperGlueinTensorkpts0,SuperGlueinTensorkpts1,
                               SuperGlueinTensorscores0,SuperGlueinTensorscores1,
                               readDim0,readDim1);
}



std::shared_ptr<MNN::Tensor> KeyPointsDetecter::GetSuperGlueIndices0(){
    return this->superGlue->GetIndices0Value();
}

std::shared_ptr<MNN::Tensor> KeyPointsDetecter::GetSuperGlueIndices1(){
    return this->superGlue->GetIndices1Value();
}

std::shared_ptr<MNN::Tensor> KeyPointsDetecter::GetSuperGlueScores0(){
    return this->superGlue->GetScores0Value();
}


std::shared_ptr<MNN::Tensor> KeyPointsDetecter::GetSuperGlueScores1(){
    return this->superGlue->GetScores1Value();
}


void KeyPointsDetecter::DecodeResult(
                          const cv::Mat image0,const cv::Mat image1,
                          std::shared_ptr<MNN::Tensor> indices0,std::shared_ptr<MNN::Tensor> indices1,
                          std::shared_ptr<MNN::Tensor> scores0,std::shared_ptr<MNN::Tensor> scores1,
                          std::shared_ptr<MNN::Tensor> kpts0,std::shared_ptr<MNN::Tensor> kpts1,
                          vector<cv::Point2f>& mkpts0,vector<cv::Point2f>& mkpts1,
                          vector<float> conf,bool isShow,string savePath){
    for(int i = 0; i < scores0->shape()[1]; i++){
        if(indices0->host<float>()[i] > -1){
            int kp1Index = (int)indices0->host<float>()[i];
            mkpts0.push_back(cv::Point2f(kpts0->host<float>()[2*i],kpts0->host<float>()[2*i+1]));
            mkpts1.push_back(cv::Point2f(kpts1->host<float>()[2*kp1Index],kpts1->host<float>()[2*kp1Index+1]));
            conf.push_back(scores0->host<float>()[i]);
        }    
    }
    if(isShow){
    int margin = 10;
    cv::Mat showImg(image0.rows,image0.cols * 2 + margin,CV_8UC1,Scalar(128));
    
    cv::Mat rectImage0 = showImg(Rect(0,0,image0.cols,image0.rows));
    addWeighted(rectImage0, 0.2, image0, 0.8, 0., rectImage0);

    cv::Mat rectImage1 = showImg(Rect(0 + image1.cols + margin,0,image1.cols,image1.rows));
    addWeighted(rectImage1, 0.2, image1, 0.8, 0., rectImage1);

    for(int i = 0; i < mkpts0.size(); i++){
        cv::line(showImg,cv::Point2d((int)mkpts0[i].x,(int)mkpts0[i].y),cv::Point2d((int)mkpts1[i].x + margin + image0.cols,(int)mkpts1[i].y),Scalar(0,0,255),1);
        cv::circle(showImg,cv::Point2d((int)mkpts0[i].x,(int)mkpts0[i].y),2,Scalar(0,255,0));
        cv::circle(showImg,cv::Point2d((int)mkpts1[i].x + margin + image0.cols,(int)mkpts1[i].y),2,Scalar(0,255,0));
    }
    cv::imwrite(savePath,showImg);
}
}

int KeyPointsDetecter::GetReadDim(){
    return this->superPoint->realDim;
}

