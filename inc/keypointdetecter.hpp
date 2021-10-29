#pragma once

#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/ImageProcess.hpp>
#include<vector>

#include "l2norm.hpp"
#include "superglue.hpp"
#include "superpoint.hpp"

using namespace std;
    
class KeyPointsDetecter{
    private:
        std::shared_ptr<SuperGlue> superGlue = std::shared_ptr<SuperGlue> (new SuperGlue());
        std::shared_ptr<L2NormModel> l2Norm = std::shared_ptr<L2NormModel> (new L2NormModel());
        std::shared_ptr<SuperPoint> superPoint = std::shared_ptr<SuperPoint> (new SuperPoint());
        
    public:
        KeyPointsDetecter();
        KeyPointsDetecter(const char* superpoint_model_name,const char* l2_model_name,
                          const char* superglue_model_name);
        
        void InderenceSpuerPoint(const cv::Mat& image);
        //获取superpoint中的scores值
        std::shared_ptr<MNN::Tensor> GetSpuerPointScores();
        //获取superpoint中的kpts值
        std::shared_ptr<MNN::Tensor> GetSpuerPointKeypoints();
        //获取superpoint中的desc值
        std::shared_ptr<MNN::Tensor> GetSpuerPointDesc();
        
        void InderenceSpuerGlue(std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorDesc1,
                                std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorkpts1,
                                std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores0,std::shared_ptr<MNN::Tensor> SuperGlueinTensorscores1,
                                int readDim0,int readDim1);

        //获取superglue中对应python代码中的matches0
        std::shared_ptr<MNN::Tensor> GetSuperGlueIndices0();
        //获取superglue中对应python代码中的matches1
        std::shared_ptr<MNN::Tensor> GetSuperGlueIndices1();
        //获取superglue中对应python代码中的mscores0
        std::shared_ptr<MNN::Tensor> GetSuperGlueScores0();
        //获取superglue中对应python代码中的mscores1
        std::shared_ptr<MNN::Tensor> GetSuperGlueScores1();

        //获取最终结果以及显示
        void DecodeResult(
                          const cv::Mat image0,const cv::Mat image1,
                          std::shared_ptr<MNN::Tensor> indices0,std::shared_ptr<MNN::Tensor> indices1,
                          std::shared_ptr<MNN::Tensor> scores0,std::shared_ptr<MNN::Tensor> scores1,
                          std::shared_ptr<MNN::Tensor> kpts0,std::shared_ptr<MNN::Tensor> kpts1,
                          vector<cv::Point2f>& mkpts0,vector<cv::Point2f>& mkpts1,
                          vector<float> conf,bool isShow,string savePath);

        int GetReadDim();


        ~KeyPointsDetecter();
};