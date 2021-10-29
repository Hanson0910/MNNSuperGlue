#include "superpoint.hpp"
#include "l2norm.hpp"
#include "superglue.hpp"
#include "keypointdetecter.hpp"
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

int main(){
    const char* superpoint_model_name = "yourpath/models/SuperPoint.mnn";
    const char* l2_model_name = "yourpath/models/L2Norm.mnn";
    const char* superglue_model_name = "yourpath/models/SuperGlue.mnn";
    
    KeyPointsDetecter kptsDeter = KeyPointsDetecter(superpoint_model_name,l2_model_name,superglue_model_name);
    
    cv::Mat image0 = cv::imread("yourImage0Path",0);
    cv::resize(image0,image0,cv::Size(640,480),0,0);

    cv::Mat image1 = cv::imread("yourImage1Path",0);
    cv::resize(image1,image1,cv::Size(640,480),0,0);

    kptsDeter.InderenceSpuerPoint(image0);
    auto spScores0 = kptsDeter.GetSpuerPointScores();
    auto kpts0 = kptsDeter.GetSpuerPointKeypoints();
    auto desc0 = kptsDeter.GetSpuerPointDesc();
    int realDim0 = kptsDeter.GetReadDim();
    
    kptsDeter.InderenceSpuerPoint(image1);
    auto spScores1 = kptsDeter.GetSpuerPointScores();
    auto kpts1 = kptsDeter.GetSpuerPointKeypoints();
    auto desc1 = kptsDeter.GetSpuerPointDesc();
    int realDim1 = kptsDeter.GetReadDim();

    kptsDeter.InderenceSpuerGlue(desc0,desc1,kpts0,kpts1,spScores0,spScores1,realDim0,realDim1);

    auto scores0 = kptsDeter.GetSuperGlueScores0();
    auto scores1 = kptsDeter.GetSuperGlueScores1();
    auto indices0 = kptsDeter.GetSuperGlueIndices0();
    auto indices1 = kptsDeter.GetSuperGlueIndices1();

    vector<cv::Point2f> mkpts0;
    vector<cv::Point2f> mkpts1;
    vector<float> conf;
    kptsDeter.DecodeResult(image0,image1,indices0,indices1,scores0,scores1,kpts0,kpts1,mkpts0,mkpts1,conf,true,"result.jpg");
    return 0;
}