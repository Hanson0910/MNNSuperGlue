#include "common.hpp"

float safeGet(float* values,int x,int y, int n,int c,int C,int H,int W){
    float value = 0.;
    if(x >= 0 && x < W && y >= 0 && y < H)
        return values[(((n * C)+c)*H + y) * W + x];
    return value;
}

int getRealDim(float* hostValue){
    for(int i = 1; i < 5000; i++){
        if(hostValue[i] == hostValue[0]){
            if(i < 4999 && hostValue[i+1] == hostValue[0])
                return i;
        }
    }
    return 5000;
}

shared_ptr<float> grid_sample(float* keypoints,float* descriptors,int N,int C,int IH,int IW,int H,int W){
    auto output = shared_ptr<float> (new float[1*C*1*W]);
    for(int n = 0; n < N; n++){
        for(int h = 0; h < H; h++){
            for(int w = 0; w < W; w++){
                float ix = keypoints[2*w];
                float iy = keypoints[2*w+1];
                ix = ((ix + 1) / 2.) * (IW - 1);
                iy = ((iy + 1) / 2.) * (IH - 1);
                int ix_nw = floor(ix);
                int iy_nw = floor(iy);
                int ix_ne = ix_nw + 1;
                int iy_ne = iy_nw;
                int ix_sw = ix_nw;
                int iy_sw = iy_nw + 1;
                int ix_se = ix_nw + 1;
                int iy_se = iy_nw + 1;
                float nw = (ix_se - ix)    * (iy_se - iy);
                float ne = (ix    - ix_sw) * (iy_sw - iy);
                float sw = (ix_ne - ix)    * (iy    - iy_ne);
                float se = (ix    - ix_nw) * (iy    - iy_nw);
                for(int c = 0; c < C; c++){
                    float nw_val = safeGet(descriptors, ix_nw, iy_nw, n, c, C,IH, IW);
                    float ne_val = safeGet(descriptors, ix_ne, iy_ne, n, c, C,IH, IW);
                    float sw_val = safeGet(descriptors, ix_sw, iy_sw, n, c, C,IH, IW);
                    float se_val = safeGet(descriptors, ix_se, iy_se, n, c, C,IH, IW);
                    float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
                    output.get()[(((n * C)+c)*H + h) * W + w] = out_val;
                }
            }
        }
    }
    return output;
}

shared_ptr<float> sample_descriptors(float* keypoints,float* descriptors,vector<int> shape, int realDim, int s){
    int N = shape[0];
    int C = shape[1];
    int IH = shape[2];
    int IW = shape[3];
    int H = 1;
    int W = realDim;
    //normalize keypointes to (-1,1)
    float* tempKeypoints = new float[realDim*2];  
    memcpy(tempKeypoints,keypoints,realDim * 2 * 4);
    for(int i = 0; i < realDim; i++){
        tempKeypoints[2*i] = tempKeypoints[2*i] - s / 2. + .5;
        tempKeypoints[2*i + 1] = tempKeypoints[2*i + 1] - s / 2. + .5;
        tempKeypoints[2*i] = tempKeypoints[2*i] / (IW*s - s/ 2. - 0.5) * 2. - 1.;
        tempKeypoints[2*i + 1] = tempKeypoints[2*i + 1] / (IH*s - s/ 2. - 0.5) * 2. - 1.;
    }
    auto output = grid_sample(tempKeypoints,descriptors,N,C,IH,IW,H,W);
    delete [] tempKeypoints;
    return output;
}