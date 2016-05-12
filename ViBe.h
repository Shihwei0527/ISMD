#pragma once  
#include <iostream>  
#include <opencv.hpp>

using namespace cv;
using namespace std;

// fixed parameter for ViBe

#define NUM_SAMPLES 20 // 每個像素點的樣本個數 number of samples per pixel
#define MIN_MATCHES 2 // #min指數 number of close samples for being part of the background
#define RADIUS 20 // Sqthere半徑 radius of the pshere
#define SUBSAMPLE_FACTOR 1 //子採樣率 amount of random subsampling，越小背景更新越快


class ViBe_BGS
{
public:
	ViBe_BGS(void);
	~ViBe_BGS(void);

	void init(const Mat _image);   //初始化  
	void processFirstFrame(const Mat _image);
	void testAndUpdate(const Mat _image, int *foreground_nums);  //更新  
	Mat getMask(void){ return m_mask; };

private:
	Mat m_samples[NUM_SAMPLES];
	Mat m_foregroundMatchCount;
	Mat m_mask;
};