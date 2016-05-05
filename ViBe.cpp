#include <opencv.hpp>  
#include <iostream>  
#include "ViBe.h"  

using namespace std;
using namespace cv;

int c_xoff[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };  //x的鄰近點  
int c_yoff[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };  //y的鄰近點 

Mat gray;

ViBe_BGS::ViBe_BGS(void){

}
ViBe_BGS::~ViBe_BGS(void){

}

/**************** Assign space and init ***************************/
void ViBe_BGS::init(const Mat _image)
{
	cvtColor(_image, gray, CV_RGB2GRAY);
	
	for (int i = 0; i < NUM_SAMPLES; i++)
	{
		m_samples[i] = Mat::zeros(gray.size(), CV_8UC1);
	}
	m_mask = Mat::zeros(gray.size(), CV_8UC1);
	m_foregroundMatchCount = Mat::zeros(gray.size(), CV_8UC1);
}

/**************** Init model from first frame ********************/
void ViBe_BGS::processFirstFrame(const Mat _image)
{
	RNG rng;
	int row, col;
	cvtColor(_image, gray, CV_RGB2GRAY);

	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			for (int k = 0; k < NUM_SAMPLES; k++)
			{
				// Random pick up NUM_SAMPLES pixel in neighbourhood to construct the model  
				int random = rng.uniform(0, 9);

				row = i + c_yoff[random];
				if (row < 0)
					row = 0;
				if (row >= gray.rows)
					row = gray.rows - 1;

				col = j + c_xoff[random];
				if (col < 0)
					col = 0;
				if (col >= gray.cols)
					col = gray.cols - 1;

				m_samples[k].at<uchar>(i, j) = gray.at<uchar>(row, col);
			}
		}
	}
}

/**************** Test a new frame and update model ********************/
void ViBe_BGS::testAndUpdate(const Mat _image, int* foreground_nums)
{
	RNG rng;
	
	cvtColor(_image, gray, CV_RGB2GRAY);
	*foreground_nums = 0;
	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			int matches(0), count(0);
			float dist;

			while (matches < MIN_MATCHES && count < NUM_SAMPLES)
			{
				dist = abs(m_samples[count].at<uchar>(i, j) - gray.at<uchar>(i, j));
				if (dist < RADIUS)
					matches++;
				count++;
			}

			if (matches >= MIN_MATCHES)
			{
				// It is a background pixel  
				m_foregroundMatchCount.at<uchar>(i, j) = 0;

				// Set background pixel to 0  
				m_mask.at<uchar>(i, j) = 0;

				// 如果一个像素是背景点，那么它有 1 / defaultSubsamplingFactor 的概率去更新自己的模型样本值  
				int random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0)
				{
					random = rng.uniform(0, NUM_SAMPLES);
					m_samples[random].at<uchar>(i, j) = gray.at<uchar>(i, j);
				}

				// 同时也有 1 / defaultSubsamplingFactor 的概率去更新它的邻居点的模型样本值  
				random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0)
				{
					int row, col;
					random = rng.uniform(0, 9);
					row = i + c_yoff[random];
					if (row < 0)
						row = 0;
					if (row >= gray.rows)
						row = gray.rows - 1;

					random = rng.uniform(0, 9);
					col = j + c_xoff[random];
					if (col < 0)
						col = 0;
					if (col >= gray.cols)
						col = gray.cols - 1;

					random = rng.uniform(0, NUM_SAMPLES);
					m_samples[random].at<uchar>(row, col) = gray.at<uchar>(i, j);
				}
			}
			else
			{
				// It is a foreground pixel  
				m_foregroundMatchCount.at<uchar>(i, j)++;

				// Set foreground pixel to 255  
				m_mask.at<uchar>(i, j) = 255;
				*foreground_nums = *foreground_nums + 1;
				
				//如果某个像素点连续N次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点  
				if (m_foregroundMatchCount.at<uchar>(i, j) > 60)
				{
					int random = rng.uniform(0, SUBSAMPLE_FACTOR);
					if (random == 0)
					{
						random = rng.uniform(0, NUM_SAMPLES);
						m_samples[random].at<uchar>(i, j) = gray.at<uchar>(i, j);
					}
				}
			}
		}
	}
}