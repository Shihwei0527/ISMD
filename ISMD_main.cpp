﻿// ****************************************************************************************************
// *** 整合式的全周監視與辨識系統 (Integrated Surrounding Monitor and Recognition System)
// ****************************************************************************************************

// ***  建立背景的三種方法，用 mark 的方式，擇一使用 *************
// ***  1. ViBe		(Visual Background extractor)
// ***  2. CBM	(CodeBook Model)
// ***  3. GMM	(Gaussian Mixture Model)

 #define ViBe
// #define CBM
// #define GMM

// *** 定義使用相機個數，最多可以使用四部相機 ********************
#define camera_num 1		// 可更改相機個數 (1, 2, 3, or 4) 或讀影片檔個數

#include <opencv.hpp>
#include <iostream>
#include <cv.h>			
#include <highgui.h>

using namespace cv;
using namespace std;

const Scalar RED(0, 0, 255);		//*** 顯示紅色字體 ---------------------------------------------------
const string CameraAddress = "http://admin:ipvrnt2k@140.115.155.21/video1.mjpg";

//*** 若無接相機，則讀取下列影片 *****************************************************************
char *videos[] = { "../../image/Cloud160410.avi", "../../image/Cloud160412.avi", "../../image/Sun160408.avi", "../../image/Night160426.avi"};		//*** 各個影片檔的檔名
char WinName[15];
// char WinName2[15];
vector<vector<Point> > contours[camera_num];
Rect bBox[camera_num];
int	 imageLen[camera_num];

#ifdef ViBe		//*** 使用 Visual Background extractor 背景相減法 ------------------------------------
	#include "ViBe.h"
	ViBe_BGS Vibe_Bgs[camera_num];
#endif

#ifdef CBM	//*** 使用 CodeBook Model 背景相減法 ----------------------------------------------------
	#include "codebook.h"
	codeBook**	image_codebook;		// 指向 codebook 結構的 array，array的長度等於影像 pixel 數量
	unsigned	codeBook_Bounds[CHANNELS];
	unsigned char**		pColor; //YUV pointer	
	int		nChannels = CHANNELS;
	int		minMod[CHANNELS];
	int		maxMod[CHANNELS];
	Mat	yuvImage[camera_num];
	// Mat	ImaskCodeBook[camera_num]; // (600, 800, CV_8UC1, Scalar(0));
#endif

void put_Font(Mat &img)
{
	string msg = "Warning";
	int baseLine = 0;
	Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
	Point textOrigin(img.cols / 4, img.rows / 2);
	putText(img, msg, textOrigin, FONT_HERSHEY_SIMPLEX, 5, RED);
}

void DeNoise(Mat&input, Mat&output)
{
	// element_shape = MORPH_ELLIPSE;
	// element_shape = MORPH_RECT;
	// element_shape = MORPH_CROSS;
	int size = 1;
	Mat element = getStructuringElement(MORPH_CROSS, Size(size * 2 + 1, size * 2 + 1), Point(1, 1));

	morphologyEx(input, output, CV_MOP_OPEN, element);
	morphologyEx(output, output, CV_MOP_CLOSE, element);
	//element = getStructuringElement(MORPH_CROSS, Size(2 * 2 + 1, 2 * 2 + 1), Point(1, 1));
	dilate(output, output, element);
	dilate(output, output, element);
}

int main() {
	//*** 定義相機 **************************************************************************************
	VideoCapture capture[camera_num];
	Mat rawImage[camera_num], DeNoised_foreground[camera_num], resizedImg[camera_num];
	Mat foreground[camera_num];
	
	int foreground_nums = 0;
	Mat xywhdata[camera_num];
	int processing_cam;

	//capture1.open(CameraAddress);
	//capture1.open(video.avi);

	for (int i = 0; i < camera_num; i++)
	{
		//*** 所有相機各自讀取影像 *****************************************************
		capture[i].open(*(videos+i));

		if (!capture[i].isOpened())
		{
			cout << "No camera or video input!\n" << endl;
			return -1;
		}
		capture[i] >> rawImage[i];
		xywhdata[i].create(4, 1, CV_32F);
		//*** 若影像的寬度大於 1000 像素，則影像長寬各縮一半 ------------------------
		if (rawImage[i].rows > 1000 || rawImage[i].cols > 1000)
		{
			imageLen[i] = rawImage[i].cols*0.5 * rawImage[i].rows*0.5;
		}
		else imageLen[i] = rawImage[i].cols * rawImage[i].rows;
	}
	// int imageLen = rawImage[0].cols*0.5 * rawImage[0].rows*0.5;
	// Size writefilesize(rawImage[0].cols*0.5, rawImage[0].rows*0.5);

	for (int i = 0; i < camera_num; i++)
	{
		if (rawImage[i].rows > 1000 || rawImage[i].cols > 1000)
		{
			foreground[i].create(Size(rawImage[i].cols / 2, rawImage[i].rows / 2), CV_8UC1);
			xywhdata[i].create(4, 1, CV_32F);
			resize(rawImage[i], resizedImg[i], Size(rawImage[i].cols / 2, rawImage[i].rows / 2));
		}
		else
		{
			foreground[i].create(Size(rawImage[i].cols, rawImage[i].rows), CV_8UC1);
			xywhdata[i].create(4, 1, CV_32F);
			resize(rawImage[i], resizedImg[i], Size(rawImage[i].cols, rawImage[i].rows));
		}
	}
	//capture[0].open(*(videos));
	//capture[1].open(*(videos + 1));

	#ifdef ViBe
		for (int i = 0; i < camera_num; i++)
		{
			Vibe_Bgs[i].init(resizedImg[i]);
			Vibe_Bgs[i].processFirstFrame(resizedImg[i]);
			cout << " Training complete!" << endl;
		}
	#endif

	#ifdef CBM
		image_codebook = new codeBook*[camera_num];
		pColor = new unsigned char*[camera_num];

		for (int i = 0; i < camera_num; i++)
		{
			image_codebook[i] = new codeBook[imageLen[i]];
			pColor[i] = new unsigned char[1];
		}

		for (int i = 0; i < camera_num; i++)	// 初始化每個codeword數目為0
			for (int j = 0; j < imageLen[i]; j++)
			{
				image_codebook[i][j].numEntries = 0;
			}
		
		for (int i = 0; i < nChannels; i++)
		{
			codeBook_Bounds[i] = 10;		// 用於確定codeword各通道的門檻值
			minMod[i] = 20;						// 用於背景差分函數中
			maxMod[i] = 20;						// 調整其值以達到最好的分割
		}
	#endif

// **********************************************************************************
	for (int i = 0, processing_cam = 0;; i++, processing_cam++) {
		processing_cam = processing_cam % camera_num;
		capture[processing_cam] >> rawImage[processing_cam];
		if (rawImage[processing_cam].rows > 1000 || rawImage[processing_cam].cols > 1000)
		{
			resize(rawImage[processing_cam], resizedImg[processing_cam], Size(rawImage[processing_cam].cols / 2, rawImage[processing_cam].rows / 2));
		}
		else resize(rawImage[processing_cam], resizedImg[processing_cam], Size(rawImage[processing_cam].cols, rawImage[processing_cam].rows));
		
		// resize(rawImage[processing_cam], resizedImg[processing_cam], writefilesize);
		if (rawImage[processing_cam].empty())
			break;
		
		// *** 選擇用 ViBe 方法做背景想減 *************************************************************************
		#ifdef ViBe
			Vibe_Bgs[processing_cam].testAndUpdate(resizedImg[processing_cam], &foreground_nums);
			foreground[processing_cam] = Vibe_Bgs[processing_cam].getMask();
			morphologyEx(foreground[processing_cam], foreground[processing_cam], MORPH_OPEN, Mat());
		#endif
	
		// *** 選擇用 Codebook Mode 方法做背景想減 *************************************************************
		#ifdef CBM
			cvtColor(resizedImg[processing_cam], yuvImage[processing_cam], CV_BGR2YCrCb);
			if (i <= 15)
			{ // 15個frame内進行背景學習
				pColor[processing_cam] = (unsigned char*)(yuvImage[processing_cam].data);  // 指向yuvImage影像的通道數據
				//for (int cc = 0; cc < 2; cc++) {
				for (int c = 0; c < imageLen[processing_cam]; c++)
					{
						updateCodeBook(pColor[processing_cam], image_codebook[processing_cam][c], codeBook_Bounds, nChannels);
						// 對每個像素,調用此函數,捕捉背景中相關變化圖像
						pColor[processing_cam] += 3;
						// 3 通道圖像, 指向下一個pixel通道數據
					}
				//}
				if (i == 15)
				{ // 到15 frame時調用下面函數,刪除codebook中舊的codeword
					for (int c = 0; c < imageLen[processing_cam]; c++)
						clearStaleEntries(image_codebook[processing_cam][c]);
				}
			}
			else
			{
				unsigned char maskPixelCodeBook;
				pColor[processing_cam] = (unsigned char *)(yuvImage[processing_cam].data); //3 channel yuv image
				unsigned char *pMask = (unsigned char *)(foreground[processing_cam].data); //1 channel image
				// 指向ImaskCodeBook 通道數據序列的首元素
				for (int c = 0; c < imageLen[processing_cam]; c++)
				{
					maskPixelCodeBook = backgroundDiff(pColor[processing_cam], image_codebook[processing_cam][c], nChannels, minMod, maxMod, &foreground_nums);
					*pMask++ = maskPixelCodeBook;
					pColor[processing_cam] += 3;
				// pColor 指向的是3通道影像
				}
				cout << "backgroundDiff" << endl;
			}
		#endif

		#ifdef CBM
			if (i > 15){
		#endif
		DeNoise(foreground[processing_cam], DeNoised_foreground[processing_cam]);
		findContours(foreground[processing_cam], contours[processing_cam], CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		for (size_t i = 0; i < contours[processing_cam].size(); i++){
			// Rect bBox;
			bBox[processing_cam] = boundingRect(contours[processing_cam][i]);

			// Searching for a bBox almost square
			if (bBox[processing_cam].area() >= 180) {	// 長方形區域面積超過 180，則畫在影像上

				//getOrientation(contours[i], tmpraw);	//畫框
				xywhdata[processing_cam].at<float>(0) = bBox[processing_cam].x + bBox[processing_cam].width / 2;
				xywhdata[processing_cam].at<float>(1) = bBox[processing_cam].y + bBox[processing_cam].height / 2;
				xywhdata[processing_cam].at<float>(2) = bBox[processing_cam].width;
				xywhdata[processing_cam].at<float>(3) = bBox[processing_cam].height;

				cout << "cam" << processing_cam << " ( " << xywhdata[processing_cam].at<float>(0) << "," << xywhdata[processing_cam].at<float>(1) << " )\n";
				// system("pause");
				rectangle(resizedImg[processing_cam], bBox[processing_cam], Scalar(255, 255, 255), 2);
			}
			if (foreground_nums > (resizedImg[processing_cam].cols * resizedImg[processing_cam].rows * 0.2)) {

				put_Font(resizedImg[processing_cam]);
				cout << "warning" << endl;
			}
			cout << setprecision(3) << "foreground : " << (float)foreground_nums / (float)imageLen[processing_cam] * 100 << " %" << endl;
		}

		sprintf(WinName, "Camera%d", processing_cam);
		// sprintf(WinName2, "DeNoised_foreground %d", processing_cam);
		imshow(WinName, resizedImg[processing_cam]);
		// imshow(WinName2, DeNoised_foreground[processing_cam]);

		if (cvWaitKey(10) == 'q')
			break;

	#ifdef CBM
		}
	#endif
	}
	return 0;
}
