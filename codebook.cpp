﻿// *****************************************************************************************************
// *** CodeBook Mode 背景相減法 *****************************************************************
// *****************************************************************************************************
#include "codebook.h"
#include <iostream>

using namespace std;

void updateCodeBook(unsigned char *p, codeBook &c, unsigned *cbBounds, int numChannels, int ccc) {

	if (c.numEntries == 0)
		c.t = 0;		// codebook 中 codeword 為零時，初始化時間為 0，c.numEntries:codeword 的數量
	c.t += 1;		// Record learning event  每調用一次加一，即每一影像加一

	// **********************************************************************************************************
	//*** set high and low bounds (設定像素門檻值的上下限) ----------------------------------------------------
	int n;
	unsigned int high[3], low[3];

	for (n = 0; n < numChannels; n++)	{
		high[n] = *(p + n) + *(cbBounds + n);	// *(p+n) 和 p[n] 結果等價，經試驗*(p+n) 速度更快
		if (high[n] > 255)
			high[n] = 255;
		low[n] = *(p + n) - *(cbBounds + n);
		if (low[n] < 0)
			low[n] = 0;
		//*** 用 p 所指像素通道數，加減 cbBonds 中數值，作為此像素的門檻值上下限
	}

	// *************************************************************************************************************
	//*** SEE IF THIS FITS AN EXISTING CODEWORD
	int matchChannel;	//*** 用來計算符合門檻值得chanel數目，假使所有channel都符合，就更新此codeword	
	int i;
	for (i = 0; i<c.numEntries; i++) {

		// 遍歷此codebook每個codeword,測試p像素是否滿足其中之一
		matchChannel = 0;
		for (n = 0; n<numChannels; n++) {			//遍歷某個codeword的每個通道

			if ((c.cb[i]->learnLow[n] <= *(p + n)) && (*(p + n) <= c.cb[i]->learnHigh[n])) { //Found an entry for this channel

				// 如果p 像素通道數值在該codeword門檻值上下限之間，c.cb[i]代表第i個codeword
				matchChannel++;
			}
		}
		if (matchChannel == numChannels)		// If an entry was found over all channels
			// 如果p 像素的每個通道和某個codeword的每個通道相差不多(小於門檻值)
		{
			c.cb[i]->t_last_update = c.t;
			// 更新該codeword時間為當前時間
			// adjust this codeword for the first channel
			for (n = 0; n<numChannels; n++)
				//調整該codeword各通道最大最小值
			{
				if (c.cb[i]->max[n] < *(p + n))
					c.cb[i]->max[n] = *(p + n);
				else if (c.cb[i]->min[n] > *(p + n))
					c.cb[i]->min[n] = *(p + n);
			}
			break;
		}	// end if 找到有符合的codeword
	}		// end for (i=0; i<c.numEntries; i++)跑遍每個codeword找有與此pixel相似的嗎

	//********************************************************************************************************
	// ENTER A NEW CODE WORD IF NEEDED
	if (i == c.numEntries)  // No existing code word found, make a new one
		// p 像素不滿足此codebook中任何一個codeword,下面創建一個新codeword
	{
		code_element **foo = new code_element*[c.numEntries + 1];
		// 申請c.numEntries+1 個指向codeword的指標
		for (int ii = 0; ii < c.numEntries; ii++)
			// 將前c.numEntries 個指標指向已存在的每個codeword
			foo[ii] = c.cb[ii];

		foo[c.numEntries] = new code_element;
		// 申請一個新的codeword
		if (c.numEntries) delete[] c.cb;
		// 如codeword數目不為0，刪除c.cb 指標數组
		c.cb = foo;
		// 把foo 頭指標賦給c.cb
		for (n = 0; n < numChannels; n++)
			// 更新新codeword各通道數據
		{
			c.cb[c.numEntries]->learnHigh[n] = high[n];
			c.cb[c.numEntries]->learnLow[n] = low[n];
			c.cb[c.numEntries]->max[n] = *(p + n);
			c.cb[c.numEntries]->min[n] = *(p + n);
		}
		c.cb[c.numEntries]->t_last_update = c.t;
		c.cb[c.numEntries]->stale = 0;
		c.numEntries += 1;
	}	//end if 沒有任何codeword與此pixel相似，新增codeword

	// ********************************************************************************************************
	// OVERHEAD TO TRACK POTENTIAL STALE ENTRIES
	for (int s = 0; s<c.numEntries; s++)
	{
		// This garbage is to track which codebook entries are going stale
		int negRun = c.t - c.cb[s]->t_last_update;//現在時間減去此pixel最後更新時間
		// 計算該codeword的不更新時間
		if (c.cb[s]->stale < negRun)
			c.cb[s]->stale = negRun;
	}

	// SLOWLY ADJUST LEARNING BOUNDS
	for (n = 0; n<numChannels; n++)
		// 如果像素通道數據在高低門檻值範圍内,但在codeword門檻值之外,則緩慢調整此codeword學習界限
	{
		if (c.cb[i]->learnHigh[n] < high[n])
			c.cb[i]->learnHigh[n] += 1;
		if (c.cb[i]->learnLow[n] > low[n])
			c.cb[i]->learnLow[n] -= 1;
	}
	// return(i);
}

unsigned char backgroundDiff(unsigned char *p, codeBook &c, int numChannels, int *minMod, int *maxMod, int* foreground_nums){

	*foreground_nums = 0;
	// 下面步驟和背景學習中查找codeword如出一轍
	int matchChannel;
	//SEE IF THIS FITS AN EXISTING CODEWORD
	int i;
	for (i = 0; i<c.numEntries; i++)
	{
		matchChannel = 0;
		for (int n = 0; n<numChannels; n++)
		{
			if ((c.cb[i]->min[n] - minMod[n] <= *(p + n)) && (*(p + n) <= c.cb[i]->max[n] + maxMod[n]))
				matchChannel++; //Found an entry for this channel
			else
			{
				if (n == 0)
				{
					if (c.cb[i]->min[n]  > *(p + n) && *(p + n)>c.cb[i]->min[n] * 0.6)
						matchChannel++;
					else
						break;
				}
				else
					break;
			}
		}
		if (matchChannel == numChannels)
			break; //Found an entry that matched all channels
	}
	if (i == c.numEntries) {
		// p像素各通道值滿足codebook中其中一個codeword,則返回白色
		return(255);
		*foreground_nums = *foreground_nums + 1;
		cout << "*foreground_nums = " << *foreground_nums << endl;
		system("pause");
	}
	return(0);
}

int clearStaleEntries(codeBook &c){

	int staleThresh = c.t >> 1;			// 設定刷新時間
	int *keep = new int[c.numEntries];	// 申請一個標記數组
	int keepCnt = 0;					// 記錄不刪除codeword數目
	//SEE WHICH CODEBOOK ENTRIES ARE TOO STALE
	for (int i = 0; i<c.numEntries; i++)
		// 遍歷codebook中每個codeword
	{
		if (c.cb[i]->stale> staleThresh)
			// 如codeword中的不更新時間大於設定的刷新時間,則標記為刪除
			keep[i] = 0; //Mark for destruction
		else
		{
			keep[i] = 1; //Mark to keep
			keepCnt += 1;
		}
	}

	// KEEP ONLY THE GOOD
	c.t = 0;						//Full reset on stale tracking
	// codebook時間清零
	code_element **foo = new code_element*[keepCnt];
	// 申請大小為keepCnt 的codeword指標數组
	int k = 0;
	for (int ii = 0; ii<c.numEntries; ii++)
	{
		if (keep[ii])
		{
			foo[k] = c.cb[ii];
			foo[k]->stale = 0;		//We have to refresh these entries for next clearStale
			foo[k]->t_last_update = 0;
			k++;
		}
	}
	//CLEAN UP
	delete[] keep;
	delete[] c.cb;
	c.cb = foo;
	// 把foo 頭指標地址賦給c.cb 
	int numCleared = c.numEntries - keepCnt;
	// 被清理的codeword個數
	c.numEntries = keepCnt;
	// 剩於的codeword地址
	return(numCleared);
}
