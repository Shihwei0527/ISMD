#include <iostream>
#include <cv.h>			

using namespace cv;

#define CHANNELS 3      // 設定處理的影像通道數,要求小於等於影像本身的通道數   
 
/*  --------------- 下面為 codebook element 的資料結構 -------------  */
/*  --------------- 處理圖像時每個像素對應一個codebook,每個codebook中可有若干個codeword -------------  */

typedef struct code_element {

	unsigned char learnHigh[CHANNELS];    // High side threshold for learning  此codeword各通道的門檻值上限(學習界限)
	unsigned char learnLow[CHANNELS];     // Low side threshold for learning   此codeword各通道的門檻值下限
										  // 學習過程中如果一個新像素各通道值x[i],均有 learnLow[i]<=x[i]<=learnHigh[i],則該像素可合併於此codeword
	unsigned char max[CHANNELS];          // High side of box boundary  屬於此codeword的像素中各通道的最大值
	unsigned char min[CHANNELS];          // Low side of box boundary   屬於此codeword的像素中各通道的最小值
	int t_last_update;					  // This is book keeping to allow us to kill stale entries
										  // 此codeword最後一次更新的時間,每一frame為一個單位時間,用於計算stale
	int	stale;							  // max negative run (biggest period of inactivity)
										  // 此codeword最長不更新時間,用於刪除規定時間不更新的codeword,精簡codebook
} codeElement;	// codeword的資料結構

typedef struct code_book {
	
	codeElement **cb;		// codeword的二維指標,理解為指向codeword指標數组的指標,使得添加codeword時不需要來回複製codeword,只需要簡單的指標賦值即可
	int			numEntries; // 此codebook中codeword的數目
	int			t=0;			// count every access  此codebook現在的時間,一個frame為一個時間單位

} codeBook;	// codebook的資料結構

/*  --------------- 下面為 codebook element 的 functions -------------  */

int updateCodeBook (unsigned char *p, codeBook &c, unsigned *cbBounds, int numChannels);
		
	/*  --------------- updateCodeBook 的 功能與參數說明 -------------  */

	/**  Updates the codebook entry with a new data point  
	 	 p            Pointer to a YUV pixel   指向某個frame yuvImage 圖像的通道數據
		 c            Codebook for this pixel  某個pixel中的codebook
		 cbBounds     Learning bounds for codebook (Rule of thumb(經驗法則 ): 10) 
	     numChannels  Number of color channels we're learning

		 NOTES:		  cvBounds must be of size cvBounds[numChannels]
	     RETURN:	  codebook index  
	*/


unsigned char backgroundDiff(unsigned char *p, codeBook &c, int numChannels, int *minMod, int *maxMod, int* foreground_nums);
	
	/*  --------------- cvbackgroundDiff 的 功能與參數說明 -------------  */
	
	/**  Given a pixel and a code book, determine if the pixel is covered by the codebook  
		 p				pixel pointer (YUV interleaved)  
		 c				codebook reference  
		 numChannels	Number of channels we are testing  
		 maxMod			Add this (possibly negative) number onto max level when code_element determining if new pixel is foreground  
		 minMod			Subract this (possible negative) number from min level code_element when determining if pixel is foreground  
		  
		 NOTES:			minMod and maxMod must have length numChannels, e.g. 3 channels => minMod[3], maxMod[3].  
		 Return:		0 => background, 255 => foreground  
	*/


int clearStaleEntries (codeBook &c);

	/*  --------------- cvclearStaleEntries 的 功能與參數說明 -------------  */
	
	/** After you've learned for some period of time, periodically call this to clear out stale codebook entries
		c      Codebook to clean up  
	  
		Return: number of entries cleared  
	*/

void DeNoise (Mat&input, Mat&output);