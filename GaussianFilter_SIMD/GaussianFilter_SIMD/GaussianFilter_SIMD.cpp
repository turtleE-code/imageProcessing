#include <iostream>
#include<opencp.hpp>
#pragma comment(lib,"opencp.lib")
#include<immintrin.h>
#include<fstream>
using namespace cv;
using namespace cp;
using namespace std;

//template<typename T>
// 3 * 3
void GaussianFilter_SIMD(const Mat& src, Mat& dst)
{
	/*Mat m8;
	Mat m4;*/
	//CV_Assert(W.size().area() == width * height * 3);
	const int width = src.cols;
	const int height = src.rows;

	//Mat m16 = Mat::zeros(Size(height, width), CV_8U);
	Mat m16;
	Mat m8 = Mat::zeros(Size(height, width), CV_8U);
	Mat m4 = Mat::zeros(Size(height, width), CV_8U);

	if (src.data != dst.data) src.copyTo(dst);
	//if (src.data != m16.data) src.copyTo(m16);
	//if (src.data != m8.data) src.copyTo(m8);
	//if (src.data != m4.data) src.copyTo(m4);

	__m256 Apr, weight1_16, weight2_16, weight4_16;
	__m256 w16 = _mm256_set1_ps(0.0625f);
	__m256 w8 = _mm256_set1_ps(0.125);
	__m256 w4 = _mm256_set1_ps(0.25);
	__m256 w1 = _mm256_set1_ps(1.0);
	__m256 w0 = _mm256_set1_ps(0.0);
	__m256 w2 = _mm256_set1_ps(2.0f);

	// LUT作成
	for (int y = 0; y < height; y++)
	{
		__m256* srcptr = (__m256*)src.ptr<float>(y + 0, 0);
		__m256* dstptr = (__m256*)dst.ptr<float>(y + 0, 0);
		__m256* p16 = (__m256*)m16.ptr<float>(y + 0, 0);
		__m256* p8 = (__m256*)m8.ptr<float>(y + 0, 0);
		__m256* p4 = (__m256*)m4.ptr<float>(y + 0, 0);

		for (int x = 0; x < width; x += 8)
		{
			*dstptr++ = _mm256_setzero_ps();
			*p16++ = _mm256_mul_ps(*srcptr, w1);
		}
	}
	imshow("m16", m16);

	for (int y = 8; y < height - 8; y++)
	{
		__m256* mD0 = (__m256*)dst.ptr<float>(y + 0, 0);
		for (int x = 8; x < width - 8; x += 8)
		{
			//*mD0++ = _mm256_set1_ps(0.0);
			//*mD0++ = _mm256_mul_ps(*mD0, w16);
		}
	}

}


int main()
{
	Mat src_ = cv::imread("lenna.png");
	Mat src;
	Mat dst_original;
	Mat dst_SIMD;
	resize(src_, src, Size(512, 512));
	//addNoise(src, src, 50);

	int key = 0;

	Timer original_t("a", cp::TIME_MSEC, false);
	Timer simd_t("a", cp::TIME_MSEC, false);
	ConsoleImage ci;

	while (key != 'q')
	{
		ci("width: %d height: %d", src.cols, src.rows);
		imshow("src", src);

		{
			original_t.start();
			GaussianBlur(src, dst_original, Size(5, 5), 5);
			original_t.getpushLapTime();
		}
		ci("original time %f ms, %f", original_t.getLapTimeMedian());
		//imshow("dst", dst_original);

		{
			simd_t.start();
			GaussianFilter_SIMD(src, dst_SIMD);
			simd_t.getpushLapTime();
		}
		ci("SIMD time %f ms, %f", simd_t.getLapTimeMedian());
		imshow("dst_SIMD", dst_SIMD);
		/*original_t.getpushLapTime();
		original_t.getLapTimeMedian();*/
		ci.show();
		//std::cout << "Hello World!\n";
		key = waitKey(1);
	}
    //std::cout << "Hello World!\n";
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
