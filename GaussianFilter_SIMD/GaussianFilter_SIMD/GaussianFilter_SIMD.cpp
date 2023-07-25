#include <iostream>
#include<opencp.hpp>
#pragma comment(lib,"opencp.lib")
#include<immintrin.h>
#include<fstream>
using namespace cv;
using namespace cp;
using namespace std;


// カーネルサイズ 3 * 3
void GaussianFilter_SIMD(Mat& src, Mat& dst)
{
	if (src.data != dst.data) src.copyTo(dst);

	// matX 重み1/xの画像作成
	Mat mat16, mat8, mat4;
	if (src.data != mat16.data) mat16.create(src.size(), src.type());
	if (src.data != mat8.data) mat8.create(src.size(), src.type());
	if (src.data != mat4.data) mat4.create(src.size(), src.type());

	const int width = src.cols;
	const int height = src.rows;

	const float div_16 = 16.0f;
	const float div_8 = 8.0f;
	const float div_4 = 4.0f;
	__m256 div16 = _mm256_broadcast_ss(&div_16);
	__m256 div8 = _mm256_broadcast_ss(&div_8);
	__m256 div4= _mm256_broadcast_ss(&div_4);

	//LUT作成
	__m256 Apr, w16, w8, w4;
	for (int y = 0; y < height; y++)
	{
		__m256* mat16_ptr = (__m256*)mat16.ptr<float>(y, 0);
		__m256* mat8_ptr = (__m256*)mat8.ptr<float>(y, 0);
		__m256* mat4_ptr = (__m256*)mat4.ptr<float>(y, 0);
		float* srcptr = src.ptr<float>(y);

		for (int x = 8; x < width - 8; x += 8)
		{
			Apr = _mm256_load_ps(srcptr);
			w16 = _mm256_div_ps(Apr, div16);
			w8 = _mm256_div_ps(Apr, div8);
			w4 = _mm256_div_ps(Apr, div4);

			*mat16_ptr++ = w16;
			*mat8_ptr++ = w8;
			*mat4_ptr++ = w4;
		}
	}

	// フィルタ計算
	for (int y = 1; y < height - 1; y++)
	{
		float* dstptr = dst.ptr<float>(y);

		__m256* A01ptr = (__m256*)mat16.ptr<float>(y - 1, 7);
		__m256* A02ptr = (__m256*)mat8.ptr<float>(y - 1, 8);
		__m256* A03ptr = (__m256*)mat16.ptr<float>(y - 1, 9);
		__m256* A11ptr = (__m256*)mat8.ptr<float>(y + 0, 7);
		__m256* A12ptr = (__m256*)mat4.ptr<float>(y + 0, 8);
		__m256* A13ptr = (__m256*)mat8.ptr<float>(y + 0, 9);
		__m256* A21ptr = (__m256*)mat16.ptr<float>(y + 1, 7);
		__m256* A22ptr = (__m256*)mat8.ptr<float>(y + 1, 8);
		__m256* A23ptr = (__m256*)mat4.ptr<float>(y + 1, 9);


		for (int x = 8; x < width - 8; x += 8)
		{
			Apr = _mm256_add_ps(*A01ptr++, _mm256_add_ps(*A02ptr++, _mm256_add_ps(*A03ptr++, _mm256_add_ps(*A11ptr++, _mm256_add_ps(*A12ptr++, 
				_mm256_add_ps(*A13ptr++, _mm256_add_ps(*A21ptr++, _mm256_add_ps(*A22ptr++, *A23ptr++))))))));


			_mm256_store_ps(dstptr++, Apr);

		}
	}
}

// カーネルサイズ 3 * 3
void GaussianFilter_naive(Mat& src, Mat& dst)
{
	if (src.data != dst.data) src.copyTo(dst);

	const int width = src.cols;
	const int height = src.rows;

	// matX 重み1/xの画像作成
	Mat mat16, mat8, mat4;
	if (src.data != mat16.data) mat16.create(src.size(), src.type());
	if (src.data != mat8.data) mat8.create(src.size(), src.type());
	if (src.data != mat4.data) mat4.create(src.size(), src.type());

	// LUT作成
	for (int c = 0; c < 3; c++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				mat16.at<Vec3b>(y, x)[c] = src.at<Vec3b>(y, x)[c] / 16;
				mat8.at<Vec3b>(y, x)[c] = src.at<Vec3b>(y, x)[c] / 8;
				mat4.at<Vec3b>(y, x)[c] = src.at<Vec3b>(y, x)[c] / 4;
			}
		}
	}
	
	// デバック用出力
	/*imshow("mat16", mat16);
	imshow("mat8", mat8);
	imshow("mat4", mat4);*/

	// フィルタ計算
	for (int c = 0; c < 3; c++)
	{
		for (int y = 1; y < height - 1; y++)
		{
			for (int x = 1; x < width - 1; x++)
			{
				dst.at<Vec3b>(y, x)[c] = mat16.at<Vec3b>(y - 1, x - 1)[c] + mat8.at<Vec3b>(y - 1, x)[c] + mat16.at<Vec3b>(y - 1, x + 1)[c]
					+ mat8.at<Vec3b>(y, x - 1)[c] + mat4.at<Vec3b>(y, x)[c] + mat8.at<Vec3b>(y, x + 1)[c]
					+ mat16.at<Vec3b>(y + 1, x - 1)[c] + mat8.at<Vec3b>(y + 1, x)[c] + mat16.at<Vec3b>(y + 1, x + 1)[c];
			}
		}
	}
}


int main()
{
	Mat src_lenna = cv::imread("lenna.png", 1);
	Mat src;
	Mat dst_naive;
	Mat dst_SIMD;
	resize(src_lenna, src, Size(512, 512));

	if (src.data != dst_SIMD.data) dst_SIMD.create(src.size(), src.type());
	addNoise(src, src, 50);

	int key = 0;

	Timer t_naive("a", cp::TIME_MSEC, false);
	Timer t_simd("a", cp::TIME_MSEC, false);
	ConsoleImage ci;

	while (key != 'q')
	{
		ci("width: %d height: %d", src.cols, src.rows);
		imshow("src", src);

		{
			t_naive.start();
			GaussianFilter_naive(src, dst_naive);
			t_naive.getpushLapTime();
		}
		ci("naive time %f ms, %f", t_naive.getLapTimeMedian());
		imshow("dst_naive", dst_naive);

		{
			t_simd.start();
			GaussianFilter_SIMD(src, dst_SIMD);
			t_simd.getpushLapTime();
		}
		ci("SIMD time %f ms, %f", t_simd.getLapTimeMedian());
		imshow("dst_SIMD", dst_SIMD);

		ci.show();
		key = waitKey(1);
	}
}
