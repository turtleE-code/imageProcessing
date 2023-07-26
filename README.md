# imageProcessing
C++で実装しています <br>
一般的によく用いられている画像処理ライブラリであるOpenCVと、研究室で主に使われているライブラリのOpenCPを用いた実装です <br>
OpenCV(https://opencv.org/) <br>
OpenCP(https://github.com/norishigefukushima/OpenCP)

## GaussianFilter
- 画像のエッジ(輪郭)を保持しつつ、ぼかす

上：ノイズを加えた画像 <br>
下：左の画像にガウシアンフィルタをかけてぼかし、ノイズを除去した画像

<img src="GaussianFilter_SIMD/images/color_noise_src.png" width = 45%>
<img src="GaussianFilter_SIMD/images/color_noise_dst.png" width = 45%>

## 実装
1. 処理に用いられる重みを計算する(mat4, mat8, mat16の画像)	<br>
重み <br>
<img src="GaussianFilter_SIMD/images/gauss.png" width = 30%>
(引用：https://imagingsolution.net/imaging/gaussian/) <br>
2. 1で計算した画像を用いてフィルタをガウシアンフィルタを計算する

### GaussianFilter_naive
通常のMat型(画像を格納する配列)を一つずつ計算

### GaussianFilter_SIMD
SIMD演算を用いてMat型を8つずつ計算

#### SIMD演算の例
_mm256_add_ps
<img src="GaussianFilter_SIMD/images/add.png" width = 100%>

## 演算速度結果
naive実装：6.51ms	SIMD実装：0.75ms <br>
およそ8.6倍の高速化 <br>
<img src="GaussianFilter_SIMD/images/result.png" width = 60%>


## 参考
メディアンフィルタ <br>
<img src="GaussianFilter_SIMD/images/median.png" width = 45%>
