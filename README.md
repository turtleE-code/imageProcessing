# imageProcessing

## GaussianFilter
- �摜�̃G�b�W(�֊s)��ێ����A�ڂ���

���F�m�C�Y���������摜 <br>
�E�F���̉摜�ɃK�E�V�A���t�B���^�������Ăڂ����A�m�C�Y�����������摜

<img src="GaussianFilter_SIMD/images/color_noise_src.png" width = 45%>
<img src="GaussianFilter_SIMD/images/color_noise_dst.png" width = 45%>

## ����
1. �����ɗp������d�݂��v�Z����(mat4, mat8, mat16�̉摜)	<br>
�d�� <br>
<img src="GaussianFilter_SIMD/images/gauss.png" width = 30%>
(���p�Fhttps://imagingsolution.net/imaging/gaussian/)
2. 1�Ōv�Z�����摜��p���ăt�B���^���K�E�V�A���t�B���^���v�Z����

### GaussianFilter_naive
�ʏ��Mat�^(�摜���i�[����z��)������v�Z

### GaussianFilter_SIMD
SIMD���Z��p����Mat�^��8���v�Z

#### SIMD���Z�̗�
_mm256_add_ps
<img src="GaussianFilter_SIMD/images/add.png" width = 100%>

## ���Z���x����
naive�����F6.51ms	SIMD�����F0.75ms <br>
���悻8.6�{�̍����� <br>
<img src="GaussianFilter_SIMD/images/result.png" width = 60%>


## �Q�l
���f�B�A���t�B���^ <br>
<img src="GaussianFilter_SIMD/images/median.png" width = 45%>