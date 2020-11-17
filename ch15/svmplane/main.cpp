#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;
// SVM 알고리즘을 이용한 2차원 점 분류
int main(void)
{	
	// 8개의 점 좌표를 포함하는 train 행렬을 생성
	Mat train = (Mat_<float>( 8, 2 ) <<
		150, 200, 200, 250, 100, 250, 150, 300,
		350, 100, 400, 200, 400, 300, 350, 400 );

	// 훈련 데이터 점들을 정의한 label 생성
	Mat label = (Mat_<int>( 8, 1 ) <<  0, 0, 0, 0, 1, 1, 1, 1 );

	Ptr<SVM> svm = SVM::create();	// SVM 객체를 생성하여 svm에 저장
	svm->setType(SVM::C_SVC);	// SVM 타입을 C_SVC로 설정
	svm->setKernel(SVM::RBF);	// SVM 커널 타입을 RBF(방사 기저 함수 커널)로 설정
	svm->trainAuto(train, ROW_SAMPLE, label);	// 가장 성능이 좋은 parameters를 찾도록 trainAuto 함수를 제공

	Mat img = Mat::zeros(Size(500, 500), CV_8UC3);	// (500x500)크기의 3채널 영상 생성

	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			// img의 모든 픽셀 좌표에 대해 SVM 응답을 구한다.
			Mat test = (Mat_<float>(1, 2) << (float)i, (float)j );
			int res = cvRound(svm->predict(test));	// test의 결과를 res 변수에 저장

			if (res == 0)	// 만약 label이 0이면 Red color circle로 표시
				img.at<Vec3b>(j, i) = Vec3b(128, 128, 255); // R
			else		// 만약 label이 1이면 Green color circle로 표시
				img.at<Vec3b>(j, i) = Vec3b(128, 255, 128); // G
		}
	}

	// train 행렬에 저장된 훈련 데이터 점을 반지름 5인 원으로 표시
	for (int i = 0; i < train.rows; i++) {
		int x = cvRound(train.at<float>(i, 0));	// train 데이터의 x value를 얻는다.
		int y = cvRound(train.at<float>(i, 1));	// train 데이터의 ㅛ value를 얻는다.
		int l = label.at<int>(i, 0);	// train 데이터의 label value를 얻는다.

		if (l == 0)	// 만약 label이 0일 경우
			circle(img, Point(x, y), 5, Scalar(0, 0, 128), -1, LINE_AA); // R
			// 입출력 이미지, 중심점, 반지름, Red , 내부 채움, 안티에일리어싱
		else	// 만약 label이 1일 경우
			circle(img, Point(x, y), 5, Scalar(0, 128, 0), -1, LINE_AA); // G
			// 입출력 이미지, 중심점, 반지름, Green , 내부 채움, 안티에일리어싱
	}

	imshow("svm", img);	// 영상을 출력

	waitKey();	// 키보드가 눌러질 때까지 기다리는 함수
	return 0;
}
