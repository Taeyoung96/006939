#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// SVM 알고리즘을 이용한 필기체 숫자 학습
// 훈련 데이터 : opencv에서 제공하는 digit.png
// Test 데이터 : 직접 GUI에 쓴 글씨체

// 함수 사용을 위해 미리 선언
Ptr<SVM> train_hog_svm(const HOGDescriptor& hog);
void on_mouse(int event, int x, int y, int flags, void* userdata);

int main()
{
#if _DEBUG
	cout << "svmdigit.exe should be built as Relase mode!" << endl;
	return 0;
#endif

	HOGDescriptor hog(Size(20, 20), Size(10, 10), Size(5, 5), Size(5, 5), 9);	// HOGDescriptor 객체 생성
	// (20x20) 영상에서 (10x10) 크기의 블록, (5x5) 크기의 셀을 사용하여 9개의 그래디언트 방향 히스토그램을 구하도록 설정

	Ptr<SVM> svm = train_hog_svm(hog);	// train_hog_svm() 함수를 이용하여 svm을 학습시킨다.

	// 학습이 잘 되었는지 확인
	if (svm.empty()) {
		cerr << "Training failed!" << endl;
		return -1;
	}

	Mat img = Mat::zeros(400, 400, CV_8U);	// (400 x 400) 크기의 영상 생성

	imshow("img", img);		// img 영상을 출력
	setMouseCallback("img", on_mouse, (void*)&img);	// img에 마우스 콜백 함수 등록

	while (true) {
		int c = waitKey();	// 키 입력을 저장

		if (c == 27) {	// esc 버튼을 누르면 종료
			break;
		} else if (c == ' ') {	// space 버튼을 누르면 아래의 코드 실행
			Mat img_resize;
			resize(img, img_resize, Size(20, 20), 0, 0, INTER_AREA);	// 영상을 (20x20) size 변환하여 img_resize에 저장

			vector<float> desc;	// Descriptor vector 생성
			hog.compute(img_resize, desc);	// HOGDescriptor 객체를 생성한 후, HOG 기술자를 계산

			Mat desc_mat(desc);	// Descriptor를 Mat으로 변환
			int res = cvRound(svm->predict(desc_mat.t()));	// Descriptor의 크기를 (1x324)로 변환하여 SVM 결과를 예측
			cout << res << endl;

			img.setTo(0);	// img 영상의 원소를 0(검은색)으로 설정
			imshow("img", img);		// img 영상을 출력
		}
	}

	return 0;
}

// SVM 알고리즘 학습을 위해 사용하는 함수
Ptr<SVM> train_hog_svm(const HOGDescriptor& hog)
{
	// digits.png를 GrayScale 이미지로 Mat 형태로 저장
	Mat digits = imread("digits.png", IMREAD_GRAYSCALE);

	// 이미지를 잘 불러왔는지 확인
	if (digits.empty()) {
		cerr << "Image load failed!" << endl;
		//return 0; //error: could not convert ‘-1’ from ‘int’ to ‘cv::Ptr<cv::ml::KNearest>’ 발생
	}

	Mat train_hog, train_labels;

	// 가로 100개, 세로 50개 (20x20) 이미지를 하나씩 불러온다.
	for (int j = 0; j < 50; j++) {
		for (int i = 0; i < 100; i++) {
			Mat roi = digits(Rect(i * 20, j * 20, 20, 20));	// 필기체 숫자 영상 저장

			vector<float> desc;	
			hog.compute(roi, desc);	// descriptor 계산

			Mat desc_mat(desc);	// descriptor를 Mat으로 변환
			train_hog.push_back(desc_mat.t());	// Descriptor의 크기를 (1x324)로 변환하여 train_hog 행렬에 추가
			train_labels.push_back(j / 5);	// 현재 추가한 필기체 숫자 영상의 정답 레이블을 train_labels 행렬에 추가
		}
	}

	Ptr<SVM> svm = SVM::create();	// SVM 객체 생성
	svm->setType(SVM::Types::C_SVC);	// SVM의 type을 C_SVC로 설정
	svm->setKernel(SVM::KernelTypes::RBF);	// SVM의 커널 타입을 RBF로 설정
	svm->setC(2.5);	// Parameter C 설정
	svm->setGamma(0.50625);	// Parameter Gamma 설정
	svm->train(train_hog, ROW_SAMPLE, train_labels);	// SVM 학습을 진행

	return svm;
}

Point ptPrev(-1, -1);

// 마우스로 숫자 그리기
void on_mouse(int event, int x, int y, int flags, void* userdata)
{
	Mat img = *(Mat*)userdata;

	if (event == EVENT_LBUTTONDOWN)
		ptPrev = Point(x, y);	// 마우스 왼쪽 버튼을 누르면 누른 위치를 ptPrev에 저장
	else if (event == EVENT_LBUTTONUP)
		ptPrev = Point(-1, -1);	// 마우스 왼쪽 버튼을 떼면 ptPrev 좌표를 (-1,-1)로 초기화
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		// 마우스 왼쪽 버튼을 누른 상태로 마우스를 움직이면 ptPrev 좌표부터 (x,y) 좌표까지 직선을 그린다.
		line(img, ptPrev, Point(x, y), Scalar::all(255), 40, LINE_AA, 0);
		// 입출력 이미지, 시작점 , 끝점, 흰색 , Thickness = 40 , 안티에일리어싱
		ptPrev = Point(x, y);

		imshow("img", img);		// 영상을 출력
	}
}