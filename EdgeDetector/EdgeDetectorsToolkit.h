#pragma once

#include <string>
#include <iostream>
#include <limits>
#include <fstream>
//#include <opencv2\core\core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp>

class EdgeDetectorsToolkit
{
private:
	cv::Mat inputImage;
	cv::Mat outputImage;
	float threshold = 0.5;
	bool secondDerivation = false;

	// support functions for operators
	void prewitt3x3();
	void prewitt5x5();
	void prewitt7x7();

	void sobel3x3();
	void sobel5x5();
	void sobel7x7();

	void kirsch3x3();
	void kirsch5x5();
	void kirsch7x7();

	void robinson3x3();
	void robinson5x5();
	void robinson7x7();

	void laplacian3x3();
	void laplacian5x5();
	void laplacian7x7();
	
	void marrHildreth3x3(double omega = 0.6);
	void marrHildreth5x5(double omega = 1.0);
	void marrHildreth7x7(double omega = 1.4);

	void core8Directions(cv::Mat kernel, cv::Mat kernel45);
	void core2Directions(cv::Mat kernel);
	void coreSecondDerivation(cv::Mat kernel);

	cv::Mat rotateKernel(cv::Mat kernel);
	void putToOutputMax(cv::Mat);


	enum errorCodes {
		EDT_OK = 0,
		EDT_UNKNOWN_KERNEL_SIZE,
		EDT_INCORRECT_THRESHOLD,
		EDT_FILENOTFOUND
	};

public:
	EdgeDetectorsToolkit();
	~EdgeDetectorsToolkit();

	int loadImage(std::string path);
	int saveImage(std::string path);
	
	void setThreshold(float);
	void findEdges();

	void applyPrewitt(int kernelSize);
	void applySobel(int kernelSize);
	void applyKirsch(int kernelSize);
	void applyRobinson(int kernelSize);
	void applyMarrHildreth(int kernelSize, double omega = -1.0);
	void applyLaplacian(int kernelSize);

	void compareWithCannyDetector(std::ofstream&, int);

	// error code
	int errorCode = 0;
};

