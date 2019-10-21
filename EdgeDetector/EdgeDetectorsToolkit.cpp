#include "EdgeDetectorsToolkit.h"


void printMatrix(cv::Mat input) {
	for (int i = 0; i < input.cols; i++) {
		for (int j = 0; j < input.rows; j++) {
			std::cout << input.at<float>(i, j) << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

EdgeDetectorsToolkit::EdgeDetectorsToolkit()
{
}

EdgeDetectorsToolkit::~EdgeDetectorsToolkit()
{
}

// function apply given kernels in 4 direction on input image
void EdgeDetectorsToolkit::core8Directions(cv::Mat kernel, cv::Mat kernel45)
{
	// set the status of EdgeDetector toolkit to second derivation
	this->secondDerivation = false;
	// normalize kernel to values 255
	kernel /= cv::sum(cv::abs(kernel))[0];
	kernel *= 255;
	kernel45 /= cv::sum(cv::abs(kernel45))[0];
	kernel45 *= 255;
	this->outputImage = cv::Mat::zeros(this->inputImage.size(), CV_32FC1);
	// calculate for 8 directions with 2 matrix -> 4 loops 
	for (int i = 0; i < 4; i++) {
		// temp result matrix
		cv::Mat tempResult = cv::Mat::zeros(this->outputImage.size(), CV_32FC1);
		cv::Mat tempResult45 = cv::Mat::zeros(this->outputImage.size(), CV_32FC1);
		// calculate result using 2 matrix
		cv::filter2D(this->inputImage, tempResult, -1, kernel);
		cv::filter2D(this->inputImage, tempResult45, -1, kernel45);
		// normalize values
		//tempResult /= cv::sum(cv::abs(kernel))[0];
		//tempResult45 /= cv::sum(cv::abs(kernel45))[0];	

		// compare with current gradient, keep only max
		this->putToOutputMax(tempResult);
		this->putToOutputMax(tempResult45);

		// rotate matrix
		kernel = this->rotateKernel(kernel);
		kernel45 = this->rotateKernel(kernel45);
	}
}

// function apply given kernel in two direction on input image
void EdgeDetectorsToolkit::core2Directions(cv::Mat kernel)
{
	// set the status of EdgeDetector toolkit to second derivation
	this->secondDerivation = false;

	// normalize kernel to values 255
	kernel /= cv::sum(cv::abs(kernel))[0];
	kernel *= 255;

	this->outputImage = cv::Mat::zeros(this->inputImage.size(), CV_32FC1);
	// temp result matrix
	cv::Mat tempResult = cv::Mat::zeros(this->outputImage.size(), CV_32FC1);
	cv::Mat tempResult90 = cv::Mat::zeros(this->outputImage.size(), CV_32FC1);
	// filter horizontal direction
	cv::filter2D(this->inputImage, tempResult, -1, kernel);
	// normalize value
	//tempResult /= cv::sum(cv::abs(kernel))[0];
	// filter vertical direction
	kernel = this->rotateKernel(kernel);
	cv::filter2D(this->inputImage, tempResult90, -1, kernel);
	// calculate result gradient
	for (int i = 0; i < this->outputImage.rows; i++) {
		for (int j = 0; j < this->outputImage.cols; j++) {
			this->outputImage.at<float>(i, j) = std::sqrt(pow(tempResult.at<float>(i, j), 2) + pow(tempResult90.at<float>(i, j), 2));
			// approximation
			//this->outputImage.at<float>(i, j) = std::abs(tempResult.at<float>(i, j)) + std::abs(tempResult90.at<float>(i, j));
		}
	}
}

// function apply given kernel to input image for methods using second derivative
void EdgeDetectorsToolkit::coreSecondDerivation(cv::Mat kernel)
{
	// normalize kernel to values 255
	kernel /= cv::sum(cv::abs(kernel))[0];
	kernel *= 255;
	// set the status of EdgeDetector toolkit to second derivation
	this->secondDerivation = true;
	this->outputImage = cv::Mat::zeros(this->inputImage.size(), CV_32FC1);
	// temp result matrix
	cv::Mat tempResult = cv::Mat::zeros(this->outputImage.size(), CV_32FC1);
	// filter using given kernel
	cv::filter2D(this->inputImage, tempResult, -1, kernel);

	// assign result to output matrix
	this->outputImage = tempResult;

}

void EdgeDetectorsToolkit::prewitt3x3()
{
	// create prewitt kernel
	float kernelData[9] = {
		1.0,1.0,1.0,
		0.0,0.0,0.0,
		-1.0,-1.0,-1.0
	};
	float kernelData45[9] = {
		0.0,1.0,1.0,
		-1.0,0.0,1.0,
		-1.0,-1.0,-0.0 
	};
	cv::Mat kernel3x3 = cv::Mat(3, 3, CV_32FC1, kernelData);
	cv::Mat kernel3x3_45 = cv::Mat(3, 3, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel3x3, kernel3x3_45);
	
}

void EdgeDetectorsToolkit::prewitt5x5()
{
	// create prewitt kernels
	/*float kernelData[25] = { 
		2.0,2.0,2.0,2.0,2.0,
		1.0,1.0,1.0,1.0,1.0,
		0.0,0.0,0.0,0.0,0.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,
		-2.0,-2.0,-2.0,-2.0,-2.0
	};
	float kernelData45[25] = {
		 0.0, 1.0, 2.0, 2.0, 2.0,
		-1.0, 0.0, 1.0, 1.0, 2.0,
		-2.0,-1.0, 0.0, 1.0, 2.0,
		-2.0,-1.0,-1.0, 0.0, 1.0,
		-2.0,-2.0,-2.0,-1.0, 0.0
	};*/
	float kernelData[25] = {
		1.0,1.0,1.0,1.0,1.0,
		1.0,1.0,1.0,1.0,1.0,
		0.0,0.0,0.0,0.0,0.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0
	};
	float kernelData45[25] = {
		0.0, 1.0, 1.0, 1.0, 1.0,
		-1.0, 0.0, 1.0, 1.0, 1.0,
		-1.0,-1.0, 0.0, 1.0, 1.0,
		-1.0,-1.0,-1.0, 0.0, 1.0,
		-1.0,-1.0,-1.0,-1.0, 0.0
	};
	cv::Mat kernel5x5 = cv::Mat(5, 5, CV_32FC1, kernelData);
	cv::Mat kernel5x5_45 = cv::Mat(5, 5, CV_32FC1, kernelData45);
	
	// calculate gradient
	this->core8Directions(kernel5x5, kernel5x5_45);
}

void EdgeDetectorsToolkit::prewitt7x7()
{
	/*float kernelData[49] = {
		 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
		 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
		 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,
		-4.0,-4.0,-4.0,-4.0,-4.0,-4.0,-4.0
	};
	float kernelData45[49] = {
		 0.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 
		-1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 4.0,
		-2.0,-1.0, 0.0, 1.0, 1.0, 2.0, 4.0,
		-4.0,-2.0,-1.0, 0.0, 1.0, 2.0, 4.0,
		-4.0,-2.0,-1.0,-1.0, 0.0, 1.0, 2.0,
		-4.0,-2.0,-2.0,-2.0,-1.0, 0.0, 1.0,
		-4.0,-4.0,-4.0,-4.0,-2.0,-1.0, 0.0
	};*/
	float kernelData[49] = {
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0
	};
	float kernelData45[49] = {
		0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		-1.0,-1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
		-1.0,-1.0,-1.0, 0.0, 1.0, 1.0, 1.0,
		-1.0,-1.0,-1.0,-1.0, 0.0, 1.0, 1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0, 0.0, 1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0, 0.0
	}; 
	cv::Mat kernel7x7 = cv::Mat(7, 7, CV_32FC1, kernelData);
	cv::Mat kernel7x7_45 = cv::Mat(7, 7, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel7x7, kernel7x7_45);
}

// generate kernel for Sobel with size 3x3
void EdgeDetectorsToolkit::sobel3x3()
{
	// create sobel kernel
	float kernelData[9] = {
		 1.0, 0.0,-1.0,
		 2.0, 0.0,-2.0,
		 1.0, 0.0,-1.0
	};
	float kernelData45[9] = {
		 0.0, 1.0, 2.0,
		-1.0, 0.0, 1.0,
		-2.0,-1.0, 0.0
	};
	cv::Mat kernel3x3 = cv::Mat(3, 3, CV_32FC1, kernelData);
	cv::Mat kernel3x3_45 = cv::Mat(3, 3, CV_32FC1, kernelData45);

	// calculate gradient
	this->core2Directions(kernel3x3);
}
// generate kernel for Sobel with size 5x5
void EdgeDetectorsToolkit::sobel5x5()
{
	// create sobel kernel
	float kernelData[25] = {
		 1.0, 2.0, 0.0,-2.0,-1.0,
		 4.0, 8.0, 0.0,-8.0,-4.0,
		 6.0,12.0, 0.0,-12.0,-6.0,
		 4.0, 8.0, 0.0,-8.0,-4.0,
		 1.0, 2.0, 0.0,-2.0,-1.0
	};
	float kernelData45[25] = {
		0.0,-2.0,-1.0,-4.0,-6.0,
		2.0, 0.0,-8.0,-12.0,-4.0,
		1.0, 8.0, 0.0,-8.0,-1.0,
		4.0,12.0, 8.0, 0.0,-2.0,
		6.0, 4.0, 1.0, 2.0, 0.0
	};
	cv::Mat kernel5x5 = cv::Mat(5, 5, CV_32FC1, kernelData);
	cv::Mat kernel5x5_45 = cv::Mat(5, 5, CV_32FC1, kernelData45);

	// calculate gradient
	this->core2Directions(kernel5x5);
}
// generate kernel for Sobel with size 7x7
void EdgeDetectorsToolkit::sobel7x7()
{
	float kernelData[49] = {
		1.0, 2.0, 3.0, 0.0, -3.0, -2.0, -1.0, 
		3.0, 5.0, 6.0, 0.0, -6.0, -5.0, -3.0,
		5.0, 8.0,12.0, 0.0,-12.0, -8.0, -5.0,
		7.0,10.0,16.0, 0.0,-16.0,-10.0, -7.0,
		5.0, 8.0,12.0, 0.0,-12.0, -8.0, -5.0,
		3.0, 5.0, 6.0, 0.0, -6.0, -5.0, -3.0,
		1.0, 2.0, 3.0, 0.0, -3.0, -2.0, -1.0
	};
	float kernelData45[49] = {
		0.0,-3.0,-2.0, -1.0, -3.0, -5.0,-7.0,
		3.0, 0.0,-6.0, -5.0, -8.0,-10.0,-5.0,
		2.0, 6.0, 0.0,-12.0,-16.0, -8.0,-3.0,
		1.0, 5.0,12.0,  0.0,-12.0, -5.0,-1.0,
		3.0, 8.0,16.0, 12.0,  0.0, -6.0,-2.0,
		5.0,10.0, 8.0,  5.0,  6.0,  0.0,-3.0,
		7.0, 5.0, 3.0,  1.0,  2.0,  3.0, 0.0,
	};
	cv::Mat kernel7x7 = cv::Mat(7, 7, CV_32FC1, kernelData);
	cv::Mat kernel7x7_45 = cv::Mat(7, 7, CV_32FC1, kernelData45);

	// calculate gradient
	this->core2Directions(kernel7x7);
}

void EdgeDetectorsToolkit::kirsch3x3()
{
	// create prewitt kernel
	float kernelData[9] = {
		 5.0, 5.0, 5.0,
		-3.0, 0.0,-3.0,
		-3.0,-3.0,-3.0
	};
	float kernelData45[9] = {
		 5.0, 5.0,-3.0,
		 5.0, 0.0,-3.0,
		-3.0,-3.0,-3.0
	};
	cv::Mat kernel3x3 = cv::Mat(3, 3, CV_32FC1, kernelData);
	cv::Mat kernel3x3_45 = cv::Mat(3, 3, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel3x3, kernel3x3_45);
}

void EdgeDetectorsToolkit::kirsch5x5()
{
	// create prewitt kernel
	float kernelData[25] = {
		 2.0, 2.0, 2.0, 2.0, 2.0,
		 7.0, 9.0, 9.0, 9.0, 7.0,
		-4.0,-5.0, 0.0,-5.0,-4.0,
		-4.0,-5.0,-5.0,-5.0,-4.0,
		-2.0,-2.0,-2.0,-2.0,-2.0
	};
	float kernelData45[25] = {
		-4.0, 7.0, 2.0, 2.0, 2.0,
		-4.0,-5.0, 9.0, 9.0, 2.0, 
		-2.0,-5.0, 0.0, 9.0, 2.0,
		-2.0,-5.0,-5.0,-5.0, 7.0,
		-2.0,-2.0,-2.0,-4.0,-4.0,
	};

	cv::Mat kernel5x5 = cv::Mat(5, 5, CV_32FC1, kernelData);
	cv::Mat kernel5x5_45 = cv::Mat(5, 5, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel5x5, kernel5x5_45);
}

void EdgeDetectorsToolkit::kirsch7x7()
{
	float kernelData[49] = {
		 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
		 4.0, 7.0, 9.0, 9.0, 9.0, 7.0, 4.0,
		-2.0,-4.0,-5.0, 0.0,-5.0,-4.0,-2.0,
		-2.0,-4.0,-5.0,-5.0,-5.0,-4.0,-2.0,
		-1.0,-2.0,-2.0,-2.0,-2.0,-2.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
	};
	float kernelData45[49] = {
		-2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0
		-2.0,-4.0, 7.0, 2.0, 2.0, 2.0, 1.0,
		-1.0,-4.0,-5.0, 9.0, 9.0, 2.0, 1.0,
		-1.0,-2.0,-5.0, 0.0, 9.0, 2.0, 1.0,
		-1.0,-2.0,-5.0,-5.0,-5.0, 7.0, 1.0,
		-1.0,-2.0,-2.0,-2.0,-4.0,-4.0, 4.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,-2.0
	};
	cv::Mat kernel7x7 = cv::Mat(7, 7, CV_32FC1, kernelData);
	cv::Mat kernel7x7_45 = cv::Mat(7, 7, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel7x7, kernel7x7_45);
}

void EdgeDetectorsToolkit::robinson3x3()
{
	// create robinson kernel
	float kernelData[9] = {
		 1.0, 1.0, 1.0,
		 1.0,-2.0, 1.0,
		-1.0,-1.0,-1.0
	};
	float kernelData45[9] = {
		 1.0, 1.0, 1.0,
		-1.0,-2.0, 1.0,
		-1.0,-1.0, 1.0
	};
	cv::Mat kernel3x3 = cv::Mat(3, 3, CV_32FC1, kernelData);
	cv::Mat kernel3x3_45 = cv::Mat(3, 3, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel3x3, kernel3x3_45);
}

void EdgeDetectorsToolkit::robinson5x5()
{
	// create robinson 5x5 kernel
	float kernelData[25] = {
		 1.0, 1.0, 1.0, 1.0, 1.0,
		 1.0, 2.0, 2.0, 2.0, 1.0,
		 1.0, 2.0,-4.0, 2.0, 1.0,
		-1.0,-2.0,-2.0,-2.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0
	};
	float kernelData45[25] = {
		 1.0, 1.0, 1.0, 1.0, 1.0,
		-1.0, 2.0, 2.0, 2.0, 1.0,
		-1.0,-2.0,-4.0, 2.0, 1.0,
		-1.0,-2.0,-2.0, 2.0, 1.0,
		-1.0,-1.0,-1.0,-1.0, 1.0
	};

	cv::Mat kernel5x5 = cv::Mat(5, 5, CV_32FC1, kernelData);
	cv::Mat kernel5x5_45 = cv::Mat(5, 5, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel5x5, kernel5x5_45);
}

void EdgeDetectorsToolkit::robinson7x7()
{
	// create robinson 7x7 kernel
	float kernelData[49] = {
		 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
		 1.0, 2.0, 4.0, 4.0, 4.0, 2.0, 1.0,
		 1.0, 2.0, 4.0,-8.0, 4.0, 2.0, 1.0,
		-1.0,-2.0,-4.0,-4.0,-4.0,-2.0,-1.0,
		-1.0,-2.0,-2.0,-2.0,-2.0,-2.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0
	};
	float kernelData45[49] = {
		 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		-1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
		-1.0,-2.0, 4.0, 4.0, 4.0, 2.0, 1.0,
		-1.0,-2.0,-4.0,-8.0, 4.0, 2.0, 1.0,
		-1.0,-2.0,-4.0,-4.0, 4.0, 2.0, 1.0,
		-1.0,-2.0,-2.0,-2.0,-2.0, 2.0, 1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0, 1.0
	};

	cv::Mat kernel7x7 = cv::Mat(7, 7, CV_32FC1, kernelData);
	cv::Mat kernel7x7_45 = cv::Mat(7, 7, CV_32FC1, kernelData45);

	// calculate gradient
	this->core8Directions(kernel7x7, kernel7x7_45);
}

void EdgeDetectorsToolkit::laplacian3x3()
{
	// create laplacian kernel
	float kernelData[9] = {
		-1.0,-1.0,-1.0,
		-1.0, 8.0,-1.0,
		-1.0,-1.0,-1.0
	};

	// apply kernel to input image
	cv::Mat kernel3x3 = cv::Mat(3, 3, CV_32FC1, kernelData);
	this->coreSecondDerivation(kernel3x3);
}

void EdgeDetectorsToolkit::laplacian5x5()
{
	// create laplacian kernel
	float kernelData[25] = {
		-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,24.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0
	};

	// apply kernel to input image
	cv::Mat kernel5x5 = cv::Mat(5, 5, CV_32FC1, kernelData);
	this->coreSecondDerivation(kernel5x5);
}

void EdgeDetectorsToolkit::laplacian7x7()
{
	// create laplacian kernel
	float kernelData[49] = {
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,48.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
		-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
	};

	// apply kernel to input image
	cv::Mat kernel7x7 = cv::Mat(7, 7, CV_32FC1, kernelData);
	this->coreSecondDerivation(kernel7x7);
}

void EdgeDetectorsToolkit::marrHildreth3x3(double omega)
{
	// generate kernel 
	cv::Mat kernel3x3 = cv::Mat::zeros(3, 3, CV_32FC1);
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			double gauss = -(1 / (CV_PI*pow(omega, 4)))* exp(-( pow(i,2)+pow(j,2) ) / ( 2*pow(omega,2) ) );
			double laplac = 1 - (pow(i, 2) + pow(j, 2)) / (2 * pow(omega, 2));
			kernel3x3.at<float>(i+1, j+1) = gauss*laplac;
		}
	}
	kernel3x3 *= -1;
	// apply kernel to image
	this->coreSecondDerivation(kernel3x3);

	//printMatrix(kernel3x3);
}

void EdgeDetectorsToolkit::marrHildreth5x5(double omega)
{
	//std::cout << omega << std::endl;
	// generate kernel 
	cv::Mat kernel5x5 = cv::Mat::zeros(5, 5, CV_32FC1);
	for (int i = -2; i < 3; i++) {
		for (int j = -2; j < 3; j++) {
			double gauss = -(1 / (CV_PI*pow(omega, 4)))* exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(omega, 2)));
			double laplac = 1 - (pow(i, 2) + pow(j, 2)) / (2 * pow(omega, 2));
			kernel5x5.at<float>(i + 2, j + 2) = gauss*laplac;
		}
	}
	kernel5x5 *= -1;
	// apply kernel to image
	this->coreSecondDerivation(kernel5x5);

	//printMatrix(kernel5x5);
}

void EdgeDetectorsToolkit::marrHildreth7x7(double omega)
{
	// generate kernel 
	cv::Mat kernel7x7 = cv::Mat::zeros(7, 7, CV_32FC1);
	for (int i = -3; i < 4; i++) {
		for (int j = -3; j < 4; j++) {
			double gauss = -(1 / (CV_PI*pow(omega, 4)))* exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(omega, 2)));
			double laplac = 1-(pow(i, 2) + pow(j, 2)) / (2 * pow(omega, 2));
			kernel7x7.at<float>(i + 3, j + 3) = gauss*laplac;
		}
	}
	// fixed approx kernel
	/*float kernelData[49] = {
		1, 3,  4,  4,  4, 3, 1,
		3, 4,  3,  0,  3, 4, 3,
		4, 3, -9,-17, -9, 3, 4,
		4, 0,-17,-30,-17, 0, 4,
		4, 3, -9,-17, -9, 3, 4,
		3, 4,  3,  0,  3, 4, 3,
		1, 3,  4,  4,  4, 3, 1,
	};
	kernel7x7 = cv::Mat(7, 7, CV_32FC1, kernelData);*/
	kernel7x7 *= -1;
	// apply kernel to image
	this->coreSecondDerivation(kernel7x7);
}

// function rotate the kernel by 90, used for 2 and 8 Directions
cv::Mat EdgeDetectorsToolkit::rotateKernel(cv::Mat kernel)
{
	// create output mat
	cv::Mat output = cv::Mat(kernel.size(), CV_32FC1, std::numeric_limits<float>::min());

	// get middle of the kernel
	int middle = kernel.rows / 2;

	cv::Mat rotMat45 = cv::getRotationMatrix2D(cv::Point2i(middle, middle), 90, 1.0);
	cv::warpAffine(kernel, output, rotMat45, kernel.size());

	return output;
}

// support function for core8Directions - keep only max value from current and previous directions
void EdgeDetectorsToolkit::putToOutputMax(cv::Mat gradientResult)
{
	for (int i = 0; i < gradientResult.rows; i++) {
		for (int j = 0; j < gradientResult.cols; j++) {
			this->outputImage.at<float>(i, j) = MAX(this->outputImage.at<float>(i, j), gradientResult.at<float>(i, j));
		}
	}
}

int EdgeDetectorsToolkit::loadImage(std::string path)
{
	// load image as gray scale
	try {
		this->inputImage = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	}
	catch (std::runtime_error& ex) {
		errorCode = EDT_FILENOTFOUND;
		std::cerr << "Exception converting image to PNG format: " << ex.what() << std::endl;
		return 2;
	}
	if (!this->inputImage.data)
	{
		errorCode = EDT_FILENOTFOUND;
		std::cerr << "Could not open or find the original image!" << std::endl;
		return 1;
	}
	// transfer matrix to CV_32FC1
	this->inputImage.convertTo(this->inputImage, CV_32FC1);
	// normalize input image
	cv::normalize(this->inputImage, this->inputImage, 0, 255, CV_MINMAX);
	return 0;
}

int EdgeDetectorsToolkit::saveImage(std::string path)
{
	// normalize image
	cv::normalize(this->outputImage, this->outputImage, 0, 255, CV_MINMAX);
	try {
		//cv::imwrite("inputGrey.jpg", this->inputImage);
		cv::imwrite(path, this->outputImage);
	}
	catch (std::runtime_error& ex) {
		std::cerr << "Exception converting image to PNG format: " << ex.what() << std::endl;
		return 2;
	}
	return 0;
}

void EdgeDetectorsToolkit::setThreshold(float value)
{
	if (value > 1.0 || value < 0.0) {
		std::cerr << "Incorrect value of threshold!" << std::endl;
		this->errorCode = EDT_INCORRECT_THRESHOLD;
	}
	this->threshold = value*255;
}

// function detect edge by given threshold in picture which was convoluted with one of the kernels
void EdgeDetectorsToolkit::findEdges()
{
	if (!this->secondDerivation) {	
		// output matrix to 0-255 range
		cv::normalize(this->outputImage, this->outputImage, 0, 255, CV_MINMAX);
		// keep only values bigger than threshold
		for (int i = 0; i < this->outputImage.rows; i++) {
			for (int j = 0; j < this->outputImage.cols; j++) {
				if (this->outputImage.at<float>(i, j) < this->threshold) {
					this->outputImage.at<float>(i, j) = 0.0;
				}else
					this->outputImage.at<float>(i, j) = 255.0;
			}
		}
	}
	else {
		cv::normalize(this->outputImage, this->outputImage, -255, 255, CV_MINMAX);
		// keep only values bigger than threshold
		int num = 0;
		double min = 0.f, max = 0.f;
		cv::minMaxLoc(this->outputImage, &min, &max);
		for (int i = 0; i < this->outputImage.rows; i++) {
			for (int j = 0; j < this->outputImage.cols; j++) {
				// set boundaries of image to 0
				if (i == 0 || j == 0 || i == this->outputImage.rows - 1 || j == this->outputImage.cols - 1) {
					this->outputImage.at<float>(i, j) = -255.0;
					continue;
				}
				// check for zeroe crossing
				// zero crossing LEFT-UP to RIGHT-DOWN
				if ( (this->outputImage.at<float>(i-1,j-1)>=0 && this->outputImage.at<float>(i + 1, j + 1) < 0) 
					|| (this->outputImage.at<float>(i - 1, j - 1) < 0 && this->outputImage.at<float>(i + 1, j + 1) >= 0) ) {
					if (this->outputImage.at<float>(i,j) > (2*this->threshold-255))
					{
						this->outputImage.at<float>(i, j) = 255.0;
						num++;
					}
					else {
						this->outputImage.at<float>(i, j) = -255.0;
					}
				}
				// zero crossing UP to DOWN
				else if ((this->outputImage.at<float>(i - 1, j) >= 0 && this->outputImage.at<float>(i + 1, j) < 0)
					|| (this->outputImage.at<float>(i - 1, j) < 0 && this->outputImage.at<float>(i + 1, j) >= 0)) {
					if (this->outputImage.at<float>(i, j) > (2 * this->threshold - 255))
					{
						this->outputImage.at<float>(i, j) = 255.0;
						num++;
					}
					else {
						this->outputImage.at<float>(i, j) = -255.0;
					}
				}
				// zero crossing RIGHT-UP to LEFT-DOWN
				else if ((this->outputImage.at<float>(i - 1, j+1) >= 0 && this->outputImage.at<float>(i + 1, j-1) < 0)
					|| (this->outputImage.at<float>(i - 1, j+1) < 0 && this->outputImage.at<float>(i + 1, j-1) >= 0)) {
					if (this->outputImage.at<float>(i, j) >  (2 * this->threshold - 255))
					{
						this->outputImage.at<float>(i, j) = 255.0;
						num++;
					}
					else {
						this->outputImage.at<float>(i, j) = -255.0;
					}
				}
				// zero crossing LEFT to RIGHT
				else if ((this->outputImage.at<float>(i, j + 1) >= 0 && this->outputImage.at<float>(i, j - 1) < 0)
					|| (this->outputImage.at<float>(i, j + 1) < 0 && this->outputImage.at<float>(i, j - 1) >= 0)) {
					if (this->outputImage.at<float>(i, j) >  (2 * this->threshold - 255))
					{
						this->outputImage.at<float>(i, j) = 255.0;
						num++;
					}
					else {
						this->outputImage.at<float>(i, j) = -255.0;
					}
				}
				else {
					this->outputImage.at<float>(i, j) = -255.0;
				}
			}
		}
	}
	
}

void EdgeDetectorsToolkit::applyPrewitt(int kernelSize)
{
	// check the size of kernel
	switch (kernelSize) {
	case 3:
		this->prewitt3x3();
		break;
	case 5:
		this->prewitt5x5();
		break;
	case 7:
		this->prewitt7x7();
		break;
	default:
		std::cerr << "Unknown Kernel size!" << std::endl;
		this->errorCode = EDT_UNKNOWN_KERNEL_SIZE;
		break;
	}
}

void EdgeDetectorsToolkit::applySobel(int kernelSize)
{
	// check the size of kernel
	switch (kernelSize) {
	case 3:
		this->sobel3x3();
		break;
	case 5:
		this->sobel5x5();
		break;
	case 7:
		this->sobel7x7();
		break;
	default:
		std::cerr << "Unknown Kernel size!" << std::endl;
		this->errorCode = EDT_UNKNOWN_KERNEL_SIZE;
		break;
	}
}

void EdgeDetectorsToolkit::applyKirsch(int kernelSize)
{
	// check the size of kernel
	switch (kernelSize) {
	case 3:
		this->kirsch3x3();
		break;
	case 5:
		this->kirsch5x5();
		break;
	case 7:
		this->kirsch7x7();
		break;
	default:
		std::cerr << "Unknown Kernel size!" << std::endl;
		this->errorCode = EDT_UNKNOWN_KERNEL_SIZE;
		break;
	}
}

void EdgeDetectorsToolkit::applyRobinson(int kernelSize)
{
	// check the size of kernel
	switch (kernelSize) {
	case 3:
		this->robinson3x3();
		break;
	case 5:
		this->robinson5x5();
		break;
	case 7:
		this->robinson7x7();
		break;
	default:
		std::cerr << "Unknown Kernel size!" << std::endl;
		this->errorCode = EDT_UNKNOWN_KERNEL_SIZE;
		break;
	}
}

void EdgeDetectorsToolkit::applyMarrHildreth(int kernelSize, double omega)
{
	// check the size of kernel
	switch (kernelSize) {
	case 3:
		if (omega != -1.0)
			this->marrHildreth3x3(omega);
		else
			this->marrHildreth3x3();
		break;
	case 5:
		if (omega != -1.0)
			this->marrHildreth5x5(omega);
		else
			this->marrHildreth5x5();
		break;
	case 7:
		if (omega != -1.0)
			this->marrHildreth7x7(omega);
		else
			this->marrHildreth7x7();
		break;
	default:
		std::cerr << "Unknown Kernel size!" << std::endl;
		this->errorCode = EDT_UNKNOWN_KERNEL_SIZE;
		break;
	}
	
}


void EdgeDetectorsToolkit::applyLaplacian(int kernelSize)
{
	// check the size of kernel
	switch (kernelSize) {
	case 3:
		this->laplacian3x3();
		break;
	case 5:
		this->laplacian5x5();
		break;
	case 7:
		this->laplacian7x7();
		break;
	default:
		std::cerr << "Unknown Kernel size!" << std::endl;
		this->errorCode = EDT_UNKNOWN_KERNEL_SIZE;
		break;
	}
}

// function perform pixel by pixel compare with Canny edge detector from OpenCV
void EdgeDetectorsToolkit::compareWithCannyDetector(std::ofstream& outFile, int kernelSize)
{
	// generate reference file
	cv::Mat cannyEdge = cv::Mat::zeros(this->inputImage.size(), CV_8U);
	cv::Mat tempInput;
	this->inputImage.convertTo(tempInput, CV_8U);
	cv::Canny(tempInput, cannyEdge, 100, 200, 3);
	cannyEdge.convertTo(cannyEdge, CV_32FC1);
	//std::string str = std::to_string(kernelSize);
	//cv::imwrite(str+"cannyOut.jpg", cannyEdge);
	// normalize if needed
	cv::normalize(this->outputImage, this->outputImage, 0, 255, CV_MINMAX);
	// compare with canny pixel by pixel
	int missPixels = 0, falseAlarm = 0, pixelCount = 0, pixelCountMy = 0;
	for (int i = 0; i < cannyEdge.rows; i++) {
		for (int j = 0; j < cannyEdge.cols; j++) {
			if (cannyEdge.at<float>(i, j) == 0.0 && this->outputImage.at<float>(i, j) != 0.0)
				falseAlarm++;
			else if (cannyEdge.at<float>(i, j) != 0.0 && this->outputImage.at<float>(i, j) == 0.0)
				missPixels++;
			if (cannyEdge.at<float>(i, j) != 0.0)
				pixelCount++;
			if (this->outputImage.at<float>(i, j) != 0.0)
				pixelCountMy++;
		}
	}

	// write stats to file
	outFile << pixelCount << "  |  " << pixelCountMy << "  |  " <<falseAlarm  << "  |  " << missPixels << std::endl;;
}

