// EdgeDetector.cpp : Defines the entry point for the console application.
//
#include <string>
#include "EdgeDetectorsToolkit.h"

void runDemo(EdgeDetectorsToolkit *edgeDetector, std::string outputPath) {
	// create ofstream to write results
	std::ofstream ofs("compare.txt", std::ofstream::out);

	// prewitt operator
	edgeDetector->applyPrewitt(3);
	edgeDetector->findEdges();
	ofs << "Prewitt 3x3: ";
	edgeDetector->compareWithCannyDetector(ofs, 3);
	edgeDetector->saveImage(outputPath + "prewitt3x3.jpg");
	edgeDetector->applyPrewitt(5);
	edgeDetector->findEdges();
	ofs << "Prewitt 5x5: ";
	edgeDetector->compareWithCannyDetector(ofs, 5);
	edgeDetector->saveImage(outputPath + "prewitt5x5.jpg");
	edgeDetector->applyPrewitt(7);
	edgeDetector->findEdges();
	ofs << "Prewitt 7x7: ";
	edgeDetector->compareWithCannyDetector(ofs, 7);
	edgeDetector->saveImage(outputPath + "prewitt7x7.jpg");
	// sobel operator
	edgeDetector->applySobel(3);
	edgeDetector->findEdges();
	ofs << "Sobel 3x3: ";
	edgeDetector->compareWithCannyDetector(ofs, 3);
	edgeDetector->saveImage(outputPath + "sobel3x3.jpg");
	edgeDetector->applySobel(5);
	edgeDetector->findEdges();
	ofs << "Sobel 5x5: ";
	edgeDetector->compareWithCannyDetector(ofs, 5);
	edgeDetector->saveImage(outputPath + "sobel5x5.jpg");
	edgeDetector->applySobel(7);
	edgeDetector->findEdges();
	ofs << "Sobel 7x7: ";
	edgeDetector->compareWithCannyDetector(ofs, 7);
	edgeDetector->saveImage(outputPath + "sobel7x7.jpg");
	// kirsch operator
	edgeDetector->applyKirsch(3);
	edgeDetector->findEdges();
	ofs << "Kirsch 3x3: ";
	edgeDetector->compareWithCannyDetector(ofs, 3);
	edgeDetector->saveImage(outputPath + "kirsch3x3.jpg");

	edgeDetector->applyKirsch(5);
	edgeDetector->findEdges();
	ofs << "Kirsch 5x5: ";
	edgeDetector->compareWithCannyDetector(ofs, 5);
	edgeDetector->saveImage(outputPath + "kirsch5x5.jpg");

	edgeDetector->applyKirsch(7);
	edgeDetector->findEdges();
	ofs << "Kirsch 7x7: ";
	edgeDetector->compareWithCannyDetector(ofs, 7);
	edgeDetector->saveImage(outputPath + "kirsch7x7.jpg");

	// robinsons operator
	edgeDetector->applyRobinson(3);
	edgeDetector->findEdges();
	ofs << "Robinson 3x3: ";
	edgeDetector->compareWithCannyDetector(ofs, 3);
	edgeDetector->saveImage(outputPath + "robinson3x3.jpg");

	edgeDetector->applyRobinson(5);
	edgeDetector->findEdges();
	ofs << "Robinson 5x5: ";
	edgeDetector->compareWithCannyDetector(ofs, 5);
	edgeDetector->saveImage(outputPath + "robinson5x5.jpg");

	edgeDetector->applyRobinson(7);
	edgeDetector->findEdges();
	ofs << "Robinson 7x7: ";
	edgeDetector->compareWithCannyDetector(ofs, 7);
	edgeDetector->saveImage(outputPath + "robinson7x7.jpg");
	// change threshold for second derivative
	edgeDetector->setThreshold(0.5);
	// laplacian operator
	edgeDetector->applyLaplacian(3);
	edgeDetector->findEdges();
	ofs << "Laplacian 3x3: ";
	edgeDetector->compareWithCannyDetector(ofs, 3);
	edgeDetector->saveImage(outputPath + "laplacian3x3.jpg");

	edgeDetector->applyLaplacian(5);
	edgeDetector->findEdges();
	ofs << "Laplacian 5x5: ";
	edgeDetector->compareWithCannyDetector(ofs, 5);
	edgeDetector->saveImage(outputPath + "laplacian5x5.jpg");

	edgeDetector->applyLaplacian(7);
	edgeDetector->findEdges();
	ofs << "Laplacian 7x7: ";
	edgeDetector->compareWithCannyDetector(ofs, 7);
	edgeDetector->saveImage(outputPath + "laplacian7x7.jpg");

	// change threshold for second derivative
	edgeDetector->setThreshold(0.59);
	// marr-hildreth operator
	edgeDetector->applyMarrHildreth(3);
	edgeDetector->findEdges();
	ofs << "MarrHildreth 3x3: ";
	edgeDetector->compareWithCannyDetector(ofs, 3);
	edgeDetector->saveImage(outputPath + "marrHildreth3x3.jpg");

	edgeDetector->applyMarrHildreth(5);
	edgeDetector->findEdges();
	ofs << "MarrHildreth 5x5: ";
	edgeDetector->compareWithCannyDetector(ofs, 5);
	edgeDetector->saveImage(outputPath + "marrHildreth5x5.jpg");

	edgeDetector->applyMarrHildreth(7);
	edgeDetector->findEdges();
	ofs << "MarrHildreth 7x7: ";
	edgeDetector->compareWithCannyDetector(ofs, 7);
	edgeDetector->saveImage(outputPath + "marrHildreth7x7.jpg");

	// close stat file
	ofs.close();
}

int main(int argc, char* argv[])
{
	// variables to ge params
	std::string method = "";
	std::string inputPath = "";
	std::string outputPath = "";
	int kernelSize = 3;
	float threshold = 0.5;
	bool mark = false;
	bool demo = false;

	// get input parameters
	if (argc < 5 || argc > 8) {
		std::cerr << "Incorrect argument count!" << std::endl;
		std::cerr << "Use: -METHOD KERNELSIZE -threshold VALUE [-type MARK|FIND] INPUTFILE OUTPUTFILE" << std::endl;
		exit(1);
	}
	else { // take parameters
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i],"-pre") == 0 || strcmp(argv[i],"-sob") == 0 || strcmp(argv[i],"-kir") == 0 || strcmp(argv[i],"-rob")==0 || strcmp(argv[i],"-lap")==0 || strcmp(argv[i],"-mah")==0) {
				method = argv[i];
				kernelSize = atoi(argv[i + 1]);
				i++;
			} else if (strcmp(argv[i],"-threshold")==0) {
				threshold = atof(argv[i + 1]);
				i++;
			} else if (strcmp(argv[i],"-mark")==0) {
				mark = true;
			} else if (strcmp(argv[i],"-demo")==0) {
				demo = true;
			} else {
				if (inputPath == "")
					inputPath = argv[i];
				else
					outputPath = argv[i];
			}
		}
	}
	//std::cout << "method: " << method << ", threshold: " << threshold << ", kernelSize: " << kernelSize << ", mark:" << mark << ", inputImage: " << inputPath << ", output: " << outputPath << ", demo: " << demo << std::endl;


	// create edgeDetectorToolkit and load image and set threshold
	EdgeDetectorsToolkit *edgeDetector = new EdgeDetectorsToolkit();
	edgeDetector->loadImage(inputPath);
	if (edgeDetector->errorCode != 0) {
		exit(edgeDetector->errorCode);
	}
	edgeDetector->setThreshold(threshold);
	
	if (!demo) {
		// find edges by given method
		if (method.compare("-pre") == 0)
			edgeDetector->applyPrewitt(kernelSize);
		else if (method.compare("-sob") == 0)
			edgeDetector->applySobel(kernelSize);
		else if (method.compare("-kir") == 0)
			edgeDetector->applyKirsch(kernelSize);
		else if (method.compare("-rob") == 0)
			edgeDetector->applyRobinson(kernelSize);
		else if (method.compare("-lap") == 0)
			edgeDetector->applyLaplacian(kernelSize);
		else if (method.compare("-mah") == 0)
			edgeDetector->applyMarrHildreth(kernelSize);

		// find edges using given threshold if option mark not given
		if (!mark)
			edgeDetector->findEdges();

		// save image to file
		edgeDetector->saveImage(outputPath);
	}
	else { // choose demo, use all operators and save files to given path
		runDemo(edgeDetector, outputPath);
	}

    return 0;
}

