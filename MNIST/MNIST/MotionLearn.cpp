// MotionLearn.cpp : Defines the entry point for the console application.


#include "lib/Eigen/Core"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "MIO.h"

using namespace std;
using namespace Eigen;

/**
* prototypes
**/
static void ShowUsage(void);
static void CheckOption(char *option, int argc, int minargc);

void ReadData(const char* file, MatrixXd& inputs, VectorXi& labels)
{
	MatrixXi data = matrixFromFile(file, 0, ',');
	labels = data.col(0);
	inputs = data.rightCols(data.cols() - 1).cast<double>();
}

void ForwardProp()
{

}

bool verbose = false;

int main(int argc, char* argv[]) {
	// first argument is program name
	argv++, argc--;

	// look for help
	for (int i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "-help")) {
			ShowUsage();
		}
	}
	//apply options 
	for (int i = 0; i < argc; i++) {
		if (!strcmp(*argv, "-v"))
		{
			verbose = true;
			argv += 1, argc -= 1;
		}
		
	}
	// no argument case
	if (argc == 0) {
		ShowUsage();
	}

	MatrixXd trainInput;
	VectorXi trainLabel;
	MatrixXd testInput;
	VectorXi testLabel;

	// parse arguments
	while (argc > 0)
	{
		if (**argv == '-')
		{
			if (!strcmp(*argv, "-trainSet"))
			{
				int numopts = 1;
				//numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);
				
				//Load the data from the test set
				ReadData(argv[1], trainInput, trainLabel);
				cout << "Read in training data with " << trainInput.rows() << " samples, each with " << trainInput.cols() << " dimensions." << endl;
				
				argv += numopts + 1, argc -= numopts + 1;
			}
			else if (!strcmp(*argv, "-testSet"))
			{
				int numopts = 1;
				//numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts+1);
				
				//Load the data from the test set
				ReadData(argv[1], testInput, testLabel);
				cout << "Read in testing data with " << testInput.rows() << " samples, each with " << testInput.cols() << " dimensions." << endl;

				argv += numopts + 1, argc -= numopts + 1;
			}
			else if (!strcmp(*argv, "-forwardProp"))
			{
				int numopts = 0;
				//numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);
				
				//TODO: call network function				
				int nHidden = 100;
				int nClasses = 10;
				MatrixXd inputToHidden = MatrixXd::Random(784, nHidden) * 0.1;
				MatrixXd hiddenToOutput = MatrixXd::Random(nHidden, nClasses) * 0.1;
				MatrixXd hiddenLayer = testInput * inputToHidden;
				relu(hiddenLayer);
				MatrixXd outputLayer = hiddenLayer * hiddenToOutput;
				//VectorXi labelPredict = argmax(outputLayer);
				softmax(outputLayer);
				cout << outputLayer.topRows(5) << endl;
			
				argv += numopts + 1, argc -= numopts + 1;
			}
			else
			{
				fprintf(stderr, "invalid option: %s\n", *argv);
				ShowUsage();
			}
		}
		else
		{
			fprintf(stderr, "DeepNav: invalid option (2): %s\n", *argv);
			ShowUsage();
		}
	}

	return EXIT_SUCCESS;
}


/**
* ShowUsage
**/
static char options[] =
"-help (show this message)\n"
"-v verbose output\n"
"- forwardProp\n"
;

static void ShowUsage(void)
{
	fprintf(stderr, "Usage: DeepNav [-option [arg ...] ...] -output \n");
	fprintf(stderr, "%s", options);
	exit(EXIT_FAILURE);
}



/**
* CheckOption
**/
static void CheckOption(char *option, int argc, int minargc)
{
	if (argc < minargc)
	{
		fprintf(stderr, "Too few arguments for %s, expected %d, received %d\n", option, minargc, argc);
		ShowUsage();
	}
}
