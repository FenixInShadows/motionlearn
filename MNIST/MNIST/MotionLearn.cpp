// MotionLearn.cpp : Defines the entry point for the console application.


#include "lib/Eigen/Core"
#include <cassert>
#include <string>
#include <vector>
#include <iostream>

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
	inputs = data.rightCols(data.cols() - 1).transpose().cast<double>();
}

void ForwardProp(const MatrixXd& inputs, const MatrixXd& inputToHidden, const MatrixXd& hiddenToOutput, MatrixXd& hiddenLayer, MatrixXd& outputLayer)
{
	hiddenLayer = inputToHidden * inputs;
	relu(hiddenLayer);
	outputLayer = hiddenToOutput * hiddenLayer;
	softmax(outputLayer);
}

void BackProp(const MatrixXd& inputs, const MatrixXd& inputToHidden, const MatrixXd& hiddenToOutput, const MatrixXd& hiddenLayer, const MatrixXd& outputLayer, const VectorXi& labels, MatrixXd& inputToHiddenGrad, MatrixXd& hiddenToOutputGrad)
{
	MatrixXd dfdz = crossentropy_softmax_gradient(outputLayer, labels);
	hiddenToOutputGrad = dfdz * hiddenLayer.transpose()/inputs.cols();
	MatrixXd dfdy = relu_gradient(hiddenToOutput.transpose() * dfdz, hiddenLayer);
	inputToHiddenGrad = dfdy * inputs.transpose()/inputs.cols();
}

void ForwardProp_Adv(const MatrixXd& inputs, const MatrixXd* weights, MatrixXd* hiddenLayers, MatrixXd& outputLayer)
{

}

void BackProp_Adv(const MatrixXd& inputs, const MatrixXd* weights, const MatrixXd* hiddenLayers, const MatrixXd& outputLayer, const VectorXi& labels, MatrixXd* weightGrad, int batchSize)
{

}

double CostEval(const MatrixXd& probs, const VectorXi& labels)
{
	return cross_entropy_discrete(probs, labels);
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

	// testing
	cout << "start" << endl;

	int nHidden = 100;
	int nClasses = 10;
	int numHiddenLayers = 1;
	vector<int> nHiddens{ nHidden };
	int batchSize;

	// parse arguments
	while (argc > 0)
	{
		if (**argv == '-')
		{
			if (!strcmp(*argv, "-nHidden")) 
			{
				int numopts = 1;
				// numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);

				nHidden = atoi(argv[1]); // this is used as single hidden layer set-up
				nHiddens[0] = nHidden; // this is used as multiple hiddden layer set-up

				argv += numopts + 1, argc -= numopts + 1;
			}
			else if (!strcmp(*argv, "-nHiddens"))
			{
				int numopts = 1;
				// numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);

				// read the number of hidden layers
				numHiddenLayers = atoi(argv[1]);

				argv += numopts + 1, argc -= numopts + 1;

				// multiple hiddden layer set-up
				nHiddens.resize(numHiddenLayers);
				for (int i = 0; i < numHiddenLayers; i++)
					nHiddens[i] = atoi(argv[i]);
				// single hiddden layer set-up	
				nHidden = nHiddens[0];

				argv += numHiddenLayers, argc -= numHiddenLayers;
			}
			else if (!strcmp(*argv, "-trainSet"))
			{
				int numopts = 1;
				// numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);
				
				// Load the data from the test set
				ReadData(argv[1], trainInput, trainLabel);
				cout << "Read in training data with " << trainInput.cols() << " samples, each with " << trainInput.rows() << " dimensions." << endl;

				batchSize = trainInput.cols();
				
				argv += numopts + 1, argc -= numopts + 1;
			}
			else if (!strcmp(*argv, "-testSet"))
			{
				int numopts = 1;
				// numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts+1);
				
				// Load the data from the test set
				ReadData(argv[1], testInput, testLabel);
				cout << "Read in testing data with " << testInput.cols() << " samples, each with " << testInput.rows() << " dimensions." << endl;

				argv += numopts + 1, argc -= numopts + 1;
			}
			else if (!strcmp(*argv, "-forwardProp"))
			{
				int numopts = 0;
				// numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);
				
				MatrixXd inputToHidden = MatrixXd::Random(nHidden, 784) * 0.1;
				MatrixXd hiddenToOutput = MatrixXd::Random(nClasses, nHidden) * 0.1;
				MatrixXd hiddenLayer;
				MatrixXd outputLayer;
				ForwardProp(testInput, inputToHidden, hiddenToOutput, hiddenLayer, outputLayer);

				cout << "Printing the result for the first 5 samples:" << endl;
				for (int j = 0; j < 5; j++)
					cout << outputLayer.col(j).transpose() << " Ground Truth: " << testLabel(j) << endl;

				cout << "The value of the cost function:" << endl;
				double cost = CostEval(outputLayer, testLabel);
				cout << cost << endl;
			
				argv += numopts + 1, argc -= numopts + 1;
			}
			else if (!strcmp(*argv, "-backProp"))
			{
				int numopts = 0;
				// numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);

				MatrixXd inputToHidden = MatrixXd::Random(nHidden, 784) * 0.1;
				MatrixXd hiddenToOutput = MatrixXd::Random(nClasses, nHidden) * 0.1;
				MatrixXd hiddenLayer;
				MatrixXd outputLayer;

				// test
				ForwardProp(testInput, inputToHidden, hiddenToOutput, hiddenLayer, outputLayer);
				
				cout << "The value of the cost function:" << endl;
				cout << CostEval(outputLayer, testLabel) << endl;
				
				// back prop
				MatrixXd inputToHiddenGrad;
				MatrixXd hiddenToOutputGrad;
				BackProp(testInput, inputToHidden, hiddenToOutput, hiddenLayer, outputLayer, testLabel, inputToHiddenGrad, hiddenToOutputGrad);

				cout << endl << "finished backprob" << endl;

				// update
				inputToHidden -= 0.001 * inputToHiddenGrad;
				hiddenToOutput -= 0.001 * hiddenToOutputGrad;

				// re-test
				ForwardProp(testInput, inputToHidden, hiddenToOutput, hiddenLayer, outputLayer);

				cout << "The value of the cost function:" << endl;
				cout << CostEval(outputLayer, testLabel) << endl;

				argv += numopts + 1, argc -= numopts + 1;
			}
			else if (!strcmp(*argv, "-ML"))
			{
				int numopts = 1;
				// numopts+1 because parameter name itself counts
				CheckOption(*argv, argc, numopts + 1);

				// read additional arguments
				int n_iter = atoi(argv[1]);

				//TODO: call network function
				MatrixXd inputToHidden = MatrixXd::Random(nHidden, 784) * 0.1;
				MatrixXd hiddenToOutput = MatrixXd::Random(nClasses, nHidden) * 0.1;
				MatrixXd trainHiddenLayer, testHiddenLayer;
				MatrixXd trainOutputLayer, testOutputLayer;

				// initial test
				ForwardProp(trainInput, inputToHidden, hiddenToOutput, trainHiddenLayer, trainOutputLayer);
				ForwardProp(testInput, inputToHidden, hiddenToOutput, testHiddenLayer, testOutputLayer);
				cout << "Training Eval: " << CostEval(trainOutputLayer, trainLabel) << endl;
				cout << "Testing Eval: " << CostEval(testOutputLayer, testLabel) << endl;


				// backprob on the training set, repeat for n_iter interations
				MatrixXd inputToHiddenGrad;
				MatrixXd hiddenToOutputGrad;

				for (int i = 0; i < n_iter; i++)
				{
					// backprob
					cout << endl << "backprob iteration " << i << endl;
					BackProp(trainInput, inputToHidden, hiddenToOutput, trainHiddenLayer, trainOutputLayer, trainLabel, inputToHiddenGrad, hiddenToOutputGrad);

					// update
					inputToHidden -= 0.001 * inputToHiddenGrad;
					hiddenToOutput -= 0.001 * hiddenToOutputGrad;

					// re-test
					ForwardProp(trainInput, inputToHidden, hiddenToOutput, trainHiddenLayer, trainOutputLayer);
					ForwardProp(testInput, inputToHidden, hiddenToOutput, testHiddenLayer, testOutputLayer);

					cout << "Training Eval: " << CostEval(trainOutputLayer, trainLabel) << endl;
					cout << "Testing Eval: " << CostEval(testOutputLayer, testLabel) << endl;					
				}

				cout << endl << "Printing the result for the first 5 samples in the train set:" << endl;
				for (int j = 0; j < 5; j++)
					cout << trainOutputLayer.col(j).transpose() << " Ground Truth: " << trainLabel(j) << endl;

				cout << endl << "Printing the result for the first 5 samples in the test set:" << endl;
				for (int j = 0; j < 5; j++)
					cout << testOutputLayer.col(j).transpose() << " Ground Truth: " << testLabel(j) << endl;

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
