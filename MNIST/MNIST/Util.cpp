#include "Util.h"

void relu(MatrixXd &x) {
	for (int j = 0; j < x.cols(); j++) {
		for (int i = 0; i < x.rows(); i++) {
			if (x(i, j) < 0)
				x(i, j) = 0;
		}
	}
}

void softmax(MatrixXd &x) {
	for (int j = 0; j < x.cols(); j++) {
		double max = -numeric_limits<double>::max();
		for (int i = 0; i < x.rows(); i++) {
			if (x(i, j) > max)
				max = x(i, j);
		}
		double esum = 0;
		for (int i = 0; i < x.rows(); i++) {
			x(i, j) = exp(x(i, j) - max);
			esum += x(i, j);
		}
		for (int i = 0; i < x.rows(); i++) {
			x(i, j) = x(i, j) / esum;
		}
	}
}

VectorXi argmax(const MatrixXd &x) {
	VectorXi result(x.cols());

	for (int j = 0; j < x.cols(); j++) {
		int amax = -1;
		double max = -numeric_limits<double>::max();
		for (int i = 0; i < x.rows(); i++) {
			if (x(i, j) > max) {
				amax = j;
				max = x(i, j);
			}
		}		
		result(j) = amax;
	}

	return result;
}

double accuracy(const MatrixXd &x, const VectorXi& labels)
{
	int count = 0;

	for (int j = 0; j < x.cols(); j++) {
		int amax = -1;
		double max = -numeric_limits<double>::max();
		for (int i = 0; i < x.rows(); i++) {
			if (x(i, j) > max) {
				amax = i;
				max = x(i, j);
			}
		}
		if (labels(j) == amax)
			count++;
	}

	return (double)count / (double)x.cols();
}

double cross_entropy_discrete(const MatrixXd& probs, const VectorXi& labels)
{
	double sum = 0;
	for (int j = 0; j < probs.cols(); j++)
		sum += log(probs(labels(j), j));
	return -sum / probs.cols();
}

MatrixXd crossentropy_softmax_gradient(const MatrixXd& probs, const VectorXi& labels)
{
	MatrixXd result = probs;

	for (int j = 0; j < probs.cols(); j++)
		result(labels(j), j) -= 1.0;

	return result;
}

MatrixXd relu_gradient(const MatrixXd& raws, const MatrixXd& vals)
{
	MatrixXd result(raws.rows(), raws.cols());

	for (int j = 0; j < raws.cols(); j++)
		for (int i = 0; i < raws.rows(); i++)
			result(i, j) = (vals(i, j) > 0 ? raws(i, j) : 0.0);

	return result;
}

void random_shuffle_in_place(vector<int>& list)
{
	for (int i = list.size() - 1; i > 0; i--)
	{
		int j = rand() % i;
		int tmp = list[j];
		list[j] = list[i];
		list[i] = tmp;
	}
}

std::vector<std::string> split_string(std::string s, char delim) {
	std::istringstream ss(s);

	std::vector<std::string> parts;
	std::string part;
	while (std::getline(ss, part, delim)) {
		if (!part.empty()) {
			parts.push_back(part);
		}
	}

	return parts;
}

