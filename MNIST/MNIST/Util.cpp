#include "Util.h"

void relu(MatrixXd &x) {
	for (int i = 0; i < x.rows(); i++) {
		for (int j = 0; j < x.cols(); j++) {
			if (x(i, j) < 0)
				x(i, j) = 0;
		}
	}
}

void softmax(MatrixXd &x) {
	for (int i = 0; i < x.rows(); i++) {
		double max = -numeric_limits<double>::max();
		for (int j = 0; j < x.cols(); j++) {
			if (x(i, j) > max)
				max = x(i, j);
		}
		double esum = 0;
		for (int j = 0; j < x.cols(); j++) {
			x(i, j) = exp(x(i, j) - max);
			esum += x(i, j);
		}
		for (int j = 0; j < x.cols(); j++) {
			x(i, j) = x(i, j) / esum;
		}
	}
}

VectorXi argmax(const MatrixXd &x) {
	VectorXi result(x.rows());

	for (int i = 0; i < x.rows(); i++) {
		int amax = -1;
		double max = -numeric_limits<double>::max();
		for (int j = 0; j < x.cols(); j++) {
			if (x(i, j) > max) {
				amax = j;
				max = x(i, j);
			}
		}		
		result(i) = amax;
	}

	return result;
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

double cross_entropy_discrete(const MatrixXd& probs, const VectorXi& labels)
{
	double sum = 0;
	for (int i = 0; i < labels.rows(); i++)
		sum += log(probs(i, labels(i)));
	return -sum / labels.rows();
}