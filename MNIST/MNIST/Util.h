#pragma once
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdio.h>
#include <limits>

#include "lib/Eigen/Core"

using namespace std;
using namespace Eigen;

void relu(MatrixXd &x);
void softmax(MatrixXd &x);
VectorXi argmax(const MatrixXd &x);
double cross_entropy_discrete(const MatrixXd& probs, const VectorXi& labels);
MatrixXd crossentropy_softmax_gradient(const MatrixXd& probs, const VectorXi& labels);
MatrixXd relu_gradient(const MatrixXd& raws, const MatrixXd& vals);
vector<string> split_string(string s, char delim);