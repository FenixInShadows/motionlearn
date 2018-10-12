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
vector<string> split_string(string s, char delim);