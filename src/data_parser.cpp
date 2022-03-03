#include <Eigen/Dense>
#include "data_parser.h"
#include <cmath>
#include <vector>
#include <fstream>

Eigen::MatrixXd DataParser::OneHotEncoder(const Eigen::RowVectorXd& vector, int max) {
        Eigen::MatrixXd oneHotEncoded = Eigen::MatrixXd::Zero(max+1, vector.size());
        for (int i=0; i<vector.size(); i++) {
            int j = (int)vector(i);
            oneHotEncoded(j, i) = 1.0; // conversion should be ok; or is ther an issue with e.g. 5 saved as 4.99999999?
        }
        return oneHotEncoded;
    }