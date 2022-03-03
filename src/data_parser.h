#ifndef _DATA_PARSER_H_
#define _DATA_PARSER_H_

#include <Eigen/Dense>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

/** 
    Class to parse data from a csv-file and one-hot-encode the data. At the moment with very limited functionality since we focus on 
    neural networks for now. 
    
    TODO: possible to add:
    + rejecting non-convertible values
    + option of how to read data from file (e.g. exchange rows and cols?)
    + iterator for data?
    + if more functionality will be added: split class into parser (only loading csv-file) and transfomer (one-hot-encode)
*/

class DataParser {
    public:
        // constructor
        DataParser() {}

        // helper function to load data from a csv-file (essentially taken from 
        // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix/39146048)
        // NOTE: definition needs to be included in the header file so that the compiler knows which 
        // function to generate
        template<typename M>
        M LoadCSV(const std::string& path) {
            std::ifstream indata;
            indata.open(path);
            std::string line;
            std::vector<double> values;
            uint rows = 0;
            while (std::getline(indata, line)) {
                std::stringstream lineStream(line);
                std::string cell;
                while (std::getline(lineStream, cell, ',')) {
                    values.push_back(std::stod(cell));
                }
                ++rows;
            }
            // transform std::vector<double> into matrix
            return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor> >(values.data(), rows, values.size()/rows);
        }

        // one-hot-encoder of an int (column) vector with values in [0, max]
        // TODO: sparse matrix better?
        Eigen::MatrixXd OneHotEncoder(const Eigen::RowVectorXd& vector, int max); 
};

#endif