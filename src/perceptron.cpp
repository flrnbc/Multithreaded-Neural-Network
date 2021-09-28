#include <stdexcept>
#include <string>

#include "perceptron.h"

PerceptronShape::PerceptronShape(int rows, int cols, std::string activation) {
    if (rows < 1 or cols < 1) throw std::invalid_argument("Not enough rows or columns!");
    SetRows(rows);
    SetCols(cols);
    // TODO: add check for activation function
    SetActivationFct(activation);
}
