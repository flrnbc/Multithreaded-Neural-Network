#ifndef TRANSFORMATION_H_
#define TRANSFORMATION_H_

#include <memory>
#include <vector>
#include <string>


/*
    Abstract base class
*/
class Transformation {
    protected:
        int _rows;
        int _cols;

    public:
        // Transformation() {}; 
        virtual int Cols() { return _cols; }
        virtual int Rows() { return _rows; }
        virtual void SetCols(int cols) { _cols = cols; }
        virtual void SetRows(int rows) { _rows = rows; }
        
        // pure virtual functions
        // overloaded transform method (want to keep input, so no pass by reference)
        virtual std::vector<double> Transform(std::vector<double>) = 0;
        virtual std::vector<std::vector<double> > Transform(std::vector<std::vector<double> >) = 0;
        
        // TODO #A: backward/derivative
        
        // summary of transformation
        static std::string PrintDoubleVector(const std::vector<double>&); // simply attach function to Class
        virtual std::string Summary() = 0;
};


/*
    Concrete implementation: Linear transformation
*/ 
class LinearTransformation: public Transformation {
    private:    
        std::vector<std::vector<double> > _weights;
        std::vector<double > _bias;

    public:
        // (default) constructor
        LinearTransformation(std::vector<std::vector<double> > weights, std::vector<double> bias) :
            _weights(weights), _bias(bias) {};


        // setters & getters
        std::vector< std::vector<double> > Weights() { return _weights; }
        void SetWeights(std::vector<std::vector<double> > weight_matrix) {
            _weights = weight_matrix;
        }
        std::vector<double> Bias() { return _bias; }
        void SetBias(std::vector<double> bias) { _bias = bias; }

        // useful helper function to transpose
        static std::vector<std::vector<double> > Transpose(const std::vector<std::vector<double> >&);

        // transform methods (just right matrix multiplication with input)
        // NOTE: we do _not_ work with transpose etc.
        std::vector<double> Transform(std::vector<double>); 
        std::vector<std::vector<double> > Transform(std::vector<std::vector<double> >);

        // summary
        std::string Summary();
};

/*
    Concrete implementation: Activation transformation
*/
// collect activation functions
double heaviside(double);
double identity(double); 
double prelu(double, double);
double relu(double);
double sigmoid(double);
// tanh built-in

class ActivationTransformation: public Transformation {
    private:
        std::string _name;
        double (*activation_fct)(double); // function pointer seems to be needed

    public:
        // (default) constructor
        ActivationTransformation(int size, std::string fct_name="identity"); 
        
        // setters & getters
        std::string Name() { return _name; }
        void SetName(std::string fct_name) { _name = fct_name; }

        // transform methods
        std::vector<double> Transform(std::vector<double>);
        std::vector<std::vector<double> > Transform(std::vector<std::vector<double> >);

        // Summary of transformation
        std::string Summary(); 
};

// TODO: to include: Flatten layer


#endif // TRANSFORMATION_H_