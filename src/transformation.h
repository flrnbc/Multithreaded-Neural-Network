#ifndef TRANSFORMATION_H_
#define TRANSFORMATION_H_

#include <memory>
#include <vector>
#include <string>


/***********************
 * ABSTRACT BASE CLASS *
 ***********************/

class Transformation {
    protected:
        // output dimension
        int _rows;
        // input dimension   
        int _cols;
        // type of transformation
        std::string _type;

        Transformation(int rows, int cols, std::string type):
            _rows(rows),
            _cols(cols),
            _type(type)
            {}

    public:
        // virtual destructor
        virtual ~Transformation() {}

        // getters 
        // NOTE: no setters because the attributes are not supposed to be changed later on
        int Cols() { return _cols; }
        int Rows() { return _rows; }
        std::string Type() { return _type; }
        
        // transform method (want to keep input, so no pass by reference)
        virtual std::vector<double> Transform(std::vector<double>) = 0;
        //std::vector<std::vector<double> > Transform(std::vector<std::vector<double> >);
        
        // TODO #A: backward/derivative

        // initialize transformation (many transformations have empty implementation)
        virtual void Initialize(std::string initialize_type="") {}
        
        static std::string PrintDoubleVector(const std::vector<double>&); // TODO: move somewhere else
        
        // Summary of transformation
        virtual std::string Summary() = 0;
};


/**************************************************
 * CONCRETE IMPLEMENTATION: LINEAR TRANSFORMATION *
 **************************************************/

class LinearTransformation: public Transformation {
    private:    
        std::vector<std::vector<double> > _weights;
        std::vector<double > _bias;

    public:
        // constructors
        // constructor for (affine) linear transformations
        LinearTransformation(std::vector<std::vector<double> > weights, std::vector<double> bias): 
            Transformation(weights.size(), weights[0].size(), "LinearTransformation"),
            _weights(weights),
            _bias(bias)
            {}

        // constructor (all zeros)
        LinearTransformation(int rows, int cols): 
            Transformation(rows, cols, "LinearTransformation"),
            _weights(std::vector<std::vector<double> >(rows, std::vector<double>(cols, 0))), 
            _bias(std::vector<double>(rows, 0))
            {}

        // TODO: the default copy and move constructors/assignment operators should be enough?!?

        // setters & getters
        std::vector< std::vector<double> > Weights() { return _weights; }
        void SetWeights(std::vector<std::vector<double> > weight_matrix) {
            _weights = weight_matrix;
        }
        std::vector<double> Bias() { return _bias; }
        void SetBias(std::vector<double> bias) { _bias = bias; }

        // random initialize (either via "He" or "Normalized Xavier")
        void Initialize(std::string initialize_type);

        // useful helper function to transpose
        static std::vector<std::vector<double> > Transpose(const std::vector<std::vector<double> >&);

        // transform methods (just right matrix multiplication with input)
        // NOTE: we do _not_ work with transpose etc.
        std::vector<double> Transform(std::vector<double>); 
        //std::vector<std::vector<double> > Transform(std::vector<std::vector<double> >);

        // summaries
        std::string Summary();
};


/******************************************************
 * CONCRETE IMPLEMENTATION: ACTIVATION TRANSFORMATION *
 ******************************************************/

// collect activation functions
double heaviside(double);
double identity(double); 
double prelu(double, double);
double relu(double);
double sigmoid(double);
// tanh built-in

class ActivationTransformation: public Transformation {
    private:
        double (*activation_fct)(double); // function pointer seems to be needed

    public:
        // constructor
        ActivationTransformation(int size, std::string fct_name="identity"): 
            Transformation(size, size, fct_name)       
            {
                if (fct_name == "heaviside") {
                    activation_fct = &heaviside;
                }
                else if (fct_name == "identity") {
                    activation_fct = &identity;
                }
                // TODO #B: to simplify, only offer relu and not prelu at the moment
                else if (fct_name == "relu") {
                    activation_fct = &relu;
                }
                else if (fct_name == "sigmoid") {
                    activation_fct = &sigmoid;
                }
                else if (fct_name == "tanh") {
                    activation_fct = &tanh;
                }
                else throw std::invalid_argument("Not a valid activation function.");
             }

        // transform methods
        std::vector<double> Transform(std::vector<double>);
        //std::vector<std::vector<double> > Transform(std::vector<std::vector<double> >);

        // Summary of transformation
        std::string Summary(); 
};

// TODO: to include: Flatten layer

#endif // TRANSFORMATION_H_