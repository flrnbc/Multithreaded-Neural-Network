#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <string>
#include <vector>


class Transformation;
class ActivationTransformation;
class LinearTransformation;

/****************
 * LINEAR CACHE *
 ****************/

class Layer {
    private:
        // ownership important: the corresponding resources will be shared with different pointers
        // (inside the Layers of other Layers)
        std::shared_ptr<std::vector<double> > _forward_input;
        std::shared_ptr<std::vector<double> > _forward_output;
        std::shared_ptr<std::vector<double> > _backward_input;
        std::shared_ptr<std::vector<double> > _backward_output;

     public: 
        // NOTE: backward/forward output should be computed, NOT set.
        // forward
        void SetForwardInput(std::shared_ptr<std::vector<double> > input_ptr); 
        void SetForwardOutput(std::shared_ptr<std::vector<double> > output_ptr);
        std::shared_ptr<std::vector<double> > GetForwardOutput();
        std::shared_ptr<std::vector<double> > GetForwardInput();
        // backward
        void SetBackwardInput(std::shared_ptr<std::vector<double> > backward_input_ptr);
        std::shared_ptr<std::vector<double> > GetBackwardOutput();

        // forward pass
        virtual void Forward() = 0;
        // backward pass
        // virtual void Backward() = 0;

        // summary
        virtual std::string Summary() = 0;
};


/**********************
 * LINEAR LAYER CLASS *
 **********************/
 
// TODO: need to refactor, e.g. with an interface Layer or a factory method?
// issues with interface Layer: Transformation class and downcasting

class LinearLayer: public Layer {
    private:
        std::unique_ptr<LinearTransformation> _transformation;

    public:
        // constructor
        LinearLayer(int rows, int cols);
        // destructor
        // TODO: always needed?
        ~LinearLayer() {};

        // initialize
        void Initialize(std::string initialization_type);

        // forward pass
        void Forward();
        // backward pass AND updating weights
        //void Backward() {}

        // summary
        std::string Summary();
};


/********************
 * ACTIVATION LAYER *
 ********************/

class ActivationLayer: public Layer {
    private:
        std::unique_ptr<ActivationTransformation> _transformation;
        std::string _activation;

    public:
        // default constructor
        ActivationLayer(int, std::string);
        // destructor
        // TODO: always needed?
        ~ActivationLayer() {};

        // setters/getters
        
        std::string GetActivationString() { return _activation; }

        // forward pass
        void Forward();
        // backward pass AND updating weights
        //void Backward() {}

        // summary
        std::string Summary();
};



// flatten layer --> extra layer class
        //static std::vector<double> Flatten(const std::vector<std::vector<double> >& matrix);



#endif // LAYER_H_