#ifndef LAYER_H_
#define LAYER_H_

#include <memory>
#include <string>
#include <vector>

class Transformation;
class LinearTransformation;

/*
    (Abstract) classes to fix concepts
*/

class Cache {
    /*
        Cache for layers to save vectors in computations and update from
        other caches.
        TODO: use template
    */
    private: 
        std::vector<double> _cached;

    public:
        // default constructor from vector
        Cache(std::vector<double>);
        // simplified constructor (zero vector of given length)
        Cache(int input_size);
        
        // setters/getters
        void Set(std::vector<double>); // TODO: better to use template
        std::vector<double> Get();

        // reset to zero vector
        void Reset();
};


/************************
 * ABSTRACT LAYER CLASS *
 ************************/

class Layer {
    protected: // it's fine that derived (concrete) classes access these attributes
        std::shared_ptr<Transformation> _transformation;
        // TODO: the following are created as nullptr's?!?
        std::shared_ptr<std::vector<double> > _forward_input;
        std::shared_ptr<std::vector<double> > _forward_output;
        std::shared_ptr<std::vector<double> > _backward_input;
        std::shared_ptr<std::vector<double> > _backward_output;

    public:
        // destructor
        // TODO: always needed in an abstract class?
        ~Layer() {};
        // forward pass (updates _forward_output)
        void Forward();
        // backward pass (updates _backward_output)
        virtual void Backward() = 0;

        // setters/getters
        // NOTE: backward/forward output should be computed, NOT set.
        // forward
        void SetForwardInput(std::shared_ptr<std::vector<double> > input_ptr); 
        std::shared_ptr<std::vector<double> > GetForwardOutput();
        // backward
        void SetBackwardInput(std::shared_ptr<std::vector<double> > backward_input_ptr);
        std::shared_ptr<std::vector<double> > GetBackwardOutput(); 

        // summary 
        // TODO: might change to include _forward_input etc.
        std::string Summary(); 
};


/****************************
 * CONCRETE IMPLEMENTATIONS *
 ****************************/

class LinearLayer : public Layer {
    public:
        // simplified constructor
        // TODO: might add others later but for now sufficient
        LinearLayer(int rows, int cols);

        // initialize
        void Initialize(std::string initialization_type);

        // backward pass AND updating weights
        void Backward() {}
};

// flatten layer --> extra layer class
        //static std::vector<double> Flatten(const std::vector<std::vector<double> >& matrix);



#endif // LAYER_H_