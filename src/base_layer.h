#ifndef BASE_LAYER_H_
#define BASE_LAYER_H_

#include <memory>
#include <string>
#include <vector>

class Transformation;

class Cache;

class BaseLayer {
    private:
        // cache for forward and backward pass
        // TODO #A: really shared pointer?
        std::shared_ptr<Cache> _forward_cache;
        std::shared_ptr<Cache> _backward_cache;
        // transformation
        std::shared_ptr<Transformation> _transformation;

    

};




#endif // BASE_LAYER_H_