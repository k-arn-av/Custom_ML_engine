#pragma once
#include "DenseLayer.hpp"
#include <memory>

class optimizer{
    protected:
        double learning_rate;
    
    public:
        optimizer(double LR):learning_rate(LR){}

        virtual ~optimizer()=default;

        virtual void GradientDescent(std::vector<std::unique_ptr<Layer>>& Layers)=0;
};

class SGD: public optimizer{

    public:
        SGD(double LR):optimizer(LR){}

        void GradientDescent(std::vector<std::unique_ptr<Layer>>& Layers) override;

};