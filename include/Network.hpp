#pragma once
#include "DenseLayer.hpp"
#include "ReluAL.hpp"
#include "SoftmaxCE.hpp"
#include "optimizer.hpp"
#include "Layer.hpp"
#include <memory>

class Network{
    private:
        std::vector<std::unique_ptr<Layer>>layers;
        SoftmaxCrossentropy lossfunction;
        std::unique_ptr<optimizer> active_optimizer;// allows for active_optimizer=std::make_unique<child>();
    
    public:
        Network()=default;
        
        template <typename T, typename... Args>
        T& create_layer(Args&&...args){

            std::unique_ptr<Layer> layer= std::make_unique<T> (std::forward<Args>(args)...);
            T& reference=*layer;
            layers.push_back(std::move(layer));
            return reference;
        }

        void setOptimizer(std::unique_ptr<optimizer> newOptimizer);

        void train_model(const Matrix& input_matrix, const Matrix& target_matrix);

};