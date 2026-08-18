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
        std::unique_ptr<optimizer> active_optimizer;
    
    public:
        template <typename T, typename... arguments>
        T& create_layer(arguments&&... other_arguments){

            std::unique_ptr<Layer> layer= std::make_unique<T> (std::forward<arguments>(other_arguments)...);
            T& reference=*layer;
            layers.push_back(std::move(layer));
            return reference;
        }

        void setOptimizer(std::unique_ptr<optimizer> newOptimizer);

        void train_model(const Matrix& input_matrix, const Matrix& target_matrix);

};