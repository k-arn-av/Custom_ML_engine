#pragma once
#include "DenseLayer.hpp"
#include "ReluAL.hpp"
#include "SoftmaxCE.hpp"
#include "optimizer.hpp"
#include "Layer.hpp"
#include <memory>
#include <type_traits>

class Network{
    private:
        std::vector<std::unique_ptr<Layer>>layers;
        SoftmaxCrossentropy lossfunction;

        //Note, creating unique ptr object in py causes double free error, so used shared ptr for both cpp and py to work
        std::shared_ptr<optimizer> active_optimizer;// allows for active_optimizer=std::make_unique<child>();
        size_t current_feature_size=0;

    public:
        Network()=default;

        //current_feature size setter for auto input_feature calculation in compiletime for denseLayers
        void setCurrentfeature(size_t input_feature);

        //optimizer setter function
        void setOptimizer(std::shared_ptr<optimizer> newOptimizer);

        DenseLayer& add_DenseLayer(std::string name, size_t neurons);
            
        Relu& add_ReluLayer(std::string name);
           
        double train_model(const Matrix& input_matrix, const Matrix& target_matrix);

};