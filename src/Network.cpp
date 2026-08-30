#include "Network.hpp"

DenseLayer& Network::add_DenseLayer(std::string name, size_t neurons){
    if (current_feature_size== 0){
        throw std::runtime_error("Call set_input_size first.");}
    //generate, auto allocate, auto set input feature, return accessible reference
    auto layer = std::make_unique<DenseLayer>(name, current_feature_size, neurons);
    DenseLayer& ref= *layer;
    current_feature_size= layer->Weights().columns();
    layers.push_back(std::move(layer));
    return ref;
}

 Relu& Network::add_ReluLayer(std::string name){
    auto layer= std::make_unique<Relu>(name);
    Relu& ref= *layer;
    layers.push_back(std::move(layer));
    return ref;
}

void Network::setCurrentfeature(size_t input_feature){
    current_feature_size=input_feature;
}

void Network::setOptimizer(std::shared_ptr<optimizer> newOptimizer){
    active_optimizer=newOptimizer;
}

double Network::train_model(const Matrix& input_matrix, const Matrix& target){

    //FEED FORWARD, keeps creating outputs for one layer and input for another layer until last denselayer
    Matrix current_data=input_matrix;

    for (const auto& layer: layers){

        current_data=layer->feedforward(current_data); //FEED FORWARD, keeps creating outputs for one layer and input for another layer until last denselayer
    }

    //Final layer preactivation is now current_data, Time for the loss function call
    double loss=lossfunction.calculate_loss(current_data,target);

    //BACKPROPAGATION, starting from lossfunction's feedbackward, that analyzes loss gradient for each layer, starting from final denselayer
    Matrix error_gradient=lossfunction.feedbackward(target);

    for (auto it= layers.rbegin(); it != layers.rend(); ++it){ // reverse loop from end to start, where "it" is vector's internal pointer

        error_gradient= (*it)->feedbackward(error_gradient); 
    }
    //OPTIMIZER CALL
    active_optimizer->GradientDescent(layers);

    return loss;
}