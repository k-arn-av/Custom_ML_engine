#include "Network.hpp"

void Network::setOptimizer(std::unique_ptr<optimizer> newOptimizer){
    active_optimizer=std::move(newOptimizer);
}

void Network::train_model(const Matrix& input_matrix, const Matrix& target){

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

}