#include "optimizer.hpp"

void SGD::GradientDescent(std::vector<std::unique_ptr<Layer>>&Layers){
    for (auto& layer: Layers){      
        
        auto saveWB= layer->getWeight_Bias();// saves the vector of W and B addresses; vector={&W, &B}
        auto saveGrd= layer->get_gradients();// saves the vector of dW and dB addresses; vector={&dW, &dB}

        if (saveWB.empty()){continue;}//checks for Relu and skips its iteration

        //W= W-dW*learning_rate
        *(saveWB[0])= *(saveWB[0])- (*(saveGrd[0])*learning_rate); // overwrites the new Weight for each denselayer, W(new)=W(old)-dW*learning rate
        
        //B= B-dB*learning_rate
        *(saveWB[1])= *(saveWB[1])- (*(saveGrd[1])*learning_rate); // overwrites new Bias for each denselayer, B(new)=B(old)-dB*learning_rate

    }
}