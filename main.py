import sys
import os 

#need these to look for .pyd file and let python sense it
build_path=os.path.abspath("build") #translates build folder name into standard os path (windows) eg, C: \Users\fnag...\build
sys.path.append(build_path) #takes that build path and appends it to sys.path (list of strings of folders),
                            #so that interpreter can detect build folder for the .pyd cpp compiled binary file

import Custom_ML

def main():

    epoch=0
    target_loss=0.01
    current_loss=1 #to run the training loop 

    #bunch of random doubles for input, 4 batches, 2 features each
    Input=Custom_ML.Matrix(4,2,[
        0.0, 0.0,  # Inputs for 0 OR 0 -> aligns with row 1 of targets
        0.0, 1.0,  # Inputs for 0 OR 1 -> aligns with row 2 of targets
        1.0, 0.0,  # Inputs for 1 OR 0 -> aligns with row 3 of targets
        1.0, 1.0   # Inputs for 1 OR 1 -> aligns with row 4 of targets
    ])

    #Targets: 4 samples, 2 classes (One-Hot Encoded)
    #[1.0, 0.0] represents class 0 (False)
    #[0.0, 1.0] represents class 1 (True)
    targets = Custom_ML.Matrix(4, 2, [
        1.0, 0.0,  # 0 XOR 0 = 0
        0.0, 1.0,  # 0 XOR 1 = 1
        0.0, 1.0,  # 1 XOR 0 = 1
        1.0, 0.0   # 1 XOR 1 = 0
    ])
    
    network=Custom_ML.Network() #create network object

    SGD= Custom_ML.SGD(0.05)

    network.setOptimizer(SGD)

    network.setCurrentfeature(2)

    #automatically pushed into network vector
    network.addDenseLayer("dense1", 8) #creates W(2*4), 8 is neurons
    network.addReluLayer("relu1")
    network.addDenseLayer("dense2", 2)


    #loop to train model, return loss after each gradient descent
    while current_loss>target_loss:

        current_loss=network.trainModel(Input,targets)

        #print every 50 epochs
        if epoch%50==0:
            print(f"Epoch: {epoch} | Loss: {current_loss:.6f}")
        epoch+=1

        #added an emergengy break
        if epoch>15000:
            print("Hit max epochs. Stopping.")
            break

    #final result
    print(f"Training Complete at Epoch {epoch} with Final Loss: {current_loss:.6f}")

if __name__ == "__main__":
    main()

