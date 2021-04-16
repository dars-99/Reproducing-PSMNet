# Reproducing-PSMNet
This repository contains code to calculate end-point error on Scene Flow dataset and to visualize the training and validation loss.

We have provided files that we made to calculate the end point error and visualizing the training loss. To know more please refer to this repository https://github.com/JiaRenChang/PSMNet.

Before starting to work on the code read this blog post https://medium.com/tu-delft-deep-learning-project/reproducing-pyramid-stereo-matching-an-advancement-in-disparity-image-generation-a91255ea1419

## File description
* **reproduce.py**                    - Calculates the end point error
* **finetune.py**                     - This file is same as from the original repository with additional code to store losses.
* **training_and_validation_plot.py** - Plots the training and validation loss.


## Calculating the end point error

**Step 1**:
Clone the PSMNet repository from the above link. 

**Step 2**:
Clone this repository.

**Step 3**:
Copy the files from this repository and paste it under the directory /your_path/PSMNet-master/

**Step 4**:
Specify the path to your trained or pretrained model in the reproduce.py and also to the folder containing Scene FLow test dataset.

**Step 5**:
Run the code to obtain the end point error of your model.

##  Visualzing the training and test loss
**Step 1**:
Run the finetune.py file that you copy pasted from this repository. This will save a file containing all losses that occurred during the finetuning.

**Step 2**:
Run the training_and_validation_plot.py. You will see your training and validation loss.

