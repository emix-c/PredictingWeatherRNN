# PredictingWeatherRNN

## Overview
This project explores the usage of a Recurrent Neural Network (RNN) to predict hourly temperatures in Dallas, Texas using historical weather data from 2012-2017. 
The model uses humidity, pressure, and temperature as inputs and is trained to output the next temperature value in the sequence.
The RNN was created from scratch and the model was evaluated using regression metrics such as MSE, MAE and R². 

We found that the RNN is effective at learning temporal patterns and forecasting the temperature. 

### Preprocessing Data
We used a publicly available dataset from Kaggle and focused on Dallas. 
We did the following: 
1. Merged humidity, pressure and temperature CSV files.
2. Interpolated missing values using the average of neighboring entries.
3. Applied min-max normalization to reduce scale differences.
The resulting cleaned dataset had 45,000+ entries.
We also analyzed the frequency distributions of all three attributes to understand their ranges and patterns.

### Sequence Construction
The cleaned dataset is broken into sliding window sequences. 
Each sequence has:
- length N = 10 (default)
- previous N timesteps of temperature, humidity and pressure
Each sequence is one training instance for the RNN.

### Train/Test Split
Sequences are divided into 20% train/80% test. 
The split is adjustable via parameters. 

### Building and Training the RNN
#### Basics
This model has the key components of all artificial neural networks: 
- Input and output layers
- Weighted connections + bias terms
- Activation functions (sigmoid, tanh, ReLU)
- Forward propagation to compute outputs
- Loss functions (MSE, MAE)
- Gradient descent optimization
However, RNNs in particular also have a recurrent hidden layer with a feedback loop that allows the model to retain memory across timesteps.

#### Hyperparameters
All hyperparameters can be customized including; 
- Activation function: sigmoid, tanh, ReLU
- Loss function: MSE or MAE
- Learning rate η
- Epoch count
- Time sequence length
- Hidden layer size

#### Forward and Backward Propagation
This is for training the RNN. 

##### Forward Propagation
For each time step: 
  - Read the current input tuple
  - Update the hidden state using the recurrent layer
  - Produce a predicted temperature for the next time step

##### Backward Propagation
We unroll the network through time and recompute weights and update parameters accordingly. 
For each time step (from last to first): 
  - Compute the gradient of the loss given the output at current time step
  - Backpropagate the error through the activation function to obtain gradients for the output layer and recurrent hidden layer
  - Calculate the gradient for the hidden state
  - Compute gradients for all weight matrices and biases
  - Accumulate gradients across time
  - Update all weights and biases using gradient descent and scaled by the learning rate η

### Model Evaluation
We use R² to evaluate predictive performance:
- R² = 1 → perfect prediction
- R² = 0 → model explains none of the variance
We also examined MSE and MAE across various hyperparameter combinations.

## Conclusion 
Our experiments show that the best-performing model is one with the following parameters: 
- Learning Rate: 0.015
- Activation Function: Sigmoid
- Hidden Layer Size: 20
This model acheived the lowest MSE and highest predictive accuracy. 

Refer to `RNN_Weather_Paper.pdf` for more details.
