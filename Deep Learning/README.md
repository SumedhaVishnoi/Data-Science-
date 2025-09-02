 deep learning is a subset of machine learning inspired by human brain 
  uses artificial neural networls with multiple layer 
this shines in images, text, speech,video 

# Structure of neural networks 
input layer 
hidden layers 
output layers 

# Activation Functions 
sigmoid- squashes values between 0 and 1 and are used in binary classification 
ReLU - rectified linear unit - are used in hidden layers 
softmax- converts outputs into probabilities  

# Training a neural network 
(a) forward propapgation - Input data → passes through layers → get prediction.
(b) Loss Function
Measures error (e.g., MSE for regression, Cross-Entropy for classification).
(c) Backpropagation
Network adjusts weights using Gradient Descent to minimize loss.
(d) Repeat (epochs)
Keep training until error is minimized.



## CNN - CONVOLUTIONAL NEURAL NETWORKS 
used for image data 
- it detect patterns and features automatically 
-works on spatial data 
 
# Key components 
1. convolution layer: applies filters that can scan the image 
detect features 
2. pooling layer : reduces size of feature maps 
3. flattening: converts pooled feature maps into single vector 
4. Fully connected layers : classification of the image 

# workflow :
Input image → 64x64 pixels.

Convolution + ReLU (feature extraction).

Pooling (downsampling).

Repeat convolution + pooling (deeper features).

Flatten.

Fully Connected Layer.

Output layer (Softmax for probabilities).





## RNN: RECURRENT NEURAL NETWORKS 
 used for sequences 
 - text/ natural language processing 
 - time series 
 - speech recognition

 uses past information and use it for current predictions 

 # workflow 

