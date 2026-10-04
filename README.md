# MNIST Neural Network From Scratch

A simple **two-layer neural network built completely from scratch using NumPy** to classify handwritten digits from the MNIST dataset.

The project implements the core components of a neural network manually — including **forward propagation, backpropagation, ReLU, Softmax, one-hot encoding, gradient descent, and parameter updates**.

> Built to understand what happens inside a neural network rather than relying on high-level deep learning frameworks.

---

## Overview

This project uses the **MNIST handwritten digit dataset**, where each image is a `28 × 28` grayscale image representing one of the digits from `0` to `9`.

Each image is flattened into **784 input features** and passed through a small neural network:

```text
784 Input Features
       │
       ▼
┌─────────────────┐
│ Hidden Layer    │
│ 10 Neurons      │
│ ReLU            │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Output Layer    │
│ 10 Neurons      │
│ Softmax         │
└─────────────────┘
       │
       ▼
  Digit Prediction
   0 ───────── 9
```

The entire training process is implemented using matrix operations with NumPy.

---

## Project Goals

The main goal of this project was to understand the fundamentals of neural networks by implementing them without using frameworks such as:

- PyTorch
- TensorFlow
- Keras
- Scikit-learn

The implementation covers:

- Data preprocessing
- Feature normalization
- Neural network initialization
- Forward propagation
- ReLU activation
- Softmax activation
- One-hot encoding
- Backpropagation
- Gradient calculation
- Gradient descent
- Model prediction
- Accuracy evaluation

---

## Dataset

The project uses the **Kaggle Digit Recognizer / MNIST dataset**.

The training dataset contains:

- **42,000 samples**
- **784 pixel features**
- **1 label column**
- `28 × 28` grayscale images
- **10 classes**, representing digits `0–9`

Dataset structure:

```text
label | pixel0 | pixel1 | ... | pixel783
  1   |   0    |   0    | ... |    0
  0   |   0    |   0    | ... |    0
  4   |   0    |   0    | ... |    0
```

The pixel values are normalized from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
X = X / 255.
```

---

## Model Architecture

The network consists of two trainable layers.

### Input Layer

```text
784 neurons
```

Each neuron corresponds to one pixel of the `28 × 28` image.

### Hidden Layer

```text
10 neurons
ReLU activation
```

### Output Layer

```text
10 neurons
Softmax activation
```

Each output neuron represents one digit:

```text
0 1 2 3 4 5 6 7 8 9
```

### Architecture

```text
Input
784
 │
 ▼
Dense Layer
10 neurons
ReLU
 │
 ▼
Dense Layer
10 neurons
Softmax
 │
 ▼
Prediction
0–9
```

---

## Forward Propagation

The first layer calculates:

```text
Z₁ = W₁X + b₁
```

ReLU is then applied:

```text
A₁ = ReLU(Z₁)
```

The second layer calculates:

```text
Z₂ = W₂A₁ + b₂
```

Finally, Softmax converts the output into class probabilities:

```text
A₂ = Softmax(Z₂)
```

Implementation:

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2
```

---

## Activation Functions

### ReLU

The hidden layer uses the Rectified Linear Unit:

```text
ReLU(x) = max(0, x)
```

Implementation:

```python
def ReLU(Z):
    return np.maximum(Z, 0)
```

### Softmax

The output layer uses Softmax to produce class probabilities:

```python
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
```

---

## Backpropagation

The model manually calculates gradients during backpropagation.

For the output layer:

```text
dZ₂ = A₂ - Y
```

Then:

```text
dW₂ = (1/m) dZ₂ A₁ᵀ
```

The gradient is propagated back through the hidden layer:

```text
dZ₁ = W₂ᵀ dZ₂ × ReLU'(Z₁)
```

Finally:

```text
dW₁ = (1/m) dZ₁ Xᵀ
```

These gradients are used to update the model parameters.

---

## Gradient Descent

The parameters are updated using:

```text
W := W - α dW
b := b - α db
```

where:

- `W` = model weights
- `b` = bias
- `dW` = gradient of weights
- `db` = gradient of bias
- `α` = learning rate

Training configuration used in the notebook:

```python
alpha = 0.10
iterations = 500
```

---

## Training

Training is performed using the custom `gradient_descent()` function:

```python
W1, b1, W2, b2 = gradient_descent(
    X_train,
    Y_train,
    0.10,
    500
)
```

The model prints training accuracy every 10 iterations.

The training accuracy improved from approximately:

```text
9.7%
```

at the beginning to approximately:

```text
81.0%
```

by iteration 490.

---

## Validation / Development Set

The dataset is shuffled and split into:

```text
1,000 samples → development set
41,000 samples → training set
```

The development set is normalized in the same way as the training data.

Final development-set accuracy reported by the notebook:

```text
80.3%
```

This gives the model a reasonable level of performance despite being a very small neural network implemented from scratch.

---

## Example Predictions

The notebook also includes a helper function for visualizing individual predictions.

Example results included:

```text
Prediction: 4
Label:      4

Prediction: 8
Label:      2

Prediction: 0
Label:      7

Prediction: 6
Label:      6
```

This demonstrates both correct and incorrect predictions.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computation and neural network implementation |
| Pandas | Loading and handling the dataset |
| Matplotlib | Visualizing handwritten digits |
| Seaborn | Data visualization |
| Jupyter Notebook | Development environment |
| Kaggle | Dataset / notebook environment |

---

## Project Structure

```text
mnist-neural-network-from-scratch/
│
├── mnist-neural-network-from-scratch.ipynb
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd mnist-neural-network-from-scratch
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
mnist-neural-network-from-scratch.ipynb
```

### 5. Dataset

The notebook currently expects the Kaggle Digit Recognizer dataset at:

```text
/kaggle/input/competitions/digit-recognizer/train.csv
```

If running locally, download the dataset and update the CSV path in the notebook.

---

## What I Learned

This project helped me understand the mechanics behind neural networks at a lower level.

Key concepts explored:

- How neural network parameters are initialized
- How matrix multiplication is used in neural networks
- How activations transform information
- How Softmax produces class probabilities
- How one-hot encoding represents classes
- How errors propagate backwards
- How gradients are calculated
- How gradient descent learns model parameters
- How image data can be represented numerically
- How a neural network improves through iterative optimization

Instead of treating a neural network as a black box, this implementation exposes the main mathematical operations involved in training one.

---

## Limitations

This implementation is intentionally simple and educational.

Some possible improvements include:

- More hidden layers
- More neurons
- Better weight initialization
- Mini-batch gradient descent
- A numerically stable Softmax implementation
- Proper loss tracking
- Learning-rate scheduling
- Regularization
- Confusion matrix analysis
- Hyperparameter tuning
- CNN architecture for improved image recognition

A Convolutional Neural Network would likely perform significantly better on MNIST because it can exploit the spatial structure of images.

---

## Future Improvements

Possible next steps:

```text
Current
  │
  ├── Two-layer neural network
  ├── ReLU
  ├── Softmax
  └── Gradient descent
        │
        ▼
Next
  │
  ├── Mini-batch training
  ├── Better initialization
  ├── Loss visualization
  ├── Hyperparameter tuning
  ├── Multiple hidden layers
  └── CNN implementation
```

---

## Key Result

**Development Accuracy: ~80.3%**

**Training Accuracy: ~81.0% at iteration 490**

The main achievement of the project is not just the accuracy, but implementing the neural network training pipeline manually using NumPy.

---

## License

This project is intended for educational and learning purposes.
