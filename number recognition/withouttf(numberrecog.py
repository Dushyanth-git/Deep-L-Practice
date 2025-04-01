import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

mnist = mnist.load_data()
(x_train,y_train),(x_test,y_test) = mnist

x_train,x_test = x_train/255.0,x_test/255.0

x_train = x_train.reshape(len(x_train),784)
x_test = x_test.reshape(len(x_test),784)

def one_hot_encoding(y,num=10):
    encode = np.zeros((len(y),num))
    encode[np.arange(len(y)),y]= 1
    return encode
y_train = one_hot_encoding(y_train)
y_test = one_hot_encoding(y_test)

def init_weights():
    np.random.seed(42)
    w1 = np.random.randn(784,128)
    b1 = np.zeros((1,128))
    w2 = np.random.randn(128,10)
    b2 = np.zeros((1,10))
    return w1,b1,w2,b2
w1,b1,w2,b2 = init_weights()

def relu(x):
    return np.maximum(0,x)
def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    expz = np.exp(z-np.max(z,axis=1,keepdims = True))
    return expz/np.sum(expz,axis=1,keepdims = True)

def forward_propagation(x,w1,b1,w2,b2):
    z1 = np.dot(x,w1)+b1
    a1 = relu(z1)
    z2 = np.dot(a1,w2)+b2
    a2 = softmax(z2)
    return z1,a1,z2,a2

def cross_entropy_loss(y_true,y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true*np.log(y_pred + 1e-8 ))/m

def backward_propagation(X, y_true, Z1, A1, A2, w1, w2):
    m = X.shape[0]  
    dZ2 = A2 - y_true 
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, w2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def update_weights(w1, b1, w2, b2, dW1, db1, dW2, db2, learning_rate):
    w1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2
epochs = 7
learning_rate = 0.01


# Training the model
for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_propagation(x_train, w1, b1, w2, b2)
    
    loss = cross_entropy_loss(y_train, A2)
    
  
    dW1, db1, dW2, db2 = backward_propagation(x_train, y_train, Z1, A1, A2, w1, w2)

    w1, b1, w2, b2 = update_weights(w1, b1, w2, b2, dW1, db1, dW2, db2, learning_rate)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

