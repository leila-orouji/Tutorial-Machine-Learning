# Gradient descent for linear regression
# yhat = wx+b
# loss = (y-yhat)**2 / N
import numpy as np

# Initial some parameters
x= np.random.randn(10,1)
y = 5*x + np.random.rand()

# parameters
w = 0.0
b = 0.0

# Hyperparameter
learning_rate = 0.01

#Create grafient descent function
def descend(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    for xi , yi in zip(x,y):
        # loss = (yi-(wxi+b))**2
        dldw += -2*xi *(yi-(w*xi+b))
        dldb += -2* (yi-(w*xi+b))
    # Make an update to the w and b parameter
    w = w - learning_rate * (1/N) * dldw
    b = b - learning_rate * (1/N) * dldb
    return w, b

#Iteratively make updates
loss = 0
for epoch in range(600):
    w, b = descend(x, y, w, b, learning_rate)
    yhat = w*x+b
    loss = np.divide(np.sum((y - yhat)**2 , axis=0), x.shape[0])
    print(f"{epoch} loss is {loss}, parameters w={w}, b={b}")
    # Run gradient descent
    pass