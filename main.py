import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1],
              [0,1,0],
              [1,0,0],
              [0,0,0],
              [1,1,0]])

for row in range(8):
    biasX = np.append(X,.....)
    

# output dataset
Y = np.array([[0,0,1,1,0,1,0,1]]).T

# seed random numbers
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for i in range(1000000):
    
    # forward propogation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    
    # error calculation
    if i%1000 == 0:
        l1_error = ((Y-l1)**2)/2
        print "Summed Error: ", sum(l1_error), " -- Summed Error Excluding Zero Vector: ", sum(l1_error)-0.125
    
    d_l1_error = -(Y-l1)
    
    #multiply error by the slope of the sigmoid at each value in l1
    l1_delta = d_l1_error * nonlin(l1,True)
    
    # update weights
    eta = 1
    syn0 -= eta*np.dot(l0.T,l1_delta)

print "\nOutput:"
print l1
print "\nSynapse Values:"
print syn0
