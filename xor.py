import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# add bias
X_b = np.insert(X,[2],[[1],[1],[1],[1]],axis=1)

# output dataset
Y = np.array([[0],
              [1],
              [1],
              [0]])

# seed random numbers... I'm told this is "good practice"
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,4)) - 1 # 3x4 - 3 inputs being sent to 4 neurons in syn0
syn1 = 2*np.random.random((4,1)) - 1 # 3x4 - 4 inputs being sent to 1 neuron in syn1

for i in range(1000000):
    
    # forward propogation
    l0 = X_b
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    
    # error calculation
    if i%1000 == 0:
        l2_error = ((Y-l2)**2)/2
        print ("Summed Error: ", sum(l2_error),)
    
    d_l2_error = -(Y-l2)
    
    # multiply error by the slope of the sigmoid at each value in l2
    l2_delta = d_l2_error * nonlin(l2,deriv=True)
    
    l1_error = np.dot(l2_delta,syn1.T) # ATTENTION figure out how this part works
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    # update weights
    alpha = 10
    syn0 -= alpha*np.dot(l0.T,l1_delta)
    syn1 -= alpha*np.dot(l1.T,l2_delta)

print ("\nTotal Error:")
print (sum(l2_error))
print ("\nOutput:")
print (l2)
print ("\nSynapse 0 Values:")
print (syn0)
print ("\nSynapse 1 Values:")
print (syn1)
