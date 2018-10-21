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
    l1_in = np.dot(l0,syn0)
    l1_out = nonlin(l1_in)
    l2_in = np.dot(l1_out,syn1)
    l2_out = nonlin(l2_in)
    
    # error calculation
    if i%1000 == 0:
        l2_error = ((Y-l2_out)**2)/2
        print ("Summed Error: ", sum(l2_error),)
    
    d_l2_error = -(Y-l2_out)
    
    # multiply error by the slope of the sigmoid at each value in l2
    l2_delta = d_l2_error * nonlin(l2_out,deriv=True)
    
    l1_error = np.dot(l2_delta,syn1.T) # ATTENTION figure out how this part works
    
    l1_delta = l1_error * nonlin(l1_out,deriv=True)
    
    # update weights
    alpha = 10
    syn0 -= alpha*np.dot(l0.T,l1_delta)
    syn1 -= alpha*np.dot(l1_out.T,l2_delta)

print ("\nTotal Error:")
print (sum(l2_error))
print ("\nOutput:")
print (l2_out)
print ("\nSynapse 0 Values:")
print (syn0)
print ("\nSynapse 1 Values:")
print (syn1)
