import numpy as np
x = np.array(([2,9], [1,5], [3,6]))
y = np.array(([92], [85], [95]))
x = x/np.amax(x, axis=0)
y = y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch = 20000
l_rate = 0.1
ip_laywr_neurons = 2
hid_laywr_neurons = 3
op_laywr_neurons = 1 

wh = np.random.uniform(size = (ip_laywr_neurons, hid_laywr_neurons))
bh = np.random.uniform(size = (1, hid_laywr_neurons))
wout = np.random.uniform(size = (hid_laywr_neurons, op_laywr_neurons))
bout = np.random.uniform(size = (1, op_laywr_neurons))

for _ in range(epoch):
    
    # Forward
    hidden = sigmoid(np.dot(x, wh) + bh)
    output = sigmoid(np.dot(hidden, wout) + bout)
    
    # Backward
    d_output = (y - output) * derivatives_sigmoid(output)
    d_hidden = d_output.dot(wout.T) * derivatives_sigmoid(hidden)
    
    # Update
    wout += hidden.T.dot(d_output) * l_rate
    wh += x.T.dot(d_hidden) * l_rate
    bout += np.sum(d_output, keepdims=True) * l_rate
    bh += np.sum(d_hidden, keepdims=True) * l_rate


print("Imput:\n", str(x))
print("Actual Output:\n ", str(y))
print("Predicated Output:\n", output)