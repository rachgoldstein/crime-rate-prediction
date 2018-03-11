#Rachel Goldstein
#COEN 140
#HW 3

import numpy as np

##################################################################
#gradient descent for linear regression:

def loss(w,x,y):
    
    a = np.dot(x,w)
    a-=y
    b = a.T
    return np.dot(b,a)


def gradient(x,y):
    alpha = 0.00001
    w_0 = np.random.normal(0,1,(96,1)) #w shape is 96 X 1
    i = y - np.dot(x,w_0)
    j = np.dot(alpha,x.T)
    w_1 = (w_0 + np.dot(j,i))

    print("initial loss is",loss(w_1,x,y))
    print("initial loss 2",loss(w_0,x,y))
    while np.absolute(loss(w_1,x, y) - loss(w_0,x, y)) >= 0.000001:
        print("loss:",loss(w_1,x,y))
        w_0 = w_1
        
        i = y - np.dot(x,w_0)
        j = np.dot(alpha,x.T)
        w_1 = (w_0 + np.dot(j,i))
    

    return w_1


def ridge_loss(w,x,y,l):
    a = np.dot(x,w)
    a-=y
    b = a.T
    c = np.dot(b,a)
    k = np.dot(w.T,w)
    k*=l
    d = c+k
    return (c + k)

def ridge_gradient(x,y,l):
    alpha = 0.00001
    w_0 = np.random.normal(0,1,(96,1)) #w shape is 96 X 1
    i = y - np.dot(x,w_0)
    i = np.dot(x.T,i)
    j = l*(w_0)
    i = i-j
    i*=alpha
    w_1 = w_0 + i


    while np.absolute(ridge_loss(w_1,x, y,l) - ridge_loss(w_0,x, y,l)) >= 0.000001:
        w_0 = w_1
        
        i = y - np.dot(x,w_0)
        i = np.dot(x.T,i)
        j = l*w_0
        i = i-j
        i*=alpha
        w_1 = w_0 + i
    
    
    
    return w_1

##################################################################
#MSE
def mse(y_new, y_true, n): #n = # instances
    sum = 0

    # y_new = y_new.T
    x = []
    x = np.subtract(y_new, y_true)
    for i in x:
        sum+=(i**2)

    sum = sum/n

    return sum

    
##################################################################
#RMSE
def rmse(y_new, y_true, n): #n = # instances

    sum = 0
    x = []
    x = np.subtract(y_new,y_true)
    for i in x:
        sum+=(i**2)
    sum = np.sqrt(sum/n)

  
    return sum
                   
##################################################################
#REGRESSION (closed)
#x=input y = output
                   
def regression(x, y, l):
    
 
    I = np.identity(96)
    
    w = np.dot(x.T,x)
    w = np.add(w,np.dot(l,I))
    w = np.linalg.inv(w)
    w = np.dot(w,(x.T))
    w = np.dot(w,y)

    #w is 96 X 1
    return w

##################################################################
#creating y predictions for closed regression (both linear and ridge)
#ignore n parameter, not used


#n = number instances

def lr_test(x, w, n):
    #print("w is",w)
    #print("x is",x)
   
    y = []
    for i in x:
        y.append(np.dot(w.T,i))
 

    return y


##################################################################
#CREATE TRAINING MATRIX (training_numpy)
training_init = []



with open('training.txt', 'r') as input_file:
    for line in input_file:
        row = line.split()
        for i in range(0,95):
            row[i] = float(row[i+1])
        row[95] = row[95].strip('\n')
        training_init.append(row)

x = np.asarray(training_init)
training_numpy = x.astype(np.float)

training_numpy[:,95] = 1


##################################################################
#CREATE Y TRAINING MATRIX (y_numpy)

y_init = []

with open('training.txt', 'r') as input_file:
    for line in input_file:
        row = line.split()
        row[0] = float(row[0])
        y_init.append(row)

y = np.asarray(y_init)
y_numpy = y.astype(np.float)

y_numpy = y_numpy[:,0]

y_numpy = y_numpy.reshape(1595,1)

##################################################################
#CREATE TEST MATRIX (test_numpy)
test_init = []



with open('test.txt', 'r') as input_file:
    for line in input_file:
        row = line.split()
        for i in range(0,95):
            row[i] = float(row[i+1])
        row[95] = row[95].strip('\n')
        test_init.append(row)

x = np.asarray(test_init)
test_numpy = x.astype(np.float)

ones = np.ones(len(test_numpy))
test_numpy[:,95] = 1



##################################################################
#CREATE Y TEST TRUE MATRIX (y_true_numpy)

y_true_init = []

with open('test.txt', 'r') as input_file:
    for line in input_file:
        row = line.split()
        row[0] = float(row[0])
        y_true_init.append(row)

y = np.asarray(y_true_init)
y_true_numpy = y.astype(np.float)

y_true_numpy = y_true_numpy[:,0]

y_true_numpy = y_true_numpy.reshape(399,1)

#print(y_true_numpy.shape)
#print("Y TEST TRUE MATRIX IS",y_true_numpy)

##################################################################
#calling linear regression functions (closed)
w = regression(training_numpy, y_numpy, 0)
#print("w shape:")
#print(w.shape)

y_new_train = lr_test(training_numpy,w, 1595)
y_new_test = lr_test(test_numpy,w, 399)
lr_closed_train_error = rmse(y_new_train,y_numpy,1595)
lr_closed_test_error = rmse(y_new_test, y_true_numpy,399)

print("lr closed train error:")
print(lr_closed_train_error)




print("lr closed test error:")
print(lr_closed_test_error)

##################################################################
#kfold cross validation


k_train_1 = training_numpy[0:319]
k_y_1 = y_numpy[0:319]

k_train_2 = training_numpy[319:638]
k_y_2 = y_numpy[319:638]

k_train_3 = training_numpy[638:957]
k_y_3 = y_numpy[638:957]

k_train_4 = training_numpy[957:1276]
k_y_4 = y_numpy[957:1276]

k_train_5 = training_numpy[1276:1596]
k_y_5 = y_numpy[1276:1596]

#round 1: train 1 2 3 4 test 5
round1_train = k_train_1
round1_train = np.append(round1_train,k_train_2, axis=0)
round1_train = np.append(round1_train,k_train_3, axis=0)
round1_train = np.append(round1_train,k_train_4, axis=0)

round1_y = k_y_1
round1_y = np.append(round1_y,k_y_2, axis=0)
round1_y = np.append(round1_y,k_y_3, axis=0)
round1_y = np.append(round1_y,k_y_4, axis=0)

round1_test_train = k_train_5
round1_test_true_y = k_y_5

#round 2: train 1 2 3 5 test 4
round2_train = k_train_1
round2_train = np.append(round2_train,k_train_2, axis=0)
round2_train = np.append(round2_train,k_train_3, axis=0)
round2_train = np.append(round2_train,k_train_5, axis=0)

round2_y = k_y_1
round2_y = np.append(round2_y,k_y_2, axis=0)
round2_y = np.append(round2_y,k_y_3, axis=0)
round2_y = np.append(round2_y,k_y_5, axis=0)

round2_test_train = k_train_4
round2_test_true_y = k_y_4

#round 3: train 1 2 4 5 test 3
round3_train = k_train_1
round3_train = np.append(round3_train,k_train_2, axis=0)
round3_train = np.append(round3_train,k_train_4, axis=0)
round3_train = np.append(round3_train,k_train_5, axis=0)

round3_y = k_y_1
round3_y = np.append(round3_y,k_y_2, axis=0)
round3_y = np.append(round3_y,k_y_4, axis=0)
round3_y = np.append(round3_y,k_y_5, axis=0)

round3_test_train = k_train_3
round3_test_true_y = k_y_3

#round 4: train 1 3 4 5 test 2
round4_train = k_train_1
round4_train = np.append(round4_train,k_train_3, axis=0)
round4_train = np.append(round4_train,k_train_4, axis=0)
round4_train = np.append(round4_train,k_train_5, axis=0)

round4_y = k_y_1
round4_y = np.append(round4_y,k_y_3, axis=0)
round4_y = np.append(round4_y,k_y_4, axis=0)
round4_y = np.append(round4_y,k_y_5, axis=0)

round4_test_train = k_train_2
round4_test_true_y = k_y_2

#round 5: train 2 3 4 5 test 1
round5_train = k_train_2
round5_train = np.append(round5_train,k_train_3, axis=0)
round5_train = np.append(round5_train,k_train_4, axis=0)
round5_train = np.append(round5_train,k_train_5, axis=0)

round5_y = k_y_2
round5_y = np.append(round5_y,k_y_3, axis=0)
round5_y = np.append(round5_y,k_y_4, axis=0)
round5_y = np.append(round5_y,k_y_5, axis=0)

round5_test_train = k_train_1
round5_test_true_y = k_y_1


lamb = 400
f = 400
minimum_error = 500

#finding optimal lambda via closed ridge regression
for i in range(10):
    w_round1 = regression(round1_train, round1_y, lamb)
    y_new_round1 = lr_test(round1_test_train, w_round1, 1)
    error_round1 = mse(y_new_round1, round1_test_true_y, 319)

    w_round2 = regression(round2_train, round2_y, lamb)
    y_new_round2 = lr_test(round2_test_train, w_round2, 1)
    error_round2 = mse(y_new_round2, round2_test_true_y, 319)

    w_round3 = regression(round3_train, round3_y, lamb)
    y_new_round3 = lr_test(round3_test_train, w_round3, 1)
    error_round3 = mse(y_new_round3, round3_test_true_y, 319)

    w_round4 = regression(round4_train, round4_y, lamb)
    y_new_round4 = lr_test(round4_test_train, w_round4, 1)
    error_round4 = mse(y_new_round4, round4_test_true_y, 319)

    w_round5 = regression(round5_train, round5_y, lamb)
    y_new_round5 = lr_test(round5_test_train, w_round5, 1)
    error_round5 = mse(y_new_round5, round5_test_true_y, 319)

    error_this_round = error_round5 + error_round4 + error_round3 + error_round2 + error_round1
    error_this_round/=5
    


    if (error_this_round < minimum_error):
        minimum_error = error_this_round
        f = lamb

    lamb = lamb/2



optimized_closed_lambda = f
print("optimized closed form lambda is",optimized_closed_lambda)



lamb_gradient = 400
f_gradient = 400
minimum_error_gradient = 500

#finding optimal lambda via gradient descent ridge regression
for i in range(10):
    w_round1 = ridge_gradient(round1_train, round1_y, lamb_gradient)
    y_new_round1 = lr_test(round1_test_train, w_round1, 1)
    error_round1 = mse(y_new_round1, round1_test_true_y, 319)
    
    w_round2 = ridge_gradient(round2_train, round2_y, lamb_gradient)
    y_new_round2 = lr_test(round2_test_train, w_round2, 1)
    error_round2 = mse(y_new_round2, round2_test_true_y, 319)
    
    w_round3 = ridge_gradient(round3_train, round3_y, lamb_gradient)
    y_new_round3 = lr_test(round3_test_train, w_round3, 1)
    error_round3 = mse(y_new_round3, round3_test_true_y, 319)
    
    w_round4 = ridge_gradient(round4_train, round4_y, lamb_gradient)
    y_new_round4 = lr_test(round4_test_train, w_round4, 1)
    error_round4 = mse(y_new_round4, round4_test_true_y, 319)
    
    w_round5 = ridge_gradient(round5_train, round5_y, lamb_gradient)
    y_new_round5 = lr_test(round5_test_train, w_round5, 1)
    error_round5 = mse(y_new_round5, round5_test_true_y, 319)
    
    error_this_round = error_round5 + error_round4 + error_round3 + error_round2 + error_round1
    error_this_round/=5
    
    
    if (error_this_round < minimum_error_gradient):
        minimum_error_gradient = error_this_round
        f_gradient = lamb_gradient
    
    lamb_gradient = lamb_gradient/2

optimized_gradient_lambda = f_gradient
print("optimized gradient descent form lambda is",optimized_gradient_lambda)



##################################################################
#closed rige regression

w_ridge = regression(training_numpy,y_numpy,optimized_closed_lambda)
y_ridge_new = lr_test(test_numpy, w_ridge, 399)
closed_ridge_error = rmse(y_ridge_new, y_true_numpy,399)

print("closed ridge test error is:")
print(closed_ridge_error)


##################################################################

#calling gradient descent function for linear regression
w_gradient = gradient(training_numpy, y_numpy)
    
y_new_train_gradient = lr_test(training_numpy,w_gradient, 1595)
y_new_test_gradient = lr_test(test_numpy,w_gradient, 399)
lr_gradient_train_error = rmse(y_new_train_gradient,y_numpy,1595)
lr_gradient_test_error = rmse(y_new_test_gradient, y_true_numpy,399)
    
print("lr gradient train error:")
print(lr_gradient_train_error)
    
    
    
print("lr gradient test error:")
print(lr_gradient_test_error)



##################################################################
#calling gradient descent function for ridge regression
w_ridge_gradient = ridge_gradient(training_numpy, y_numpy, optimized_gradient_lambda)

y_new_train_ridge_gradient = lr_test(training_numpy,w_ridge_gradient, 1595)
y_new_test_ridge_gradient = lr_test(test_numpy,w_ridge_gradient, 399)
ridge_gradient_train_error = rmse(y_new_train_ridge_gradient,y_numpy,1595)
ridge_gradient_test_error = rmse(y_new_test_ridge_gradient, y_true_numpy,399)

print("ridge gradient train error:")
print(ridge_gradient_train_error)


print("ridge gradient test error:")
print(ridge_gradient_test_error)

                   

