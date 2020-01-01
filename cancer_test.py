#!/usr/bin/env python
# coding: utf-8

# In[231]:


import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
tf.disable_v2_behavior() 
import pandas as pd
from sklearn import preprocessing
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[232]:


def create_placeholder(Xlen, Ylen):
    X = tf.placeholder(tf.float64, shape=[Xlen,None], name = "X")
    Y = tf.placeholder(tf.float64, shape = [Ylen,None],name = "Y")
    
    return X,Y


# In[233]:


def intialize_parameters():
    W1 = tf.get_variable("W1", [25,30],initializer = tf.random_normal_initializer, dtype = "float64")
    B1 = tf.get_variable("B1", [25,1],initializer = tf.zeros_initializer(), dtype = "float64")
   
    W2 = tf.get_variable("W2", [2,25],initializer = tf.random_normal_initializer, dtype = "float64")
    B2 = tf.get_variable("B2", [2,1],initializer = tf.zeros_initializer(), dtype = "float64")
    
    parameters = {"W1": W1,
                  "W2": W2,
                  "B1": B1,
                  "B2" :B2}
    
    return parameters


# In[234]:


def one_hot_matrix(Y, C):

    C = tf.constant(C, name='C')
    Y = tf.one_hot(indices=Y, depth=C, axis=0)
    
    
    sess = tf.Session()
    Y_encoded = sess.run(Y)
    sess.close()
    
    return Y_encoded
    


# In[282]:


def Cost_fn(Y_h, Y, parameter):
    
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z2, labels=Y))
    loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(Y_h), labels= tf.transpose(Y))) +
    0.01*tf.nn.l2_loss(parameter["W1"]) +
    0.01*tf.nn.l2_loss(parameter["B1"]) +
    0.01*tf.nn.l2_loss(parameter["W2"]) +
    0.01*tf.nn.l2_loss(parameter["B2"]))
    
    return loss


# In[283]:


def Forward_propogation(X, parameters):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    B1 = parameters["B1"]
    B2 = parameters["B2"]
    
    #print("w1 * X =", W1.shape, X.shape)
    mul1= tf.matmul(W1,X)
    #print("mul1 shape = ", mul1.shape)
    Z1 = tf.add(mul1 , B1)
    #print("z1 -", Z1.shape)
    A1 = tf.nn.relu(Z1)
    
    #print("W2 * A1 = ", W2.shape, A1.shape)
    mul2= tf.matmul(W2,A1)
    #print("mul2 shape = ", mul2.shape)
    
    Z2 = tf.add(mul2, B2)
    A2 = tf.nn.softmax(Z2)
    
    
    return A2
    


# In[287]:


def model(X_train, X_test, Y_train, Y_test, learning_rate =0.1, print_cost =True, num_iters = 10000):
    
    
    ops.reset_default_graph() 
    costList = []
    Rows, Cols = X_train.shape
    X, Y = create_placeholder(X_train.shape[0], Y_train.shape[0])
    #print("y shape = " Y.shape)
    
    parameters = intialize_parameters()
    
    Y_h = Forward_propogation(X, parameters)
    
    cost = Cost_fn(Y_h, Y, parameters)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(cost)
    
    init = tf.initialize_all_variables()
    
    with tf.Session() as ss:
        
        ss.run(init)
        
        for i in range(num_iters):
            _, costV = ss.run([optimizer,cost], feed_dict={X:X_train,Y:Y_train})
            print ("Cost after", i," = ", costV )
            costList.append(costV)
        
    
                   
    #plt.plot(np.squeeze(costList))
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per tens)')
    #plt.title("Learning rate =" + str(learning_rate))
    #plt.show()

    
    
    #parameters = ss.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Y_h), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
    return parameters
    
    


# In[288]:


df = pd.read_csv("data.csv")
df.drop('id', inplace = True, axis =1)
df.drop('Unnamed: 32',axis=1,inplace=True)

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
train, test = train_test_split(df, test_size=0.3)

X_train   = train.drop('diagnosis', axis = 1, inplace = False)
scaler    = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(X_train)
X_train   = pd.DataFrame(scaled_df)

X_test = test.drop('diagnosis', axis = 1, inplace = False)
scaler    = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(X_test)
X_test   = pd.DataFrame(scaled_df)


Y_train = train['diagnosis']
Y_test = test['diagnosis']

X_train = np.array(np.transpose(X_train))
X_test =  np.array(np.transpose(X_test))
Y_train = np.array(np.transpose(Y_train))
Y_test =  np.array(np.transpose(Y_test))



Y_train = one_hot_matrix(Y_train,2)
Y_test  = one_hot_matrix(Y_test,2)


print("X_train shape : ", X_train.shape)
print("Y_train shape : ", Y_train.shape)
print("X_test shape : ", X_test.shape)
print("Y_test shape : ", Y_test.shape)


# In[289]:


tf.reset_default_graph()
parameters = model(X_train, X_test, Y_train, Y_test)


# In[219]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




