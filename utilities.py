import numpy as np
import sys
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification


def online_training(X, Y, learning_rate=1):
    # initialize the data
    n=X.shape[1]
    data_length= len(X)
    w= np.random.rand(n)*2 -1
    e=sys.float_info.epsilon
    delta=np.ones(n)
    steps=[]
    chng_in_w=[]
    epochs = 0
    while np.linalg.norm(delta,1)> e:
        delta=np.zeros(n)
        for i in range(data_length):
            u=w.dot(X[i])
            if Y[i] * u <= 0:
                delta= delta - (Y[i] * X[i])
                delta= delta/data_length
                w= w - (learning_rate*delta)
                steps.append(w)
        epochs += 1
        chng_in_w.append(np.linalg.norm(delta,1))
    return w, steps,epochs,chng_in_w



def perceptron(X,Y,learning_rate=1):
    # initialize the data
    n=X.shape[1]
    data_length= len(X)
    w= np.random.rand(n)*2 -1
    e=sys.float_info.epsilon
    delta=np.ones(n)
    steps=[]
    chng_in_w=[]
    epochs=0
    while np.linalg.norm(delta,1)> e:
        delta=np.zeros(n)
        for i in range(data_length):
            u=w.dot(X[i])
            if Y[i] * u <= 0:
                delta= delta - (Y[i] * X[i])
        delta= delta/data_length
        w= w - (learning_rate*delta)
        steps.append(w)
        epochs +=1
        chng_in_w.append(np.linalg.norm(delta,1))        

    return w, steps,epochs,chng_in_w




def classification():
    # generate data with classification method
    x, y = make_classification(25, n_features=2,n_redundant=0, n_informative=1,n_clusters_per_class=1)
    mask=y==0
    y[mask]=-1
    return x,y




def Accuracy(x_test,y_test,w):
    # calculate y predicted
    y_predicted=[]
    for i in x_test:
        y_predicted.append( np.sign(w.dot(i)))

    # calculate accuracy score
    score=accuracy_score(y_test,y_predicted)

    return score*100


