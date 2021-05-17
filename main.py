import numpy as np
import sys
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import utilities


def question1():
    # input data 
    x=np.array([[50, 55, 70, 80, 130, 150, 155, 160], [1,1,1,1,1,1,1,1]]).T
    y=np.array([1,1,1,1,-1,-1,-1,-1])

    # apply online training on data and print it's report
    w1,steps1,epochs1,chng_q1_on= utilities.online_training(x,y)
    print('Results of applying Online Training algorithm on question#1:',w1)
    print('weight updates:',len(steps1))
    print("Epochs:", epochs1)
    print("-----------")
    # apply Batch Perceptron on data and print it's report
    w2,steps2,epochs2,chng_q1_per=utilities.perceptron(x,y)
    print('Results of applying Batch Perceptron algorithm on question#1:',w2)
    print('weight updates:',len(steps2))
    print("Epochs:", epochs2)
    return chng_q1_on,chng_q1_per




def question2():
    # input data 
    x=np.array([[0, 255, 0, 0, 255, 0, 255, 255],[0, 0, 255, 0, 255, 255, 0, 255],[0, 0, 0, 255, 0, 255, 255, 255],[1, 1, 1, 1, 1, 1, 1, 1]]).T
    y=np.array([1, 1, 1, -1, 1, -1, -1, 1])
    # apply online training on data and print it's report
    w1,steps1,epochs1,chng_q2_on= utilities.online_training(x,y)
    print('Results of applying Online Training algorithm on question#2:',w1)
    print('weight updates:',len(steps1))
    print("Epochs:", epochs1)
    print("-----------")
    # apply Batch Perceptron on data and print it's report
    w2,steps2,epochs2,chng_q2_per=utilities.perceptron(x,y)
    print('Results of applying Batch Perceptron algorithm on question#2:',w2)
    print('weight updates:',len(steps2))
    print("Epochs:", epochs2)
    return chng_q2_on,chng_q2_per



def classification_of_data():
    # generate data from calssification
    x,y=utilities.classification()
    # split data to train and test
    x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.25, train_size=0.75)
    # apply online training on data and print it's report
    w1,steps1,epochs1,chng_class_on= utilities.online_training(x_train,y_train)
    print('Results of applying Online Training algorithm on data after making classification:',w1)
    print('weight updates:',len(steps1))
    print("Epochs:", epochs1)
    print("Model accuracy:", utilities.Accuracy(x_test,y_test,w1),"%")
    print("-----------")
    # apply Batch Perceptron on data and print it's report
    w2,steps2,epochs2,chng_class_per=utilities.perceptron(x_train,y_train)
    print('Results of applying Batch Perceptron algorithm on data after making classification:',w2)
    print('weight updates:',len(steps2))
    print("Epochs:", epochs2)
    print("Model accuracy:", utilities.Accuracy(x_test,y_test,w2),"%")
    return x_train,y_train,w1,w2,chng_class_on,chng_class_per


def Comparison(title1,d1,title2,d2):
    # Comparison plot
    figure, axis = plt.subplots(1, 2)

    # plot first data
    axis[0].plot(d1)
    axis[0].set_title(title1)

    # plot first data
    axis[1].plot(d2)
    axis[1].set_title(title2)

    plt.show()

def Model_Visualization(title,x,y,w):
    # calculate slop
    slop=-(w[0]/w[1])
    # calculate max and min to be the line limits
    maxX=max(x[:,0])
    minX=min(x[:,0])
    # plot original data
    plt.title(title)
    plt.scatter(x[:,0],x[:,1],marker='o',c=y,s=25,edgecolors='k')
    # plot model (line)
    plt.plot([minX,maxX],[slop*minX,slop*maxX])
    plt.show()

if __name__== '__main__':
    print("==========================Q1==============================")
    q1_on,q1_pre=question1()
    Comparison("Online Training Problem 1",q1_on,"Batch Perceptron Problem 1",q1_pre)

    print("==========================Q2==============================")
    q2_on,q2_pre=question2()
    Comparison("Online Training Problem 2",q2_on,"Batch Perceptron Problem 2",q2_pre)

    print("==========================classification==============================")
    x,y,w1,w2,class_on,class_pre=classification_of_data()
    Comparison("Online Training classification",class_on,"Batch Perceptron classification",class_pre)
    Model_Visualization("Online Training Model visualization",x,y,w1)
    Model_Visualization("Batch Perceptron Model visualization",x,y,w2)
