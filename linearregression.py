import numpy as np
import pandas as pd

def costfun(x,y,theta):
    m=len(y) #number of training set
    total=0;
    pred=np.dot(x,theta); 
    return (1/(2*m))*np.sum(np.square(pred-y))


def gradeint_decent(x,y,theta):
    itr=1000
    learning_rate=0.1
    cost_history=np.zeros(itr)
    m=len(x)
    for i in range(itr):
        pred=np.dot(x,theta)
        theta= theta - (1/m)*learning_rate*(x.T.dot((pred-y)))
        cost=float(np.round(costfun(x,y,theta),5))
        if cost == 0.00:
            print(" θ0 = ",theta[0],"\n","θ1 = ",theta[1])
            break;

x=[1,2,3,4]
y=[7,9,11,13]
x=np.c_[np.ones((len(x),1)),x]
y=np.c_[y]
theta=np.random.randn(2,1)
gradeint_decent(x,y,theta)

#Answer is theta0=5,theta1=2
