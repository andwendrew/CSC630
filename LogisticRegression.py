import math
import numpy as np
from variable import Variable

class LogisticRegression():
    # Creation of a LogisticRegression object, which takes in a learning rate and a threshold of error we wish for our final fit to not exceed. 
    def __init__(self, learning_rate, epoch) :        
        self.lr = learning_rate       
        self.ep = epoch
####
    def fit(self, X, y):
        n, d = np.shape(X) # n counts the number of data points in X, and d is dimension of the data points in X
        
        mvars = [] # create list of d variables for the d-dimensional [m_0, m_1, ... , m_d-1] in the next few lines. This is all in preparation
                   # for the creation of a cost function, which given a dataset to fit, takes in all variables m_i and b
        for i in range(0, d): 
            mvars.append(Variable(name = f'm_{i}')) # thanks to william yue '22 here for telling me to convert into f-string to fix my code
        bvar = Variable(name = 'b')
        
        y_hats = [] # here we create a list of n y_hat values which represent the predicted outputs of the n data points under the fit, where each of
                   # the y_hat depends on the m_i's and b
        
        for i in range(0, n):
            y_hats.append(1/(1 + Variable.exp(-1*(np.dot(mvars, X[i]) + bvar))))
        
        # here's the actual cost function we will be performing gradient descent on, in terms of the variables m_i and b
        cost_function = sum([(-1 * y[i] * Variable.log(y_hats[i]) + (y[i] - 1) * Variable.log(1 - y_hats[i])) for i in range(0, n)])
        
        # Now, let's perform the gradient descent on the cost function, which we wish to minimize!
        
        storage = np.random.rand(d+1)-0.5 # this array will store and update the values [m__0, m_1, ... m_d-1, b] throughout this gradient descent
        iterations = 0
        self.minarray = storage
        lr = self.lr
        
        while (iterations <= self.ep):
            dictionary = {} # creating dictionary from storage at each point in the process in order to put into cost_function.gradient()
            for i in range(0, d):
                dictionary.update({f'm_{i}': storage[i]})
            dictionary.update({'b': storage[d]})
            
            cost = cost_function.evaluate(dictionary) # cost function 
            
            moving_direction = cost_function.gradient(dictionary) # gradient
            
            # now let's move away from the direction of the gradient, keeipng in mind that storage[0] to storage[d-1] store the m variables and that
            # storage[d] stores b
            
            storage = storage - lr * moving_direction # update variables to move away from the direction of the gradient
            
            newdict = {} # dictionary after update
            for i in range(0, d):
                newdict.update({f'm_{i}': storage[i]})
            newdict.update({'b': storage[d]})
            
            newcost = cost_function.evaluate(newdict) # cost function after update
            
            mindict = {} # create corresponding dictionary to the minarray array
            for i in range(0, d):
                mindict.update({f'm_{i}': self.minarray[i]})
            mindict.update({'b': self.minarray[d]})
            
            if(newcost < cost_function.evaluate(mindict)): # if after update, cost decrease past existing minimum, then update the array that stores the variables that minimizes cost
                self.minarray = storage
            
            if(newcost < cost): 
                lr = self.lr
            else: # adjusting lr if cost keeps going up
                lr = lr * 0.95
                
            print("Cost is: " + str(newcost))
            print(newdict)
        
            iterations = iterations + 1 
            
        mindict = {} # create corresponding dictionary to the minarray array after everything finishes running
        for i in range(0, d):
            mindict.update({f'm_{i}': self.minarray[i]})
        mindict.update({'b': self.minarray[d]})
        
        mincost = cost_function.evaluate(mindict) # minimum possible cost
        print("-----------------\n")
        print("The minimum value of the cost function is " + str(mincost) + " occurs when")
        print(mindict)
        
    def predict(self, X):
        n, d = np.shape(X)
        proj_values = []
        
        for i in range(0, n):
            proj_values.append(1/(1 + math.exp(-1*(np.dot(self.minarray[0:d], X[i]) + self.minarray[d]))))
        
        return proj_values
            
model = LogisticRegression(0.00001, 50000)
#model.fit([[1], [1.5], [1.5], [3], [4], [4], [4.5], [6]], [0, 0, 0, 0, 0, 1, 1, 1])
model.fit([[0,0],[2,1],[3,5],[5,6],[5,7],[6,9],[7,11]],[0,0,0,0,0,1,1])
#model.fit([[1,1,1],[-1,-1,-1],[-2,-2,-2],[-2,-2,0],[0,-2,-2],[3,0,4]],[1,0,0,0,0,1])

print(model.predict([[0, 0], [0, 6], [1.5, 2], [6.5, 3], [4, 3]]))
            
        