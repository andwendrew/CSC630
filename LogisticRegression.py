import math
import numpy as np
from variable import Variable

class LogisticRegression():
    # Creation of a LogisticRegression object, which takes in a learning rate and a threshold of error we wish for our final fit to not exceed. 
    def __init__(self, learning_rate, epoch,) :        
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
        
        storage = 2*np.random.rand(d+1) - 1 # this array will store and update the values [m__0, m_1, ... m_d-1, b] throughout this gradient descent
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
            
            if(newcost < cost_function.evaluate(mindict)): # update the array that stores the variables that minimizes cost
                self.minarray = storage
            
            lr = lr * 0.9995 # temporary way of adjusting learning rate
            if(newcost > cost):
                lr = lr * 0.99
                
            if(iterations % 100 == 0): # verbal update every 50 iterations else too much output
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
        print("-----------------\n")
        return mindict
    
    def n_fit(self, X, y, times, learning_rate, epoch): # basically fit as many times to get minimum cost, depending on different points
        model = LogisticRegression(learning_rate, epoch)
        self.m_dict = model.fit(X, y)
        self.times = times
############################
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
############################        
        for i in range(1, times): # fit "times" # of times
            placeholder = model.fit(X, y)
            if(cost_function.evaluate(placeholder) < cost_function.evaluate(self.m_dict)): # update dictionary if new fit gives smaller cost
                self.m_dict = placeholder
                 
        print("-----------------\n")
        print("Best fit at: ")
        print(self.m_dict)
        print("With a minimum cost of: ")
        print(cost_function.evaluate(self.m_dict))
        return self.m_dict
        
    def predict_fit(self, X): # must run after a fit is called in order for self.minarray to exist
        n, d = np.shape(X)
        proj_values = []
        
        for i in range(0, n):
            proj_values.append(1/(1 + math.exp(-1*(np.dot(self.minarray[0:d], X[i]) + self.minarray[d])))) # plugging in the variables from minarray
        
        print(proj_values)
        return proj_values
    
    def predict_n_fit(self, X): # must run after a n_fit is called in order for self.m_dict to exist
        m_dict_toarray = np.array(list(self.m_dict.values()))
        n, d = np.shape(X)
        proj_values = []
        
        for i in range(0, n):
            proj_values.append(1/(1 + math.exp(-1*(np.dot(m_dict_toarray[0:d], X[i]) + m_dict_toarray[d])))) # plugging in the variables from minarray
        
        print(proj_values)
        return proj_values
            
model = LogisticRegression(0.001, 5000)
#model.fit([[0, -2], [1,0], [3, 1], [2, 5], [1.5, 6]], [0,0,1,1,1]) 
model.n_fit([[0, -2], [1,0], [3, 1], [2, 5], [1.5, 6]], [0,0,1,1,1], 5, 0.001, 5000)  
model.predict_n_fit([[-1, -3], [2,4]])        
        
