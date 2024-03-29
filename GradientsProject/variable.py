import numpy as np
import math

class Variable():
########
    def __init__(self, name=None, evaluate=None, gradient=None) :
        if evaluate == None:
            self.evaluate = lambda values: values[self.name]
        else:
            self.evaluate = evaluate 
        if gradient == None: 
            self.gradient = lambda values: np.array(list(map((lambda x: int(x == self.name)), sorted(list(values.keys()))))) 
        else:
            self.gradient = gradient            
        if name != None:
            self.name = name          # its key in the evaluation dictionary
########
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) + other,
                            gradient = lambda values: self.gradient(values))    
        return Variable(evaluate = lambda values: self.evaluate(values) + other.evaluate(values),
                        gradient = lambda values: self.gradient(values) + other.gradient(values))
        
    def __radd__(self, other):
        return self + other
########
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) * other,
                            gradient = lambda values: self.gradient(values) * other)    
        return Variable(evaluate = lambda values: self.evaluate(values) * other.evaluate(values),
                        gradient = lambda values: self.gradient(values) * other.evaluate(values) + other.gradient(values) * self.evaluate(values))
    
    def __rmul__(self, other):
       return self * other
########
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) ** other,
                            gradient = lambda values: other * (self.evaluate(values) ** (other - 1)) * self.gradient(values))    
        return Variable.exp(other * Variable.log(self))
    def __rpow(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: other ** self.evaluate(values),
                            gradient = lambda values: math.log(other) * (other ** self.evaluate(values)) * self.gradient(values))    
######## 
    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return (self * -1) + other
######## 
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return (self ** -1) * other
########   
    @staticmethod
    def exp(thing):
        if type(thing) in (int, float):
            return Variable(evaluate = lambda values: math.e ** thing,
                            gradient = lambda values: np.zeroes(len(values)))
        return Variable(evaluate = lambda values: math.e ** thing.evaluate(values),
                        gradient = lambda values: thing.gradient(values) * (math.e ** thing.evaluate(values)))
########
    @staticmethod
    def log(thing):
        if type(thing) in (int, float):
            return Variable(evaluate = lambda values: math.log(thing),
                            gradient = lambda values: np.zeroes(len(values)))
        return Variable(evaluate = lambda values: math.log(thing.evaluate(values)),
                        gradient = lambda values: thing.gradient(values) / thing.evaluate(values))
    
    
    
