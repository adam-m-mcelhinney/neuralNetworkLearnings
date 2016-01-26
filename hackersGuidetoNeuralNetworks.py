# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:19:41 2016

@author: adam.mcelhinney

Hacker's Guide to Neural Networks
Python Implementation of
http://karpathy.github.io/neuralnets/
"""
import numpy as np


def forwardMultiplyGate(x, y):
    return (x * y)


# How to change x and y to increase the result?

# Strategy 1: Random Search

x = -2
y = 3
tweak_amount = .01
bestOut = -np.inf
bestX = x
bestY = y

for i in xrange(99):
    xTry = x + tweak_amount * (np.random.rand() * 2 - 1)
    yTry = y + tweak_amount * (np.random.rand() * 2 - 1)
    out = forwardMultiplyGate(xTry, yTry)
    if(out > bestOut):
        bestOut = out
        bestX = xTry
        bestY = yTry


# Strategy 2: Numerical Gradient
x = -2
y = 3
h = .0001
out = forwardMultiplyGate(x, y)

xph = x + h
out2 = forwardMultiplyGate(xph, y)
x_derivative = (out2 - out) / h

yph = y + h
out3 = forwardMultiplyGate(x, yph)
y_derivative = (out3 - out) / h


step_size = .01
out = forwardMultiplyGate(x, y)
x = x + step_size * x_derivative
y = y + step_size * y_derivative
out_new = forwardMultiplyGate(x, y)

# Strategy 3, Analytical Gradient
# because d/dx (x * y) = y
x = -2; y = 3;
out = forwardMultiplyGate(x, y)
x_gradient = y
y_gradient = x
step_size = .01
x += step_size * x_gradient
y += step_size * y_gradient
out_new = forwardMultiplyGate(x, y)

# Recursive Case

def forwardAddGate(a, b):
    return a + b

def forwardCircuit(x, y, z):
    q = forwardAddGate(x, y)
    f = forwardMultiplyGate(q, z)
    return f

x = -2; y = 5; z = -4;
f = forwardCircuit(x, y, z)

# Backpropogation
x = -2; y = 5; z = -4;
q = forwardAddGate(x, y)
f = forwardMultiplyGate(q, z)
derivative_f_wrt_z = q
derivative_f_wrt_q = z

derivative_q_wrt_x = 1
derivative_q_wrt_y = 1

derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q

derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q

gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

step_size = .01
x = x + step_size * derivative_f_wrt_x
y = y + step_size * derivative_f_wrt_y
z = z + step_size * derivative_f_wrt_z

print x, y, z

q = forwardAddGate(x, y)
f = forwardMultiplyGate(q, z)

# Numerical Gradient Check
# Verify that our analytical gradient is correct
x = -2; y = 5; z = -4;
h = .0001
x_derivative = (forwardCircuit(x + h, y, z) - forwardCircuit(x, y, z))/h
y_derivative = (forwardCircuit(x, y + h, z) - forwardCircuit(x, y, z))/h
z_derivative = (forwardCircuit(x, y, z + h) - forwardCircuit(x, y, z))/h

print x_derivative, y_derivative, z_derivative


# Example, Single Neuron

class Units:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

class multiplyGate:
    def __init__(self, unit1, unit2):
        self.unit1 = unit1
        self.unit2 = unit2

    def forward(self):
        self.utop = Units(self.unit1.value * self.unit2.value, 0)

    def backword(self):
        self.unit1.grad += self.unit2.value * self.utop.grad
        self.unit2.grad += self.unit1.value * self.utop.grad


