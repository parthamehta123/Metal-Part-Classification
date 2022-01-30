"""
file: trainMLP.py
language: python3
This program trains a multi-layer perceptron on training data
provided by the user.
"""

import matplotlib.pyplot
import csv
import math
import random

def readFile(file):
    '''
    Reads the train data csv file given as input.
    Returns the features and classes.
    :param file:
    :return: X,Y
    '''

    X = []
    Y = []
    with open(file,'r') as csvDatafile:
        csvReader = csv.reader(csvDatafile)
        for row in csvReader:
            X.append(row[0:2])
            Y.append(row[2:])
    return X,Y

def init_network(Nin,Nh,Nop):
    '''
    Initializes the multi-layer perceptron with random weights.
    :param Nin:
    :param Nh:
    :param Nop:
    :return: hiddenWeights, outputWeights
    '''

    hiddenWeights = [[random.uniform(1,-1)for i in range(2)]for i in range(Nh)]
    outputWeights = [[random.uniform(1,-1)for i in range(5)]for i in range(Nop)]
    return hiddenWeights,outputWeights

def logistic_func_hid(hiddenWeights,X):
    '''
    Calculates the Sigmoid values from the given weights
    for hidden layer. Input given is X values.
    :param hiddenWeights:
    :param X:
    :return: Sigmoid
    '''

    Sigmoid = []

    for j in range(5):
        z = 1 + hiddenWeights[j][0] * X[0] + hiddenWeights[j][1] * X[1]
        Sigmoid.append(1.0 / (1.0 + math.exp(-z)))
    #print(Sigmoid)
    return Sigmoid

def logistic_func_op(outputWeights,Sigmoid,Sig):
    '''
    Calculates the Sigmoid values from the given weights
    for output layer. Input given is Sigmoid Values for hidden Layer(a values).
    :param outputWeights:
    :param Sigmoid:
    :param Sig:
    :return:OpSig,Sig
    '''

    OpSig = []
    for j in range(4):
        z = 1 + outputWeights[j][0] * Sigmoid[0] + outputWeights[j][1] * Sigmoid[1]+outputWeights[j][2] * Sigmoid[2]+outputWeights[j][3] * Sigmoid[3]+outputWeights[j][4] * Sigmoid[4]
        OpSig.append(1.0 / (1.0 + math.exp(-z)))
    #print(Sigmoid)
    Sig.append(OpSig)
    return OpSig,Sig

def backprop_Op(OpSig,Y):
    '''
    Calculates the delta values for the output layers
    and returns it. This is first stage of back propagation.
    :param Sigmoid:
    :param OpSig:
    :param Y:
    :return: D
    '''

    D = []

    for j in range(4):
        D.append((Y[j] - OpSig[j]) * OpSig[j] * (1 - OpSig[j]))
    return D

def backprop_Hid(D,Sigmoid,outputWeights):
    '''
    Calculates the delta values for the hidden layer
    and returns it. This is second stage of back propagation.
    :param D:
    :param Sigmoid:
    :param outputWeights:
    :return: DH
    '''

    DH = []

    for j in range(5):
        D2 = ((D[0]*outputWeights[0][j]+D[1]*outputWeights[1][j]+D[2]*outputWeights[2][j]+D[3]*outputWeights[3][j])*Sigmoid[j]*(1 -Sigmoid[j]))
        DH.append(D2)
    return DH

def update_weights_hid(X,hiddenWeights,DH):
    '''
    Updates the weights for the connection between
    the input layer and hidden layer. Third stage
    of back propagation.
    :param X:
    :param hiddenWeights:
    :param DH:
    :return:hiddenWeights
    '''

    a = 0.01

    for j in range(len(hiddenWeights)):
        hiddenWeights[j][0] = hiddenWeights[j][0] +  (a * DH[j] * X[0])
        hiddenWeights[j][1] = hiddenWeights[j][1] + (a * DH[j] * X[1])
    return hiddenWeights

def update_weights_op(Sigmoid,outputWeights,D):
    '''
    Updates the weights for the connection between
    hidden layer and output layer. Fourth stage
    of back propagation.
    :param Sigmoid:
    :param outputWeights:
    :param D:
    :return: outputWeights
    '''

    a = 0.01

    for j in range(len(outputWeights)):
        outputWeights[j][0] = outputWeights[j][0] + (a * D[j] * Sigmoid[0])
        outputWeights[j][1] = outputWeights[j][1] + (a * D[j] * Sigmoid[1])
        outputWeights[j][2] = outputWeights[j][2] + (a * D[j] * Sigmoid[2])
        outputWeights[j][3] = outputWeights[j][3] + (a * D[j] * Sigmoid[3])
        outputWeights[j][4] = outputWeights[j][4] + (a * D[j] * Sigmoid[4])
    return outputWeights

def calc_SSE(Y,OpSig,err):
    '''
    Calculates the Sum of Squared Error
    :param Y:
    :param OpSig:
    :param err:
    :return: err
    '''

    val = 0
    for i in range(len(Y)):
        for j in range(4):
            val+= pow((Y[i][j]-OpSig[i][j]),2)
    val = val/2
    err.append(val)
    return err

def plot_SSE_Epoch(err):
    '''
    Plots the graph of SSE vs Epoch
    :param err:
    :return: None
    '''

    plt = matplotlib.pyplot
    plt.xlabel('Epoch')
    plt.ylabel('Sum of Squared Errors')
    plt.title("SSE vs Epoch")
    epoch = list(range(1,10001))
    plt.plot(epoch,err,label='Training Data')
    plt.show()

def main():
    '''
    The main function of the code.
    Runs the MLP for 10,000 iterations.
    Writes the weights for 0,10,100,1000,10000
     to respective weights.csv files.
    :return: None
    '''

    file = input("Enter File Name:")
    X,Y = readFile(file)
    ## Convert the list created from csv file into floats ##
    for i in range(len(X)):
        X[i][0] = float(X[i][0])
        X[i][1] = float(X[i][1])
        Y[i][0] = int(Y[i][0])
    for i in range(0,len(Y)):
        if Y[i][0]==1:
            Y[i]= [1,0,0,0]
        if Y[i][0]==2:
            Y[i]= [0,1,0,0]
        if Y[i][0]==3:
            Y[i]= [0,0,1,0]
        if Y[i][0]==4:
            Y[i]= [0,0,0,1]

    # Initialize the network with random weights.
    hiddenWeights, outputWeights = init_network(2,5,4)

    #print(hiddenWeights)
    #print(outputWeights)

    err = []
    Sig = []

    # Store the randomly initialized weights to weights0.csv file.
    with open("weights0.csv", "w",newline='') as f:
        writer = csv.writer(f)
        writer.writerows(hiddenWeights)
        writer.writerows(outputWeights)

    # Runs the MLP for 10,000 Epochs
    for j in range(10000):

# Stochastic Training:- Forward propagation, Back propagation after reading each sample from csv file.
        for i in range(len(X)):
            Sigmoid = logistic_func_hid(hiddenWeights, X[i])
            # print(Sigmoid)
            OpSig,Sig = logistic_func_op(outputWeights,Sigmoid,Sig)
            #print(OpSig)
            D = backprop_Op(OpSig,Y[i])
            # print(D)
            DH = backprop_Hid(D,Sigmoid,outputWeights)
            # print(DH)
            hiddenWeights = update_weights_hid(X[i],hiddenWeights,DH)
            # print(hiddenWeights)
            outputWeights = update_weights_op(Sigmoid,outputWeights,D)
            # print(outputWeights)
        err = calc_SSE(Y, Sig, err)
        Sig[:] = []
# Store weights after 10 epochs to weights10.csv file.
        if j == 9:
            with open("weights10.csv", "w",newline='') as f:
                writer = csv.writer(f)
                writer.writerows(hiddenWeights)
                writer.writerows(outputWeights)

# Store weights after 100 epochs to weights100.csv file.
        elif j == 99:
            with open("weights100.csv", "w",newline='') as f:
                writer = csv.writer(f)
                writer.writerows(hiddenWeights)
                writer.writerows(outputWeights)

# Store weights after 1000 epochs to weights1000.csv file.
        elif j == 999:
            with open("weights1000.csv", "w",newline='') as f:
                writer = csv.writer(f)
                writer.writerows(hiddenWeights)
                writer.writerows(outputWeights)

# Store weights after 10,000 epochs to weights10000.csv file.
        elif j == 9999:
            with open("weights10000.csv", "w",newline='') as f:
                writer = csv.writer(f)
                writer.writerows(hiddenWeights)
                writer.writerows(outputWeights)
    #print(hiddenWeights)
    #print(outputWeights)
    #print(err)
    #=print(OpSig)

    # Plot SSE vs Epoch.
    plot_SSE_Epoch(err)

if __name__ == '__main__':
    main()