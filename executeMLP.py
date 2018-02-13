"""
file: executeMLP.py
language: python3
author: rss1103@rit.edu Rohan Shiroor
        sm2290@rit.edu Sandhya Murali

Uses a trained MLP to perform classification of the test data.
"""

import matplotlib.pyplot as plt
import csv
import math

def readInput(file):
    '''
    Reads the test data csv file given as input.
    Returns the features and classes.
    :param file:
    :return: X,Y
    '''

    X = []
    Y = []
    with open(file,'r',newline='') as csvDatafile:
        csvReader = csv.reader(csvDatafile)
        for row in csvReader:
            X.append(row[0:2])
            Y.append(row[2:])
    return X,Y

def readWeight(file2):
    '''
    Reads the weights csv file given as input.
    Returns the features and classes.
    :param file2:
    :return: hiddenWeights,outputWeights
    '''

    hiddenWeights = []
    outputWeights = []
    count = 0
    with open(file2, 'r',newline='') as csvDatafile:
        csvReader = csv.reader(csvDatafile)
        for row in csvReader:
            count+=1
            if count <=5:
                hiddenWeights.append(row)
            else:
                outputWeights.append(row)
    return hiddenWeights,outputWeights

def conversion(X,Y,hiddenWeights,outputWeights):
    '''
    Convert the list created from csv file into floats
    :param X:
    :param Y:
    :param hiddenWeights:
    :param outputWeights:
    :return: X,Y,hiddenWeights,outputWeights
    '''

    for i in range(len(X)):
        X[i][0] = float(X[i][0])
        X[i][1] = float(X[i][1])
        Y[i][0] = int(Y[i][0])
    for i in range(len(hiddenWeights)):
        hiddenWeights[i][0] = float(hiddenWeights[i][0])
        hiddenWeights[i][1] = float(hiddenWeights[i][1])
    for i in range(len(outputWeights)):
        for j in range(5):
            outputWeights[i][j]=float(outputWeights[i][j])

    return X,Y,hiddenWeights,outputWeights

def logistic_func_hid(hiddenWeights,X,Sigmoid):
    '''
    Calculates the Sigmoid values from the given weights
    for hidden layer. Input given is X values.
    :param hiddenWeights:
    :param X:
    :param Sigmoid:
    :return: Sigmoid
    '''

    a = []

    for j in range(5):
        z = 1 + hiddenWeights[j][0] * X[0] + hiddenWeights[j][1] * X[1]
        a.append(1.0 / (1.0 + math.exp(-z)))
    Sigmoid.append(a)
    #print(Sigmoid)
    return Sigmoid

def logistic_func_op(outputWeights,Sigmoid,OpSig):
    '''
    Calculates the Sigmoid values from the given weights
    for output layer. Input given is Sigmoid Values for hidden Layer(a values).
    :param outputWeights:
    :param Sigmoid:
    :param OpSig:
    :return:OpSig
    '''

    Sig = []
    for j in range(4):
        z = 1 + outputWeights[j][0] * Sigmoid[0] + outputWeights[j][1] * Sigmoid[1]+outputWeights[j][2] * Sigmoid[2]+outputWeights[j][3] * Sigmoid[3]+outputWeights[j][4] * Sigmoid[4]
        Sig.append(1.0 / (1.0 + math.exp(-z)))
    #print(Sigmoid)
    OpSig.append(Sig)
    return OpSig

def predict_class(OpSig,Y_pred):
    '''
    Predicts the class according to the
    value of the sigmoid function.
    The max value given by output node
    is the class.
    :param OpSig:
    :param Y_pred
    :return: Y_pred
    '''

    max = -999999999999999
    node = 0

    for j in range(4):
        if OpSig[j] > max:
            max = OpSig[j]
            node = j+1
                #pred.append(1)
    Y_pred.append(node)
    return Y_pred

def recognition_rate(Y,Y_pred):
    '''
    Calculates and prints the recognition rate
    and mean per class recognition rate.
    :param Y:
    :param Y_pred:
    :return: None
    '''

    correctClass = 0
    incorrectClass = 0
    correct_prediction = [0, 0, 0, 0]
    incorrect_prediction = [0, 0, 0, 0]

    for i in range(len(Y)):
        if Y_pred[i] == Y[i][0]:
            correctClass += 1
            correct_prediction[Y[i][0] - 1] += 1
        else:
            incorrectClass+=1
            incorrect_prediction[Y[i][0] - 1] += 1

    mean_per_class_accuracy = 0

    for i in range(4):
        mean_per_class_accuracy += correct_prediction[i] / ((correct_prediction[i] + incorrect_prediction[i]))

    mean_per_class_accuracy = ((mean_per_class_accuracy / 4) * 100)

    recognition_rate = correctClass / (correctClass + incorrectClass)
    percent_recognition_rate = recognition_rate * 100
    print("------------------TOTAL ACCURACY (OVERALL RECOGNITION RATE)-----------------------")
    print()
    print('Total accuracy (overall recognition rate) is : ', percent_recognition_rate, '%')
    print()
    print("--------------------------------------------------------------------------")
    print()
    print("--------------------------MEAN PER CLASS ACCURACY-------------------------------")
    print()
    print('Mean per class accuracy is : ', mean_per_class_accuracy, '%')
    print()
    print("--------------------------------------------------------------------------")


def plot_decisionBoundry(hiddenWeights,outputWeights,X,Y):
    '''
    Plots the points on the graph and draws
    the decision boundry seperating the sets
    of points according to the class it belongs to.
    :param X:
    :param Y:
    :param hiddenWeights:
    :param outputWeights:
    :return: None
    '''
    i = 0
    class_1_x = []
    class_1_y = []

    class_2_x = []
    class_2_y = []

    class_3_x = []
    class_3_y = []

    class_4_x = []
    class_4_y = []

    Sig1 = []
    Sig2 = []
    Y_class = []
    while (i < 1):
        j = 0
        while (j < 1):
            X1 = [i,j]
            Sig1 = logistic_func_hid(hiddenWeights, X1, Sig1)
            Sig2 = logistic_func_op(outputWeights, Sig1[0], Sig2)
            Y_class = predict_class(Sig2[0], Y_class)
            if (Y_class[0] == 1):
                class_1_x.append(i)
                class_1_y.append(j)

            elif (Y_class[0] == 2):
                class_2_x.append(i)
                class_2_y.append(j)

            elif (Y_class[0] == 3):
                class_3_x.append(i)
                class_3_y.append(j)

            elif (Y_class[0] == 4):
                class_4_x.append(i)
                class_4_y.append(j)

            Sig1[:] = []
            Sig2[:] = []
            Y_class[:] = []
            #print(Sig1)
            #print(Sig2)
            j += 0.01
        i += 0.001

    plt.plot(class_1_x, class_1_y, color='b')
    plt.plot(class_2_x, class_2_y, color='r')
    plt.plot(class_3_x, class_3_y, color='darkred')
    plt.plot(class_4_x, class_4_y, color='orange')

    marker = ['ro', 'c+', 'mx', 'g*']
    classes = ['bolt', 'nut', 'ring', 'scrap']
    for i in range(4):
        list_class_x = []
        list_class_y = []
        for j in range(len(Y)):
            if (Y[j][0] == (i + 1)):
                list_class_x.append(X[j][0])
                list_class_y.append(X[j][1])

        plt.plot(list_class_x, list_class_y, marker[i], label=classes[i])

    plt.legend(loc='upper right')
    plt.xlabel("Six fold Rotational Symmetry")
    plt.ylabel("Eccentricity")
    plt.title("Decision boundary")
    plt.show()

def conf_Matrix(Y,Y_pred):
    '''
    Calculates the confusion matrix.
    :param Y:
    :param Y_pred:
    :return: confusion_matrix
    '''

    confusion_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    for i in range(len(Y)):
        confusion_matrix[Y_pred[i] - 1][Y[i][0] - 1] += 1
    print()
    print("--------------------------CONFUSION MATRIX-------------------------------")
    print('\t\t\t\t\t\t\t\t ACTUAL \t\t\t\t\t\t\t\t')
    print()
    print('PREDICTED\t CLASS 1\t\tCLASS 2\t\tCLASS 3\t\tCLASS 4\t\tTOTAL')
    print("--------------------------------------------------------------------------")
    total = 0
    for i in range(len(confusion_matrix)):
        print('CLASS ' + str((i + 1)) + "\t\t\t", end="")
        sum = 0
        for j in range(len(confusion_matrix[i])):
            sum = sum + confusion_matrix[i][j]
            print(str(confusion_matrix[i][j]) + "\t\t\t", end=" ")
        print(sum)
        total += sum
        print("\n")
        print("-----------------------------------------------------------------------")
    total_1 = 0
    total_2 = 0
    total_3 = 0
    total_4 = 0

    for i in range(len(confusion_matrix)):
        total_1 += confusion_matrix[i][0]
        total_2 += confusion_matrix[i][1]
        total_3 += confusion_matrix[i][2]
        total_4 += confusion_matrix[i][3]

    print("TOTAL\t\t\t" + str(total_1) + "\t\t\t" + str(total_2) + "\t\t\t" + str(total_3) + "\t\t\t" + str(
        total_4) + "\t\t\t" + str(total))

    return confusion_matrix

def calc_profit(confusion_matrix):
    '''
    Calculates the profit obtained from the classification
    and class prediction given by MLP.
    :param confusion_matrix:
    :return:None
    '''

    profit_matrix=[[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]]

    profit=0

    for i in range(len(confusion_matrix)):
        for j in range(len(profit_matrix)):
            product=confusion_matrix[i][j]*profit_matrix[i][j]
            profit+=product
    print()
    print("-------------------------- TOTAL PROFIT -------------------------------")
    print()
    print("total profit is : ", profit)
    print()
    print("--------------------------------------------------------------------------")


def main():
    '''
    The main function of the code.
    Uses the trained MLP to predict
    classes for the test data.
    Calculates profit and the recognition rate.
    :return: None
    '''

    file = input("Enter Input File Name:")
    X, Y = readInput(file)
    file2 = input("Input Weights File Name:")
    hiddenWeights,outputWeights = readWeight(file2)
    X,Y,hiddenWeights,outputWeights = conversion(X,Y,hiddenWeights,outputWeights)
    OpSig = []
    Sigmoid = []
    Y_pred = []

# Calculates the Sigmoid values for class prediction.Does classification.
    for i in range(len(X)):
        Sigmoid = logistic_func_hid(hiddenWeights, X[i],Sigmoid)
        OpSig = logistic_func_op(outputWeights, Sigmoid[i],OpSig)
        Y_pred = predict_class(OpSig[i],Y_pred)

# Calculates the recognition rate
    recognition_rate(Y,Y_pred)

# Calculates confusion Matrix
    confusion_matrix = conf_Matrix(Y,Y_pred)

# Calculates profit obtained.
    calc_profit(confusion_matrix)

# Plots the decision boundary between different classes.
    plot_decisionBoundry(hiddenWeights,outputWeights,X,Y)

    #print(Sigmoid)
    #print(OpSig)
    #print(hiddenWeights)
    #print(outputWeights)
    #print(Y_pred)
if __name__ == '__main__':
    main()
