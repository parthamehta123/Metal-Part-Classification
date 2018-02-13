"""
file: executeDT.py
language: python3
author: rss1103@rit.edu Rohan Shiroor
        sm2290@rit.edu Sandhya Murali

Uses a trainDT.py to perform classification of the test data.
"""

from trainDT import Node
import csv
import matplotlib.pyplot as plt


def main():
    '''
    Main program to classify the test data and comoute the accuracy
    '''
    filename=input("enter filename: ")
    dataset,classes = read_csv(filename) #read csv file of test data


    root=Node() #create root node
    tree_from_file=read_tree_file('DecisionTree_withoutPruning.csv') #read unpruned tree from file

    print("------------------------BEFORE PRUNING---------------------------------")
    build_tree(tree_from_file,root) #build tree from file

    print_tree(root,0,0) #print tree
    print()

    recognition_percent(root,dataset,classes) #compute recognition (overall accuracy and mean per class)
    print()
    confusion_matrix=build_confusion_matrix(root,dataset,classes) #determine comfusion matrix
    print()
    compute_profit(confusion_matrix) #compute profit
    print()

    plt.figure(1)
    plot_decision_boundary(root, dataset,classes) #plot decision boundary

    print()

    plot_only_boundary(root,0,1,0,1)
    plt.xlabel("Six fold Rotational Symmetry")
    plt.ylabel("Eccentricity")
    plt.title("Decision boundary Lines for Test data")
    plt.show()

    print("------------------------AFTER PRUNING---------------------------------")
    tree_from_file = read_tree_file('DecisionTree_withPruning.csv')
    build_tree(tree_from_file, root)
    print_tree(root, 0, 0)
    print()

    recognition_percent(root, dataset, classes)
    confusion_matrix = build_confusion_matrix(root, dataset, classes)
    print()
    compute_profit(confusion_matrix)
    print()
    plot_decision_boundary(root, dataset,classes)

    plot_only_boundary(root, 0, 1, 0, 1)
    plt.xlabel("Six fold Rotational Symmetry")
    plt.ylabel("Eccentricity")
    plt.title("Decision boundary Lines for Test data")
    plt.show()



def plot_only_boundary(root,xmin,xmax,ymin,ymax):
    '''
        Plots the decision boundary based on split
        :param root: root of decision tree
        :param xmin: minimum x value
        :param xmax: maximum x value
        :param ymin: minimum y value
        :param ymax: maximum y value
        :return
        '''

    if(root.left_pointer==None and root.right_pointer==None): #checks if leaf
        return

    if(root.attribute_value==1): #checks if first attribute

        x1 = root.node_value
        x2 = root.node_value

        x_list = [x1, x2]
        y_list = [ymin, ymax]
        plt.plot(x_list, y_list)
        plot_only_boundary(root.left_pointer, xmin, x2, ymin, ymax)
        plot_only_boundary(root.right_pointer, x1, xmax, ymin, ymax)

    elif(root.attribute_value==2): #checks if second attribute
        y1 = root.node_value
        y2 = root.node_value

        y_list = [y1, y2]
        x_list = [xmin, xmax]

        plt.plot(x_list, y_list)
        plot_only_boundary(root.left_pointer, xmin, xmax, ymin, y2)
        plot_only_boundary(root.right_pointer, xmin, xmax, y1, ymax)

def read_csv(filename):
    '''
    Reads csv file
    :param filename: name of file
    :return dataset,classes: list of samples, list of class from test data
    '''
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        dataset=[]
        classes=[]
        for row in csvReader:
            dataset.append([float(row[0]),float(row[1])])
            classes.append(int(row[2]))

    return dataset,classes


def read_tree_file(filename):
    '''
    Reads tree from file
    :param filename: name of file
    :return tree: list of attributes from file
    '''

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        tree=[]
        for row in csvReader:
            tree.append(row[:])

    return tree

def build_tree(tree_list,root):
    '''
    Builds the tree from file
    :param tree_list:list of tree samples from file
    :param root: root of tree
    :return:
    '''


    element_list=tree_list[0]
    elements=element_list[0]
    data=elements.split('-') #split after -
    attribute_value=int(data[0])
    split_value=float(data[1])
    class_value=int(data[2])
    tree_list.pop(0)

    if(attribute_value==9): #check if leaf node
        root.class_value=class_value
        return

    else:
        root.attribute_value=attribute_value
        root.node_value=split_value

    left_node=Node()
    root.left_pointer=left_node

    build_tree(tree_list,left_node)

    right_node=Node()
    root.right_pointer=right_node
    build_tree(tree_list, right_node)


def print_tree(root,indent,depth):
    '''
    Prints the tree
    :param root: root node of decision tree
    :param indent: indentation parameter
    :param depth: depth of tree
    :return:
    '''

    if(root.left_pointer==None and root.right_pointer==None):
        indentation = '   ' * indent
        print(indentation, '**LEAF NODE** CLASS :', root.class_value)
        return

    indentation = '   ' * indent
    print(indentation, 'ATTRIBUTE SPLIT : ', root.attribute_value, 'ATTRIBUTE SPLIT VALUE : ',
          root.node_value, 'CLASS : ',root.class_value)
    print_tree(root.left_pointer,indent+1,depth+1)
    print_tree(root.right_pointer,indent+1,depth+1)


def recognition_percent(root,test_data,classes):
    '''
    Computes recognition percent
    :param root: root node of decision tree
    :param test_data: list of test data
    :param classes: list of classes
    :return:
    '''

    correct=0
    incorrect=0

    for i in range(len(test_data)):
        predict_class=classify(test_data[i],root)

        if(predict_class==classes[i]):
            correct+=1
        else:
            incorrect+=1

    recognition_rate=correct/(correct+incorrect)
    percent_recognition_rate=recognition_rate*100

    print("--------------------TOTAL ACCURACY (OVERALL RECOGNITION RATE)-------------------------------")
    print()
    print('Overall Accuracy (Recognition Rate) is : ',percent_recognition_rate,'%')
    print()


    correct_prediction=[0,0,0,0]
    incorrect_prediction=[0,0,0,0]

    for i in range(len(test_data)):
        predict_class=classify(test_data[i],root)

        if(predict_class==classes[i]):
            correct_prediction[classes[i]-1]+=1

        else:
            incorrect_prediction[classes[i]-1]+=1

    mean_per_class_accuracy=0
    for i in range(4):
        mean_per_class_accuracy += correct_prediction[i] / ((correct_prediction[i] + incorrect_prediction[i]))

    mean_per_class_accuracy=((mean_per_class_accuracy/4)*100)

    print("-------------------------- MEAN PER CLASS ACCURACY -------------------------------")
    print()
    print('Mean per Class Accuracy is : ',mean_per_class_accuracy)
    print()


def classify(data,root):
    '''
       Classifies data
       :param data: list of samples
       :param root: root node of tree
       :return leaves_list: list of leaves
    '''

    while(root.left_pointer!=None and root.right_pointer!=None):
        split_attribute=root.attribute_value
        split_value=root.node_value

        if(split_attribute==1):
            if(data[0]>split_value):
                root=root.right_pointer
            else:
                root=root.left_pointer

        elif(split_attribute==2):
            if (data[1] > split_value):
                root = root.right_pointer
            else:
                root = root.left_pointer

    return root.class_value


def build_confusion_matrix(root,test_data,classes):
    '''
      Builds confusion matrix
    :param root: root of decision tree
    :param test_data: test data
    :param classes: list of classes
       :return confusion_matrix: confusion matrix
    '''

    confusion_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    for i in range(len(test_data)):
        predict_class = classify(test_data[i], root)

        confusion_matrix[predict_class-1][classes[i]-1]+=1
    print()
    print("--------------------------CONFUSION MATRIX-------------------------------")
    print()
    print('\t\t\t\t\t\t\t\t ACTUAL \t\t\t\t\t\t\t\t')
    print()
    print('ASSIGNED\t CLASS 1\t\tCLASS 2\t\tCLASS 3\t\tCLASS 4\t\tTOTAL')
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

    print()
    return confusion_matrix


def compute_profit(confusion_matrix):
    '''
      Computes profit
       :param confusion_matrix: confusion matrix
       :return
    '''

    profit_matrix=[[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]]

    profit=0

    for i in range(len(confusion_matrix)):
        for j in range(len(profit_matrix)):
            product=confusion_matrix[i][j]*profit_matrix[i][j]
            profit+=product

    print("-------------------------- TOTAL PROFIT -------------------------------")
    print()
    print("Total Profit is : ",profit)

def plot_decision_boundary(root,dataset,classes):
    '''
    Determine the second last node to check if last node is to be pruned
    :param root: root node of tree
    :param dataset: list of samples
    :return leaves_list: list of leaves
    '''

    i=0
    class_1_x=[]
    class_1_y=[]

    class_2_x = []
    class_2_y = []

    class_3_x = []
    class_3_y = []

    class_4_x = []
    class_4_y = []


    while(i<1):
        j=0

        while(j<1):
            predict=classify([i,j],root)

            if(predict==1):
                class_1_x.append(i)
                class_1_y.append(j)

            elif(predict==2):
                class_2_x.append(i)
                class_2_y.append(j)

            elif(predict==3):
                class_3_x.append(i)
                class_3_y.append(j)

            elif(predict==4):
                class_4_x.append(i)
                class_4_y.append(j)
            j+=0.01

        i+=0.001

    plt.plot(class_1_x, class_1_y, color='b')
    plt.plot(class_2_x, class_2_y, color='r')
    plt.plot(class_3_x, class_3_y, color='darkred')
    plt.plot(class_4_x, class_4_y, color='orange')

    marker = ['ro', 'c+', 'mx', 'g*']
    classes_names=['bolts','nuts','ring','scrap']

    for i in range(4):
        list_class_x = []
        list_class_y = []
        for j in range(len(dataset)):
            if (classes[j] == (i + 1)):
                list_class_x.append(dataset[j][0])
                list_class_y.append(dataset[j][1])

        plt.plot(list_class_x, list_class_y, marker[i], label=classes_names[i])

    plt.legend(loc='upper right')
    plt.xlabel("Six fold Rotational Symmetry")
    plt.ylabel("Eccentricity")
    plt.title("Decision boundary for Test data")
    plt.show()




if __name__ == '__main__':
    main()