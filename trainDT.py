"""
file: trainDT.py
language: python3

Builds a decision tree model from training samples.
"""

import csv
import math
import operator
import matplotlib.pyplot as plt

SIGNIFICANCE=0.05 #significance for Chi-Square Pruning
class Node:
    __slots__ ='class_value','count_nodes','classes','left_child','right_child','left_pointer','right_pointer','node_value',\
               'attribute_value'

    def __init__(self):

        '''
        Creates a Node object object for tree node
        :param class_value: class value
        :param count_nodes: total number of nodes
        :param classes:total number of classes
        :param left_child: left child list
        :param right_child: right child list
        :param left_pointer: pointer to left node
        :param right_pointer: pointer to right node
        :param node_value: split value
        :param attribute_value: feature name
        '''
        self.class_value=0
        self.count_nodes=0
        self.classes={}
        self.left_child=[]
        self.right_child=[]
        self.left_pointer=None
        self.right_pointer=None
        self.node_value=0
        self.attribute_value=0

def write_to_file(file_ptr,root):
    '''
    Write decision tree model to file
    :param file_ptr: file pointer
    :param root: root of decision tree
    :return:
    '''


    if(root.left_pointer==None and root.right_pointer==None): #checks leaf node
        string="9-0-"+str(root.class_value)
        file_ptr.write(string)
        file_ptr.write("\n")
        return


    string=""+str(root.attribute_value)+"-"+str(root.node_value)+"-"+str(root.class_value)
    file_ptr.write(string)
    file_ptr.write("\n")
    write_to_file(file_ptr,root.left_pointer)
    write_to_file(file_ptr,root.right_pointer)



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

    if(root.left_pointer==None and root.right_pointer==None): #if leaf node
        return

    if(root.attribute_value==1): # if first feature

        x1 = root.node_value
        x2 = root.node_value

        x_list = [x1, x2]
        y_list = [ymin, ymax]
        plt.plot(x_list, y_list)
        plot_only_boundary(root.left_pointer, xmin, x2, ymin, ymax)
        plot_only_boundary(root.right_pointer, x1, xmax, ymin, ymax)

    elif(root.attribute_value==2): #if second feature
        y1 = root.node_value
        y2 = root.node_value

        y_list = [y1, y2]
        x_list = [xmin, xmax]

        plt.plot(x_list, y_list)
        plot_only_boundary(root.left_pointer, xmin, xmax, ymin, y2)
        plot_only_boundary(root.right_pointer, xmin, xmax, y1, ymax)


def main():
    '''
       Main program to call the functions to build decision tree
    '''

    file_name=input("enter filename :")
    dataset=read_csv(file_name)

    f = open('DecisionTree_withoutPruning.csv', 'w')
    print("------------------------BEFORE PRUNING---------------------------------")
    print()
    root,leaves_list=build_DT(dataset,[],0) #build Decision tree
    print_tree(root,0,0) #print tree
    print()
    write_to_file(f,root) #write to file



    print("-----------------TREE METRICS BEFORE PRUNING---------------------------")
    print()
    depth = compute_max_depth(root) #max depth
    print('Max Depth Before Pruning :',depth)
    print()

    depth_min=compute_min_depth(root) #minimum depth
    print('Min Depth Before Pruning', depth_min)
    print()

    average_depth=sum(leaves_list)/len(leaves_list) #average depth
    print('Average Depth Before Pruning',average_depth)
    print()

    count=get_total_nodes(root) #total number of nodes
    print('Total Nodes Before Pruning',count)
    print()

    count_leaf=get_leaf_nodes(root) #total number of leaf nodes
    print('Leaf Nodes Before Pruning is :',count_leaf)


    plot_decision_boundary(root,dataset) #plots decision boundary
    plt.show()
    print()

    print("-----------------AFTER PRUNING---------------------------")

    depth_just_before_leaf=depth-1

    while(depth_just_before_leaf!=0):
        leaves_list=compute_chisquare(root,0,depth_just_before_leaf,leaves_list,0) #performs Chi-Square test
        depth_just_before_leaf-=1

    f = open('DecisionTree_withPruning.csv', 'w')
    print_tree(root,0,0)
    print()
    write_to_file(f, root)



    print("-----------------TREE METRICS AFTER PRUNING---------------------------")
    print()
    depth = compute_max_depth(root)
    print('Max Depth After Pruning', depth)
    print()

    depth_min = compute_min_depth(root)
    print('Min Depth After Pruning', depth_min)
    print()

    average_depth = sum(leaves_list) / len(leaves_list)
    print('Average Depth After Pruning', average_depth)
    print()

    count = get_total_nodes(root)
    print('Total Nodes After Pruning', count)
    print()

    count_leaf = get_leaf_nodes(root)
    print('Leaf Nodes After Pruning is :', count_leaf)
    print()

    plot_decision_boundary(root, dataset)
    #plot_only_boundary(root,0,1,0,1)
    plt.show()


def read_csv(filename):
    '''
    Reads csv file
    :param filename: Name of file
    :return dataset: list of data
    '''
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        dataset=[]
        for row in csvReader:
            dataset.append([float(row[0]),float(row[1]),int(row[2])])

    return dataset

def print_tree(root,indent,depth):
    '''
    Prints the tree
    :param root: root node of decision tree
    :param indent: indentation parameter
    :param depth: depth of tree
    :return:
    '''

    if(root.left_pointer==None and root.right_pointer==None): #check leaf node
        indentation = '   ' * indent
        print(indentation, '**LEAF NODE** CLASS :', root.class_value)
        return

    indentation = '   ' * indent
    print(indentation, 'ATTRIBUTE SPLIT : ', root.attribute_value, 'ATTRIBUTE SPLIT VALUE : ',
          root.node_value, 'CLASS : ',root.class_value)

    print_tree(root.left_pointer,indent+1,depth+1)
    print_tree(root.right_pointer,indent+1,depth+1)

def build_DT(dataset,list_leaves,count):
    '''
    Builds the decision tree
    :param dataset: list of samples
    :param list_leaves: list of leaves
    :param count: depth value
    :return best_split_node, list_leaves: node of tree and list of leaf nodes with depth value
    '''

    same_class=False
    class_value=-1

    classes = [row[-1] for row in dataset]
    if classes.count(classes[0]) == len(classes): #check if all belong to same class


        same_class=True
        if(classes[0]==1):
            class_value=1
        elif(classes[0]==2):
            class_value=2
        elif(classes[0]==3):
            class_value=3
        else:
            class_value=4

    if(same_class==True): #if leaf node
        node=Node()
        node.class_value=class_value
        node.attribute_value=3
        node.count_nodes=len(dataset)
        dict_classes=get_class_and_count(dataset)
        node.classes=dict_classes
        list_leaves.append(count)
        return node,list_leaves
    else:

        best_split_node,threshold,attribute=best_split(dataset) #determine best node after split

        best_split_node.left_pointer,list_leaves=build_DT(best_split_node.left_child,list_leaves,count+1)

        best_split_node.right_pointer,list_leaves=build_DT(best_split_node.right_child,list_leaves,count+1)

    return best_split_node,list_leaves


def get_class_and_count(dataset):
    '''
      Get count of each class
       :param dataset: training samples
       :return dict_classes: dictionary of classes
    '''

    dict_classes={}
    for i in range(len(dataset)):

        if(dataset[i][2] in dict_classes):
            count=dict_classes[dataset[i][2]]
            count+=1
            dict_classes[dataset[i][2]]=count
        else:
            dict_classes[dataset[i][2]]=1



    if(1 not in dict_classes.keys()):
        dict_classes[1]=0

    if (2 not in dict_classes.keys()):
        dict_classes[2] = 0

    if (3 not in dict_classes.keys()):
        dict_classes[3] = 0

    if (4 not in dict_classes.keys()):
        dict_classes[4] = 0


    return dict_classes

def best_split(dataset):
    '''
    Determines best node based on best split
    :param dataset: list of samples

    :return node,midpoint,attribute: returns node with best split, best split value, and attribute name for split
    '''

    sort_first_attribute=sorted(dataset, key = lambda x:x[0]) #sort on first attribute
    list_midpoints_x=compute_midpoint_x(sort_first_attribute) #compute midpoint
    info_gain_x=compute_info_gain_x(list_midpoints_x,dataset) #compute information gain

    sort_second_attribute=sorted(dataset, key = lambda x:x[1]) #sort on second attribute
    list_midpoints_y=compute_midpoint_y(sort_second_attribute)
    info_gain_y = compute_info_gain_y(list_midpoints_y, dataset)

    max_info_gain_x=max(info_gain_x.items(), key=operator.itemgetter(1))[1] #max information gain
    max_info_gain_y = max(info_gain_y.items(), key=operator.itemgetter(1))[1]

    if(max_info_gain_x>max_info_gain_y): #max information gain from both attributes and if it is first feature
        attribute='x1'
        midpoint=-1
        for mid, info_gain in info_gain_x.items():
            if info_gain == max_info_gain_x:
                midpoint=mid
                break

        node=Node() #create node for best split
        dict_classes=get_class_and_count(dataset)
        node.classes=dict_classes
        node.count_nodes=len(dataset)
        node.node_value=midpoint
        left,right=compute_split_x(midpoint,dataset)
        node.left_child=left
        node.right_child=right
        node.attribute_value=1

        return node,midpoint,attribute


    else: #if second feature is best split attribute

        midpoint = -1
        attribute = 'x2'
        for mid, info_gain in info_gain_y.items():
            if info_gain == max_info_gain_y:
                midpoint = mid
                break

        node = Node()
        dict_classes = get_class_and_count(dataset)
        node.classes = dict_classes
        node.node_value=midpoint
        node.count_nodes = len(dataset)
        left, right = compute_split_y(midpoint, dataset)
        node.left_child = left
        node.right_child = right
        node.attribute_value=2

        return node,midpoint,attribute


def compute_midpoint_x(dataset):
    '''
    Compute midpoints for for first feature
    :param dataset: list of samples
    :return list_midpoints_x: list of midpoint values
    '''

    list_midpoints_x=[]
    for i in range(len(dataset)-1):
        midpoint=(dataset[i][0]+dataset[i+1][0])/2
        list_midpoints_x.append(midpoint)

    return list_midpoints_x


def compute_info_gain_x(list_midpoints,list_points):
    '''
       Compute information gain for for first feature
       :param list_midpoints: list of midpoints
       :param list_points: list of samples
       :return information_gain_dict: dictionary of information gain where key is midpoint and value is information gain
    '''

    info_gain_dict={}

    for i in range(len(list_midpoints)):
        midpoint=list_midpoints[i]

        left_split,right_split=compute_split_x(midpoint,list_points) #determine left and right lists
        entropy_left=compute_entropy(left_split) #compute entropy
        entropy_right=compute_entropy(right_split)
        parent_entropy=compute_entropy(list_points)
        info_gain=parent_entropy-(((len(left_split)/len(list_points))*entropy_left) + ((len(right_split)/len(list_points))*entropy_right))
        info_gain_dict[midpoint]=info_gain

    return info_gain_dict



def compute_split_x(midpoint,list_points):
    '''
    Split into left and right lists based on split value of first feature
    :param midpoint: midpoint value
    :param list_points: list of samples
    :return left_split,right_split: split into two lists,left and right
    '''

    left_split=[]
    right_split=[]

    for i in range(len(list_points)):
        if(list_points[i][0]<=midpoint):
            left_split.append(list_points[i])
        else:
            right_split.append(list_points[i])

    return left_split,right_split


def compute_midpoint_y(dataset):
    '''
       Compute midpoints for for second feature
       :param dataset: list of samples
       :return list_midpoints_x: list of midpoint values
    '''
    list_midpoints_y = []
    for i in range(len(dataset) - 1):
        midpoint = (dataset[i][1] + dataset[i + 1][1]) / 2
        list_midpoints_y.append(midpoint)

    return list_midpoints_y


def compute_info_gain_y(list_midpoints,list_points):
    '''
    Compute information gain for for second feature
    :param list_midpoints: list of midpoints
    :param list_points: list of samples
    :return information_gain_dict: dictionary of information gain where key is midpoint and value is information gain
    '''

    info_gain_dict={}

    for i in range(len(list_midpoints)):
        midpoint=list_midpoints[i]

        left_split,right_split=compute_split_y(midpoint,list_points)
        entropy_left=compute_entropy(left_split)
        entropy_right=compute_entropy(right_split)
        parent_entropy=compute_entropy(list_points)
        info_gain=parent_entropy-(((len(left_split)/len(list_points))*entropy_left) + ((len(right_split)/len(list_points))*entropy_right))
        info_gain_dict[midpoint]=info_gain

    return info_gain_dict



def compute_split_y(midpoint,list_points):
    '''
    Split into left and right lists based on split value of second feature
    :param midpoint: midpoint value
    :param list_points: list of samples
    :return left_split,right_split: split into two lists,left and right
    '''

    left_split=[]
    right_split=[]

    for i in range(len(list_points)):
        if(list_points[i][1]<=midpoint):
            left_split.append(list_points[i])
        else:
            right_split.append(list_points[i])

    return left_split,right_split


def compute_entropy(list_points):
    '''
    Compute entropy
    :param list_points: list of samples
    :return entropy: entropy value
    '''

    entropy=0
    class_1=0 #class 1 total
    class_2=0 #class 2 total
    class_3=0 #class 3 total
    class_4=0 #class 4 total

    for i in range(len(list_points)):
        if(list_points[i][2]==1):
            class_1+=1

        elif (list_points[i][2] == 2):
            class_2 += 1

        elif (list_points[i][2] == 3):
            class_3 += 1

        elif (list_points[i][2] == 4):
            class_4 += 1


    if(class_1>0):
        entropy+=(-((class_1/len(list_points)) * math.log(class_1/len(list_points), 2)))
    else:
        entropy+=0

    if (class_2 > 0):
        entropy += (-((class_2 / len(list_points)) * math.log(class_2 / len(list_points), 2)))
    else:
        entropy += 0

    if (class_3 > 0):
        entropy += (-((class_3 / len(list_points)) * math.log(class_3 / len(list_points), 2)))
    else:
        entropy += 0

    if (class_4 > 0):
        entropy += (-((class_4 / len(list_points)) * math.log(class_4 / len(list_points), 2)))
    else:
        entropy += 0

    return entropy

def compute_max_depth(node):
    '''
       Determine max depth of the tree
       :param node: root node of tree
       :return max depth
    '''

    if(node.left_pointer==None and node.right_pointer==None):
        return 0

    return 1+max(compute_max_depth(node.left_pointer),compute_max_depth(node.right_pointer))

def compute_min_depth(node):
    '''
    Determine min depth of the tree
    :param node: root node of tree
    :return min depth
    '''

    if(node.left_pointer==None and node.right_pointer==None):
        return 0

    return 1+min(compute_min_depth(node.left_pointer),compute_min_depth(node.right_pointer))


def get_total_nodes(root):
    '''
       Get total number of nodes in the tree
       :param root: root node of tree
       :return total nodes
    '''
    if(root == None):
        return 0

    if(root.left_pointer ==None and root.right_pointer==None):
        return 1

    else:
	    return get_total_nodes(root.left_pointer)+ get_total_nodes(root.right_pointer)+1

def get_leaf_nodes(root):
    '''
    Get leaf nodes count
    :param root: root node of tree
    :return count of leaf nodes
    '''

    list_nodes=[]
    list_nodes.append(root)
    count_nodes=0

    while(len(list_nodes)>0):
        node=list_nodes.pop(0)

        if(node.left_pointer==None and node.right_pointer==None): #if leaf node
            count_nodes+=1
        else:

            list_nodes.append(node.left_pointer)
            list_nodes.append(node.right_pointer)

    return count_nodes


def compute_chisquare(root,current_depth,prune_depth,leaves_list,count=0):
    '''
       Determine the second last node to check if last node is to be pruned
       :param root: root node of tree
       :param current_depth: current depth of tree
       :param prune_depth: prune depth of tree
       :param leaves_list :list of leaves
       :param count: depth
       :return leaves_list: list of leaves
    '''

    if(current_depth==prune_depth):
        if(root.left_pointer!=None and root.right_pointer!=None and root.left_pointer.left_pointer==None
           and root.left_pointer.right_pointer==None and root.right_pointer.left_pointer==None and
            root.right_pointer.right_pointer==None):
            leaves_list=compute_chisquare_value(root,leaves_list,count) #compute chi square value
            return leaves_list


    if(root.left_pointer!=None and root.right_pointer!=None):
        leaves_list=compute_chisquare(root.left_pointer,current_depth+1,prune_depth,leaves_list,count+1)
        leaves_list=compute_chisquare(root.right_pointer,current_depth+1,prune_depth,leaves_list,count+1)

    return leaves_list


def compute_chisquare_value(root,leaves_list,count):
    '''
    Compute
    :param root: root node of tree
    :param leaves_list :list of leaves
    :param count: depth
    :return leaves_list: list of leaves
    '''

    pL=[] #left expected value for each class
    pR=[] #right expect value for each class


    dict_classes_parent_node=root.classes
    total_class_1_parent_node = dict_classes_parent_node[1]
    pl_left = total_class_1_parent_node * root.left_pointer.count_nodes / root.count_nodes
    pL.append(pl_left)

    pr_right = total_class_1_parent_node * root.right_pointer.count_nodes / root.count_nodes
    pR.append(pr_right)

    total_class_2_parent_node = dict_classes_parent_node[2]
    pl_left = total_class_2_parent_node * root.left_pointer.count_nodes / root.count_nodes
    pL.append(pl_left)

    pr_right = total_class_2_parent_node * root.right_pointer.count_nodes / root.count_nodes
    pR.append(pr_right)

    total_class_3_parent_node = dict_classes_parent_node[3]
    pl_left = total_class_3_parent_node * root.left_pointer.count_nodes / root.count_nodes
    pL.append(pl_left)

    pr_right = total_class_3_parent_node * root.right_pointer.count_nodes / root.count_nodes
    pR.append(pr_right)

    total_class_4_parent_node = dict_classes_parent_node[4]
    pl_left = total_class_4_parent_node * root.left_pointer.count_nodes / root.count_nodes
    pL.append(pl_left)

    pr_right = total_class_4_parent_node * root.right_pointer.count_nodes / root.count_nodes
    pR.append(pr_right)

    dict_left_node = root.left_pointer.classes
    list_classes_left_node = list(dict_left_node.keys())
    list_classes_left_node = sorted(list_classes_left_node)

    left_delta = 0.0 #compute delta
    for i in range(len(list_classes_left_node)):
        if (pL[i] > 0):
            left = math.pow((pL[i] - dict_left_node[list_classes_left_node[i]]), 2)
            left = left / pL[i]
            left_delta += left

    dict_right_node = root.right_pointer.classes
    list_classes_right_node = list(dict_right_node.keys())
    list_classes_right_node = sorted(list_classes_right_node)

    right_delta = 0.0
    for i in range(len(list_classes_right_node)):
        if (pR[i] > 0):
            right = math.pow((pR[i] - dict_right_node[list_classes_right_node[i]]), 2)
            right = right / pR[i]
            right_delta += right

    delta = left_delta + right_delta

    #chi-square pruning threshold for 5%
    if(delta<7.82 and SIGNIFICANCE==0.05):
        left_child_length = len(root.left_child)
        right_child_length = len(root.right_child)

        if (left_child_length > right_child_length):
            root.class_value = root.left_pointer.class_value
            leaves_list.remove(count+1)
            leaves_list.remove(count+1)
            leaves_list.append(count)
            root.left_pointer = None
            root.right_pointer = None
        else:
            root.class_value = root.right_pointer.class_value
            root.left_pointer = None
            root.right_pointer = None
            leaves_list.remove(count + 1)
            leaves_list.remove(count + 1)
            leaves_list.append(count)

    #chi-square pruning for 1%
    if (delta < 11.35 and SIGNIFICANCE == 0.01):
        left_child_length = len(root.left_child)
        right_child_length = len(root.right_child)

        if (left_child_length > right_child_length):
            root.class_value = root.left_pointer.class_value
            leaves_list.remove(count + 1)
            leaves_list.remove(count + 1)
            leaves_list.append(count)
            root.left_pointer = None
            root.right_pointer = None
        else:
            root.class_value = root.right_pointer.class_value
            root.left_pointer = None
            root.right_pointer = None
            leaves_list.remove(count + 1)
            leaves_list.remove(count + 1)
            leaves_list.append(count)

    #BONUS QUESTION
    #chi-square pruning with significance 0.20
    if (delta < 4.642 and SIGNIFICANCE == 0.2):
        left_child_length = len(root.left_child)
        right_child_length = len(root.right_child)

        if (left_child_length > right_child_length):
            root.class_value = root.left_pointer.class_value
            leaves_list.remove(count + 1)
            leaves_list.remove(count + 1)
            leaves_list.append(count)
            root.left_pointer = None
            root.right_pointer = None
        else:
            root.class_value = root.right_pointer.class_value
            root.left_pointer = None
            root.right_pointer = None
            leaves_list.remove(count + 1)
            leaves_list.remove(count + 1)
            leaves_list.append(count)


    return leaves_list


def plot_decision_boundary(root,dataset):
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
            predict=classify([i,j],root) #predict the class

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
    classes=['bolts','nuts','ring','scrap']
    for i in range(4):
        list_class_x = []
        list_class_y = []
        for j in range(len(dataset)):
            if (dataset[j][2] == (i + 1)):
                list_class_x.append(dataset[j][0])
                list_class_y.append(dataset[j][1])

        plt.plot(list_class_x, list_class_y, marker[i],label=classes[i]) #plot points

    plt.legend(loc='upper right')
    plt.xlabel("Six fold Rotational Symmetry")
    plt.ylabel("Eccentricity")
    plt.title("Decision Boundary for Training data")
    plt.show()


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


if __name__ == '__main__':
    main()