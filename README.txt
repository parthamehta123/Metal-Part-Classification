Foundations of Intelligent Systems
Project 2


SANDHYA MURLI (sm2290@g.rit.edu)
ROHAN SHIROOR (rss1103@g.rit.edu)


----------------------------------------------------------------------------------
Files in this directory:


trainDT.py:  The python file which consists of building the decision tree.
executeDT.py: The python file which consists of testing the decision tree.
trainMLP.py: The python file which consists of building the MLP.
executeMLP.py:The python file which consists of testing the MLP.
DecisionTree_withoutPruning.csv: CSV file that consists of the model after the decision tree is built without pruning
DecisionTree_withPruning.csv: CSV file that consists of the model after the decision tree is built with pruning
README.txt: README file
#add MLP weight files


a.pdf: Report containing discussions
----------------------------------------------------------------------------------
Steps to Run the code:


----------------------------FOR DECISION TREE--------------------------------------------
For training the decision tree:


1. Extract zip file
1. OPEN cmd (or terminal if Linux or MAC OSX)
2. GO to the path where the folder is saved
3. RUN command: python trainDT.py
4. Give input of file name containing csv file for training data




INTERPRETATION OF OUTPUT:


A:  BEFORE PRUNING :


1. Prints the tree


------------------------BEFORE PRUNING---------------------------------


 ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.25143 CLASS :  0
    ATTRIBUTE SPLIT :  1 ATTRIBUTE SPLIT VALUE :  0.29935 CLASS :  0
       ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.10739 CLASS :  0
          **LEAF NODE** CLASS : 3
          ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.108765 CLASS :  0
             **LEAF NODE** CLASS : 4
             **LEAF NODE** CLASS : 3
       ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.099544 CLASS :  0
          **LEAF NODE** CLASS : 2
          ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.10125 CLASS :  0
             **LEAF NODE** CLASS : 4
             **LEAF NODE** CLASS : 2
    ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.7854 CLASS :  0
       ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.7355700000000001 CLASS :  0
          **LEAF NODE** CLASS : 4
          ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.74132 CLASS :  0
             **LEAF NODE** CLASS : 1
             **LEAF NODE** CLASS : 4
       **LEAF NODE** CLASS : 1


2. Tree Metrics before Pruning:
1.  Prints the maximum depth
2.  Prints the maximum depth
3.  Prints the maximum depth
4. Prints Total number of nodes
5. Prints the number of leaf nodes


-----------------TREE METRICS BEFORE PRUNING---------------------------


Max Depth Before Pruning : 4  
Min Depth Before Pruning 2  
Average Depth Before Pruning 3.5


Total Nodes Before Pruning 19   
Leaf Nodes Before Pruning is : 10 


3. Decision Boundary before Pruning:


  











B. AFTER PRUNING 


1. Prints the Tree
-----------------AFTER PRUNING---------------------------
 ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.25143 CLASS :  0
    ATTRIBUTE SPLIT :  1 ATTRIBUTE SPLIT VALUE :  0.29935 CLASS :  0
       **LEAF NODE** CLASS : 3
       **LEAF NODE** CLASS : 2
    ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.7854 CLASS :  0
       **LEAF NODE** CLASS : 4
       **LEAF NODE** CLASS : 1




2. Tree Metrics after  Pruning:
1.  Prints the maximum depth
2.  Prints the maximum depth
3.  Prints the maximum depth
4. Prints Total number of nodes
5. Total number of Leaf Nodes


-----------------TREE METRICS AFTER PRUNING---------------------------


Max Depth After Pruning 2


Min Depth After Pruning 2


Average Depth After Pruning 2.0


Total Nodes After Pruning 7


Leaf Nodes After Pruning is : 4


 3. Decision Boundary after Pruning: 


  



For testing the decision tree:


1. Extract zip file
1. OPEN cmd (or terminal if Linux or MAC OSX)
2. GO to the path where the folder is saved
3. RUN command: python executeDT.py
4. Give input of file name containing csv file for test data


INTERPRETATION:
A:  BEFORE PRUNING :


1. Prints the tree




------------------------BEFORE PRUNING---------------------------------
 ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.25143 CLASS :  0
    ATTRIBUTE SPLIT :  1 ATTRIBUTE SPLIT VALUE :  0.29935 CLASS :  0
       ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.10739 CLASS :  0
          **LEAF NODE** CLASS : 3
          ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.108765 CLASS :  0
             **LEAF NODE** CLASS : 4
             **LEAF NODE** CLASS : 3
       ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.099544 CLASS :  0
          **LEAF NODE** CLASS : 2
          ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.10125 CLASS :  0
             **LEAF NODE** CLASS : 4
             **LEAF NODE** CLASS : 2
    ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.7854 CLASS :  0
       ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.7355700000000001 CLASS :  0
          **LEAF NODE** CLASS : 4
          ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.74132 CLASS :  0
             **LEAF NODE** CLASS : 1
             **LEAF NODE** CLASS : 4
       **LEAF NODE** CLASS : 1


2. Performance Metrics 
1. Total accuracy (Overall recognition Rate)
2. Mean Per class Accuracy
3. Confusion Matrix
4. Profit Obtained


--------------------TOTAL ACCURACY (OVERALL RECOGNITION RATE)-------------------------------


Overall Accuracy (Recognition Rate) is :  95.0 %




-------------------------- MEAN PER CLASS ACCURACY -------------------------------


Mean per Class Accuracy is :  93.75






--------------------------CONFUSION MATRIX-------------------------------


                                 ACTUAL                                                                 


ASSIGNED         CLASS 1        CLASS 2        CLASS 3        CLASS 4        TOTAL
--------------------------------------------------------------------------------------------------------------------------------
CLASS 1        5                0                 0                1                6
--------------------------------------------------------------------------------------------------------------------------------


CLASS 2        0                 6                 0                0                6
--------------------------------------------------------------------------------------------------------------------------------


CLASS 3        0                0                 5                 0                 5
--------------------------------------------------------------------------------------------------------------------------------


CLASS 4        0                 0                0                3                3
--------------------------------------------------------------------------------------------------------------------------------
TOTAL                5                6                5                4                20




-------------------------- TOTAL PROFIT -------------------------------


Total Profit is :  199


3. Prints the decision Boundary


  





B:  AFTER PRUNING :
1. Prints the tree


------------------------AFTER PRUNING---------------------------------
 ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.25143 CLASS :  0
    ATTRIBUTE SPLIT :  1 ATTRIBUTE SPLIT VALUE :  0.29935 CLASS :  0
       **LEAF NODE** CLASS : 3
       **LEAF NODE** CLASS : 2
    ATTRIBUTE SPLIT :  2 ATTRIBUTE SPLIT VALUE :  0.7854 CLASS :  0
       **LEAF NODE** CLASS : 4
       **LEAF NODE** CLASS : 1


2. Performance Metrics of the tree


--------------------TOTAL ACCURACY (OVERALL RECOGNITION RATE)-------------------------------


Overall Accuracy (Recognition Rate) is :  95.0 %


-------------------------- MEAN PER CLASS ACCURACY -------------------------------


Mean per Class Accuracy is :  93.75


--------------------------CONFUSION MATRIX-------------------------------


                                        ACTUAL                                                         


ASSIGNED         CLASS 1        CLASS 2        CLASS 3        CLASS 4        TOTAL
--------------------------------------------------------------------------------------------------------------------------------
CLASS 1                5                 0                0                1          6
--------------------------------------------------------------------------------------------------------------------------------
CLASS 2                0                6                0                0          6
--------------------------------------------------------------------------------------------------------------------------------
CLASS 3                0                 0                 5                 0          5
--------------------------------------------------------------------------------------------------------------------------------
CLASS 4                0                 0                 0                3        3
--------------------------------------------------------------------------------------------------------------------------------
TOTAL                        5                6                5                4        20




-------------------------- TOTAL PROFIT -------------------------------


Total Profit is :  199








3. Decision Boundary
  



----------------------------FOR MLP--------------------------------------------


For training the MLP:


1. Extract zip file
1. OPEN cmd (or terminal if Linux or MAC OSX)
2. GO to the path where the folder is saved
3. RUN command: python trainMLP.py
4. Give input of file name containing csv file for training data
5. It generates 5 weight files: weights0.csv, weights10.csv, weights100.csv, weights1000.csv, weights10000.csv


INTERPRETATION OF OUTPUT:


1) Enter the Filename which contains training data. 


Enter File Name: train_data.csv


2) The MLP is trained and prints the SSE vs Epoch curve:


SSE vs Epoch:
  

3) This plots the SSE vs Epoch Graph for 10,000 epochs. 




For testing the MLP:


1. Extract zip file
1. OPEN cmd (or terminal if Linux or MAC OSX)
2. GO to the path where the folder is saved
3. RUN command: python executeMLP.py
4. Give input of file name containing csv file for testing data as well as input of the weight file containing the weights after the model is trained.
5. Use one of the 5 weight files generated: weights0.csv, weights10.csv, weights100.csv, weights1000.csv, weights10000.csv




INTERPRETATION OF OUTPUT:


1) Enter the filename which contains the test data


Enter Input File Name: test_data.csv


2) Enter the filename which contains network weights. Enter one of the 5 generated weight files.
   weights0.csv, weights10.csv, weights100.csv, weights1000.csv, weights10000.csv




Enter Weights File Name: weights10000.csv


1. Performance Metrics of the MLP


--------------------TOTAL ACCURACY (OVERALL RECOGNITION RATE)-------------------------------


Overall Accuracy (Recognition Rate) is :  100 %


-------------------------- MEAN PER CLASS ACCURACY -------------------------------


Mean per Class Accuracy is :  100 %




--------------------------CONFUSION MATRIX-------------------------------


                                        ACTUAL                                                         


ASSIGNED         CLASS 1        CLASS 2        CLASS 3        CLASS 4        TOTAL
--------------------------------------------------------------------------------------------------------------------------------
CLASS 1                5                 0                0                0          5
--------------------------------------------------------------------------------------------------------------------------------
CLASS 2                0                6                0                0          6
--------------------------------------------------------------------------------------------------------------------------------
CLASS 3                0                 0                 5                 0          5
--------------------------------------------------------------------------------------------------------------------------------
CLASS 4                0                 0                 0                4        4
--------------------------------------------------------------------------------------------------------------------------------
TOTAL                        5                6                5                4        20






-------------------------- TOTAL PROFIT -------------------------------


Total Profit is :  203






2. Decision Boundary


  



                Fig: Decision Boundary for 10,000 epochs