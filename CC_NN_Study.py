import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,confusion_matrix, recall_score, precision_score, classification_report, average_precision_score, roc_curve, roc_auc_score, mean_squared_error, mean_absolute_error, multilabel_confusion_matrix, log_loss

def outlier(df, col):
    Q1, Q3 = np.quantile(df[col], [0.0, 0.80])
    IQR = Q3 - Q1
    min = Q1 - IQR*1.5
    max = Q3 + IQR*1.5
    return df[(df[col] >= min) & (df[col] <= max)]

def ApplyEncoder(OriginalColumn): 
    global df
    Encoder = LabelEncoder()
    Encoder.fit(df[OriginalColumn])
    return Encoder.transform(df[OriginalColumn])

def modelValidation(y_test, y_pred, startTrain =0, endTrain=0, startTest=0, endTest=0):
    print("Results")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="micro")
    rec = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    msr = mean_squared_error(y_test, y_pred)
    rmsr = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mcr = 1-acc
    trainingTime = endTrain - startTrain
    testingTime = endTest - startTest
    totalTime = trainingTime + testingTime

    print("Accuracy Score: {}".format(acc))
    print("Precision Score: {}".format(prec))
    print("Recall Score: {}".format(rec))
    print("F1 Score Score: {}".format(f1))
    print("Mean Square Error: {}".format(msr))
    print("Root Mean Square Error: {}".format(rmsr))
    print("Mean Absolute Error: {}".format(mae))
    print("Misclassification Rate: {}".format(mcr))
    print("Training Time: {}".format(trainingTime))
    print("Testing Time: {}".format(testingTime))
    print("Total Time: {}".format(totalTime))

    print()
    return(acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime)
    
def nn_pipeline(x_train, x_test, y_train, y_test, hidden_layer_size, activation, solver, max_iter, shuffle = False, random_state = 42):
    mlp = MLPClassifier(hidden_layer_size, activation=activation, solver=solver, max_iter=max_iter, shuffle=shuffle, random_state=random_state)
    
    startTrain = time.time()
    mlp.fit(x_train, y_train)
    endTrain = time.time()

    startTest = time.time()
    y_pred = mlp.predict(x_test)
    endTest = time.time()

    acc, f1, prec, msr, rmsr, mae, mcr, trainingTime, testingTime, totalTime = modelValidation(y_test, y_pred, startTrain, endTrain, startTest, endTest)
    return [acc, f1, prec, msr, rmsr, mae, mcr, trainingTime, testingTime, totalTime]
     
#testing effect of different activation function and solvers
def MLPActivationAndSolverTest():
    activation_functions = ["identity", "logistic", "tanh", "relu"]
    solvers = ["lbfgs", "sgd", "adam"]
    
    max_iteration = 1000
    hidden_layer = (32, 16)
    testedParameterAndMetrics = []
    
    for activation in activation_functions:
        for solver in solvers:
            print("Test {} activation with {} solver".format(activation, solver))
            acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime = nn_pipeline(x_train, x_test, y_train, y_test,hidden_layer,activation,solver,max_iteration)
            testedParameterAndMetrics.append([activation,solver, acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime])
            print()
            
    testedParameterAndMetrics.sort(key=lambda x:(x[2], x[3], x[4]), reverse=True)
    print(testedParameterAndMetrics[0:3]) # change to just 0 to get best parameter  
    nnScoreFrame = pd.DataFrame(testedParameterAndMetrics)  
    nnScoreFrame.columns=["Activation", "Solver", "Acc", "F1-Score", "Prec","MSR","RMSR","MAE","MCR","Training Time (s)","Testing Time (s)","Total Time (s)"]
    print(nnScoreFrame)

#testing effect of different hidden layers
def MLPNoOfHiddenLayerAndNeuron():
    activation = "tanh" # previous test shows combination of tanh and sgd produced the best scores
    solver = "sgd"
    max_layers = 3
    max_neuron = 30
    max_iteration = 1000
    testedParameterAndMetrics = []
    for noOfLayers in range(1, max_layers+1):
        testedParMet = []
        for noOfNeurons in range(1,max_neuron+1):
            print("Test {} hidden layers with {} neurons".format(noOfLayers, noOfNeurons))
            hidden_layer = tuple([noOfNeurons for noLayer in range(noOfLayers)])
            acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime = nn_pipeline(x_train, x_test, y_train, y_test,hidden_layer,activation,solver,max_iteration)
            testedParameterAndMetrics.append([noOfLayers,noOfNeurons,hidden_layer, acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime])
            testedParMet.append([noOfLayers,noOfNeurons,hidden_layer, acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime])
            print()
        x = [i for i in range(1,max_neuron+1)]
        y = [parmet[3] for parmet in testedParMet]
        plt.plot(x, y)
        plt.title("No. of Neurons Test. {} Hidden Layer".format(noOfLayers))
        plt.ylabel("Acc Score")
        plt.xlabel("No. of Neurons")
        plt.show()
    
    testedParameterAndMetrics.sort(key=lambda x:(x[3], x[4], x[5]), reverse=True)
    print(testedParameterAndMetrics[0:5]) # change to just 0 to get best parameter        
    nn_score_frame = pd.DataFrame(testedParameterAndMetrics[0:10][:])  
    nn_score_frame.columns=["No. of Hidden Layers", "No. of Neurons", "Hidden Layers", "Acc", "F1-Score", "Prec","MSR","RMSR","MAE","MCR", "Training Time (s)", "Testing Time (s)", "Total Time (s)"]
    print(nn_score_frame)

#testing effect of different max iteration
def MLPMaxIterTest():
    activation = "tanh" # previous test shows combination of tanh and sgd produced the best scores
    solver = "sgd"
    hidden_layer = (4, 4) # replace with best hidden layer neuron combination
    max_iteration = 1000
    testedParameterAndMetrics = []
    
    for max_i in range(1, max_iteration):
        print("Test {} maximum iterations".format(max_i))
        acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime = nn_pipeline(x_train, x_test, y_train, y_test,hidden_layer,activation,solver,max_i)
        testedParameterAndMetrics.append([max_i, acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime])
        print()
        
    x = [i for i in range(1,max_iteration)]
    y = [parmet[1] for parmet in testedParameterAndMetrics]
    plt.plot(x, y)
    plt.title("Max Iteration Test")
    plt.ylabel("Acc Score")
    plt.xlabel("Max No. of Iterations")
    plt.show()
    testedParameterAndMetrics.sort(key=lambda x:(x[1]), reverse=True)
    print(testedParameterAndMetrics[0:5])

    nn_score_frame = pd.DataFrame(testedParameterAndMetrics[0:10][:])
    nn_score_frame.columns=["Maximum No. of Iterations", "Acc", "F1-Score", "Prec","MSR","RMSR","MAE","MCR", "Training Time (s)", "Testing Time (s)", "Total Time (s)"]
    print(nn_score_frame)         

# testing the MLP model with the best performing parameters (found from the previous tests), performing it repeatedly to obtain mean score and observe any changes or fluctuations in scores and evaluation metric results
def BestPerformingMLPModel():
    activation = "tanh" # previous test shows combination of tanh and sgd produced the best scores
    solver = "sgd"
    hidden_layer = (4, 4) # replaced with best hidden layer neuron combination
    max_iteration = 283 # replaced with best performing max iteration value
    testedParameterAndMetrics = []
    numberOfTestRuns = 100
    
    for testRun in range(numberOfTestRuns):
        print("Test Run No. {}".format(testRun))
        acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime = nn_pipeline(x_train, x_test, y_train, y_test,hidden_layer,activation,solver,max_iteration)
        testedParameterAndMetrics.append([testRun, acc, f1, prec,msr,rmsr,mae,mcr,trainingTime,testingTime,totalTime])
        print()

    x = [i for i in range(1,len(testedParameterAndMetrics)+1)]
    y = [testedParMet[1] for testedParMet in testedParameterAndMetrics]
    
    mean_acc = sum(y)/len(y)
    print("Mean Accuracy Score: {}".format(mean_acc))

    plt.plot(x, y)
    plt.title("Accuracy for Best Performing MLP Model")
    plt.ylabel("Acc Score")
    plt.xlabel("Test Run No.")
    plt.show()
    testedParameterAndMetrics.sort(key=lambda x:(x[1]), reverse=True)
    print(testedParameterAndMetrics[0:5])

    nn_score_frame = pd.DataFrame(testedParameterAndMetrics[0:10][:])
    nn_score_frame.columns=["Test Run", "Acc", "F1-Score", "Prec","MSR","RMSR","MAE","MCR", "Training Time (s)", "Testing Time (s)", "Total Time (s)"]
    print(nn_score_frame)

if __name__ == "__main__":
    # Data Acquisition
    df = pd.read_csv('creditcard.csv')
    
    # Data Understanding
    print("Size of the UCI Credit Card Dataset: " + str(df.shape))
    print(df.dtypes) # to identify the datatypes, identify the categorical datatype values to convert to dummy variables
    
    print("Data Description")
    nullAttributes = df.isnull().sum()
    print(df.describe().transpose())
    
    print("Dataset Info")
    print(df.info())
    
    missingVals = df.isna().sum()
    print("Missing Values")
    print(missingVals) # observe missing values in the dataset # none found
    
    print("Distribution of Target")
    print(df["Approved"].value_counts()) #observe the distribution of approval

    # removing outliers in the dataset
    num_cols = df.columns[df.dtypes != 'object']
    for i in num_cols:
        df = outlier(df, i)
        
    df[df.duplicated()] #determine whether contains duplicates
    
    obj_cols = df.columns[df.dtypes == "object"]
    list(obj_cols)
    
    # Label Encoding
    encoder = LabelEncoder()
    for col in obj_cols:
        df[col] = ApplyEncoder(col)

    # data partitioning, separating target attribute (y) from rest of the factors (x)
    x = df.iloc[:,0:15]
    y = df.iloc[:, 15]
    
    # One hot encoding categorical attributes - Creating Dummy Attributes/one hot encoding
    x=pd.get_dummies(x)
    
    # oversampling with SMOTE 
    smt = SMOTE(sampling_strategy='not majority')
    x, y = smt.fit_resample(x, y)
        
    # Data Scaling - StandardScaler found to perform the best
    sc = StandardScaler()
    x = sc.fit_transform(x)
    
    # to verify effects of preprocessing and scaling
    print(df.describe().transpose())
    
    # Data splitting, train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30,shuffle=True)
    
    isMenuValid = False
    
    while not isMenuValid:
        menuInput = input("Select to test the effetcs of: \n1 - Different activation function and solvers\n2 - Different hidden layers\n3 - Maximum iterations\n4 - MLP with best performing parameters\n")
        try:
            menuN = int(menuInput)
            if (menuN <= 0 or menuN > 4):
                raise ValueError
            else:
                if menuN == 1:
                    MLPActivationAndSolverTest()
                elif menuN == 2:
                    MLPNoOfHiddenLayerAndNeuron()
                elif menuN == 3:
                    MLPMaxIterTest()
                else:
                    BestPerformingMLPModel()
                isMenuValid = True
        except ValueError:
            print("Please enter a valid selection\n")