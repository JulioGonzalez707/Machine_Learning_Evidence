import pandas as pd 

dataset = pd.read_csv(r'C:\Users\julio\OneDrive - Universidad Politécnica de Yucatán\Machine_Learning\Unit_1\Social_Network_Ads.csv', sep = ",")

#print(dataset)

dataset = dataset.drop(columns = "User ID")

#print(dataset)

#dataset = dataset.add(columns = "Male")
#dataset = dataset.add(columns = "Female")
dataset['Male'] = (dataset["Gender"] == "Male").astype(int)
dataset['Female'] = (dataset["Gender"] == "Female").astype(int)

dataset = dataset.drop(columns = "Gender")
columns_order = list(dataset.columns.difference(["Male", "Female", "Purchased"])) + ["Male","Female"] + ["Purchased"]
dataset = dataset[columns_order]
#print(dataset.head(5))

from sklearn.linear_model import Perceptron

xtrain = dataset.iloc[:319, 0:4]
ytrain = dataset.iloc[:319, 4]
xtest = dataset.iloc[320:, 0:4]
ytest = dataset.iloc[320:, 4]

#print(xtrain.head(1))
#print(ytrain.head(1))
#print(xtest.head(1))
#print(ytest.head(1))

#X, y = load_digits(return_X_y=True)

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(xtrain, ytrain)
Perceptron()
#print(clf.score(xtrain, ytrain))
print(clf.score(xtest, ytest))



    
 
