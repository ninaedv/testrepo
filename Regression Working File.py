import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())  # View the data

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]    # Attributes
print(data.head())  # View the data

predict = "G3"  # Label

# All attributes and labels
X = np.array(data.drop([predict], axis=1)) # Removes predict from the data, axis=1 is the first column
y = np.array(data[predict])

# Split the attributes and labels into different training and test arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Loop used to retrain the model and save the best version of it based on the acc
'''
best = 0
for _ in range(30):
    # Split the attributes and labels into different training and test arrays
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Training model (Linear Regression)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)    # Fit the data to find the best fit line, store the line in "linear"
    acc = linear.score(x_test, y_test)    # Accuracy of our model
    print(acc)
    # We can determine what a students grade is going to be with an accuracy of acc

    if acc > best:
        best = acc
        # Saves the model (pickle file) into the file f
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f) '''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in) # Load our model into linear

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
# One line: 18.938084631109632 [18 18  1  0  8] 18
# 18.938.. is the predicted answer. Beginning grade is 18, end grade is 18, 1 hour studytime, 0 failures, 8 absences,
# and the actual grade was 18.

# Plotting
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()



