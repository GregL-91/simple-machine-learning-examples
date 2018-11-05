import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import graphviz
import numpy as np


# Import the adult.txt file into Python
data = pd.read_csv('adults.txt', sep=',')

# DO NOT WORRY ABOUT THE FOLLOWING 2 LINES OF CODE
# Convert the string labels to numeric labels
for label in ['race', 'occupation']:
    data[label] = LabelEncoder().fit_transform(data[label])

# Take the fields of interest and plug them into variable X
X = data[['race', 'hours_per_week', 'occupation']]
# Make sure to provide the corresponding truth value
Y = data['sex'].values.tolist()

# Split the data into test and training (30% for test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Instantiate the classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier using the train data
clf = clf.fit(X_train, Y_train)

# Validate the classifier
accuracy = clf.score(X_test, Y_test)
print('Accuracy: ' + str(accuracy))

# Make a confusion matrix
prediction = clf.predict(X_test)

cm = confusion_matrix(prediction, Y_test)
print(cm)

#stampo l'albero 
target_name=np.array(['female','male'])
feature_name=np.array(['race','hours_per_week','occupation'])
dot_data=tree.export_graphviz(clf, out_file=None, 
                           feature_names=feature_name,
                           class_names=target_name,
                         filled=True, rounded=True,  
                         special_characters=True) 
graf=graphviz.Source(dot_data)
graf.render(filename="alberoGenerato")

