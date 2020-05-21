import numpy as np
import pandas as pd

def answer_one():

    # Your code here

    df = pd.read_csv('fraud_data.csv')
    fraud = df[df['Class'] == 1]
    percentage = len(fraud)/len(df)

    return percentage
answer_one()

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    # Your code here

    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    accuracy_score = dummy_majority.score(X_test, y_test)
    y_dummy_predictions = dummy_majority.predict(X_test)
    recall_score = recall_score(y_test, y_dummy_predictions)

    return (accuracy_score, recall_score)
answer_two()

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    # Your code here

    svm = SVC().fit(X_train, y_train)
    svm_predicted = svm.predict(X_test)
    accuracy_score = svm.score(X_test, y_test)
    recall_score = recall_score(y_test, svm_predicted)
    precision_score = precision_score(y_test, svm_predicted)

    return (accuracy_score, recall_score, precision_score)
answer_three()

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Your code here

    clf = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)
    clf_predicted = (clf.decision_function(X_test) > -220)
    confusion = confusion_matrix(y_test, clf_predicted)

    return confusion
answer_four()

def answer_five():

    # Your code here

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    import matplotlib.pyplot as plt

    lr = LogisticRegression().fit(X_train, y_train)
    lr_predicted = lr.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, lr_predicted)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_predicted)

    %matplotlib notebook
    plt.figure(figsize=(7,7))
    plt.subplot(121)
    plt.plot(precision, recall)
    plt.plot(precision, recall)
    plt.vlines(0.75, 0, 1)
    plt.xlabel('Precision')
    plt.ylabel('Recall')

    plt.subplot(122)
    plt.plot(fpr_lr, tpr_lr)
    plt.vlines(0.16, 0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    return (0.84, 0.93)
answer_five()

def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Your code here

    lr = LogisticRegression()
    grid_params = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
    grid_recall = GridSearchCV(lr, param_grid=grid_params, scoring='recall')
    grid_recall.fit(X_train, y_train)
    cv_results = grid_recall.cv_results_
    results = cv_results['mean_test_score']
    results = results.reshape(5,2)

    return results
answer_six()

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    %matplotlib notebook
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

GridSearch_Heatmap(answer_six())
