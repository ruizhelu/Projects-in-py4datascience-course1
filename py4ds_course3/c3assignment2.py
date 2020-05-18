import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);


# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
part1_scatter()

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures


    # Your code here

    result = []

    for i, degree in enumerate([1,3,6,9]):
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x.reshape(15,1))
        X_train, X_test, y_train, y_test = train_test_split(x_poly, y, random_state = 0)
        linreg = LinearRegression().fit(X_train, y_train)
        y_i = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)));
        result.append(y_i)

    return result

# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

plot_one(answer_one())

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # Your code here

    r2_train = []
    r2_test = []

    for i in range(10):
        poly = PolynomialFeatures(degree=i)

        x_poly = poly.fit_transform(x.reshape(15,1))
        X_train, X_test, y_train, y_test = train_test_split(x_poly, y, random_state = 0)

        linreg = LinearRegression().fit(X_train, y_train)
        r2_train.append(linreg.score(X_train, y_train))
        r2_test.append(linreg.score(X_test, y_test))

    return (r2_train, r2_test)

def answer_three():

    # Your code here

    import matplotlib.pyplot as plt
    #from scipy.interpolate import spline
    (r2_train, r2_test) = answer_two()

    #In this plot we will see clear underfitting, overfitting and good_generalization degrees respectively

    #%matplotlib notebook
    #plt.figure(figsize=(10,5))
    #T = np.array(range(10))
    #xnew = np.linspace(T.min(), T.max(), 500)
    #r2_train_smooth = spline(T, r2_train, xnew)
    #r2_test_smooth = spline(T, r2_test, xnew)
    #plt.plot(xnew, r2_train_smooth, alpha=0.8, lw=2, label='training data', markersize=10)
    #plt.plot(xnew, r2_test_smooth, alpha=0.8, lw=2, label='test data', markersize=10)
    #plt.ylim(-1,2.5)
    #plt.legend(loc=4)

    df = pd.DataFrame({'training_score':r2_train, 'test_score':r2_test})
    df['diff'] = df['training_score'] - df['test_score']
    df = df.sort_values(['training_score'])
    underfitting = df.index[0]
    df = df.sort_values(['diff'], ascending=False)
    overfitting = df.index[0]
    df = df.sort_values(['diff'])
    good_generalization = df.index[0]

    return (underfitting, overfitting, good_generalization)

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Your code here

    poly = PolynomialFeatures(degree=12)
    x_poly = poly.fit_transform(x.reshape(15,1))
    X_train, X_test, y_train, y_test = train_test_split(x_poly, y, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    LinearRegression_R2_test_score = linreg.score(X_test, y_test)

    linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_train, y_train)
    Lasso_R2_test_score = linlasso.score(X_test, y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('readonly/mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    # Your code here

    clf = DecisionTreeClassifier().fit(X_train2, y_train2)
    features = []
    for index, importance in enumerate(clf.feature_importances_):
        features.append([importance, X_train2.columns[index]])
    features.sort(reverse=True)
    features = features[0:5]
    features_names = []
    for i in range(5):
        features_names.append(features[i][1])
    return features_names

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Your code here

    param_range = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(kernel='rbf', C=1, random_state=0),
                            X_subset, y_subset,
                            param_name='gamma',
                            param_range=param_range,
                            scoring='accuracy')

    train_scores = train_scores.mean(axis=1)
    test_scores = test_scores.mean(axis=1)

    return (train_scores, test_scores)

def answer_seven():

    # Your code here

    import matplotlib.pyplot as plt
    (train_scores, test_scores) = answer_six()
    param_range = np.logspace(-4,1,6)
    plt.figure()

    plt.semilogx(param_range, train_scores, label='Train score',
            color='darkorange', lw=2)

    plt.semilogx(param_range, test_scores, label='Test score',
            color='navy', lw=2)

    plt.legend(loc='best')
    #plt.show()

    #Result comes from plots
    return (0.001, 10, 0.1)
