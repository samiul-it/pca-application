import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA

data_initial = pd.read_excel("output.xlsx")

X = data_initial.iloc[:, 0:59]
Y = data_initial.Performance
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

pca = PCA(n_components=8)
pca.fit(data_initial)
data = pca.transform(data_initial)

print("Initial Data ",data_initial.shape)
print("Data After PCA Applied",data.shape)

def RandomForest():
    parameters = {'max_depth': [1, 3, 25],
                  'criterion': ['gini', 'entropy'],
                  'max_features': [0.1, 0.2, 0.3],
                  'min_samples_leaf': [1, 2, 3],
                  'min_samples_split': [2, 3, 4],
                  'n_estimators': [20, 60]}
    grid_search = RandomForestClassifier()
    grid_search = GridSearchCV(
        grid_search,
        parameters,
        cv=5,
        scoring='accuracy', n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)
    print('Best Params: ', grid_result.best_params_)
    print('Best Score: ', grid_result.best_score_)


# RandomForest()

def dtree_grid_search(X, y, nfolds):  # Decision Tree

    param_grid = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                  'min_samples_split': np.arange(2, 10), 'max_depth': np.arange(2, 15),
                  'random_state': np.arange(1, 10)}
    dtree_model = DecisionTreeClassifier()
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    dtree_gscv.fit(X, y)

    return dtree_gscv.best_score_, dtree_gscv.best_params_

# print_decisiontree_result=dtree_grid_search(X,Y,5)
# print("Best Score and Pramas",print_decisiontree_result)
##DECISION TREE

def KNN():
    parameters = {'n_neighbors': [1, 3, 6, 9, 20, 23, 25, 28, 36, 40]}
    grid_search = KNeighborsClassifier()
    grid_search = GridSearchCV(
        grid_search,
        parameters,
        cv=5,
        scoring='accuracy', n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)
    print('Best Params: ', grid_result.best_params_)
    print('Best Score: ', grid_result.best_score_)


# KNN()


estimator = []
estimator.append(('LR',LogisticRegression()))
estimator.append(('SVC', SVC()))
# estimator.append(('DTC', DecisionTreeClassifier()))


parameters = {'LR__C': [1.0, 100.0], #Parameter for VOTINGCLASSIFIER
      'SVC__C': [0.1,1.5,2,3,4]}
def votingwithgrid(votingType):
    grid_search = VotingClassifier(estimators=estimator, voting=votingType)
    grid_search = GridSearchCV(
        grid_search,
        parameters,
        cv=5,
        scoring='accuracy', n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)
    print('Best Params: ', grid_result.best_params_)
    print('Best Score: ', grid_result.best_score_)

# votingwithgrid(votingType='hard')

def FindSVMforBoth():
    def SVMafterApplyinOVR():
        OvR_clf = OneVsRestClassifier(svm.SVC())
        OvR_clf.fit(X_train, y_train)

        y_pred = OvR_clf.predict(X_test)

        # print('Accuracy of OvR Classifier: {:.2f}'.format(accuracy_score(y_test, y_pred)))

        tuned_parameters = [{'estimator__C': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}]

        OvR_clf = OneVsRestClassifier(svm.SVC())

        grid = GridSearchCV(OvR_clf, tuned_parameters, cv=3, scoring='accuracy')

        grid.fit(X_train, y_train)
        print("Best Score after Applying OVR in SVM", grid.best_score_)
        print("Best Param", grid.best_params_)
        grid_predictions = grid.predict(X_test)

        print('Accuracy OVR: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))

    def SVMafterApplyinOVO():
        OvO_clf = OneVsOneClassifier(svm.SVC())
        OvO_clf.fit(X_train, y_train)

        y_pred = OvO_clf.predict(X_test)

        # print('Accuracy of OvR Classifier: {:.2f}'.format(accuracy_score(y_test, y_pred)))

        tuned_parameters = [{'estimator__C': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}]

        OvO_clf = OneVsRestClassifier(svm.SVC())

        grid = GridSearchCV(OvO_clf, tuned_parameters, cv=3, scoring='accuracy')

        grid.fit(X_train, y_train)
        print("Best Score after Applying OVO in SVM", grid.best_score_)
        grid_predictions = grid.predict(X_test)

        print('Accuracy OVO: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))
    SVMafterApplyinOVR()
    SVMafterApplyinOVO()

# FindSVMforBoth()

def FindLOGISTICforBoth():
    def LOGISTICafterApplyinOVR():
        OvR_clf = OneVsRestClassifier(LogisticRegression())
        OvR_clf.fit(X_train, y_train)

        y_pred = OvR_clf.predict(X_test)

        # print('Accuracy of OvR Classifier: {:.2f}'.format(accuracy_score(y_test, y_pred)))

        tuned_parameters = [{'estimator__C': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}]

        OvR_clf = OneVsRestClassifier(LogisticRegression())

        grid = GridSearchCV(OvR_clf, tuned_parameters, cv=3, scoring='accuracy')

        grid.fit(X_train, y_train)
        print("Best Score after Applying OVR in LOGISTIC REGRESSION", grid.best_score_)
        grid_predictions = grid.predict(X_test)

        print('Accuracy OVR: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))

    def LOGISTICafterApplyinOVO():
        OvO_clf = OneVsOneClassifier(LogisticRegression())
        OvO_clf.fit(X_train, y_train)

        y_pred = OvO_clf.predict(X_test)

        # print('Accuracy of OvR Classifier: {:.2f}'.format(accuracy_score(y_test, y_pred)))

        tuned_parameters = [{'estimator__C': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}]

        OvO_clf = OneVsOneClassifier(LogisticRegression())

        grid = GridSearchCV(OvO_clf, tuned_parameters, cv=3, scoring='accuracy')

        grid.fit(X_train, y_train)
        print("Best Score after Applying OVO in LOGISTIC REGRESSION", grid.best_score_)
        print("Best Param", grid.best_params_)
        grid_predictions = grid.predict(X_test)

        print('Accuracy OVO: {:.2f}'.format(accuracy_score(y_test, grid_predictions)))
    LOGISTICafterApplyinOVR()
    LOGISTICafterApplyinOVO()

FindLOGISTICforBoth()





