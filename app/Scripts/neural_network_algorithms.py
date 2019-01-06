from sklearn.neural_network import MLPClassifier
from app.Scripts.helper import *
import os
from PhanLoaiNam.settings import PROJECT_ROOT
import pickle
from sklearn.externals import joblib


def nn_algorithms(input):
    # calculate performing time
    file = os.path.join(PROJECT_ROOT, "app\\Scripts\\agaricus-lepiota.data.txt")
    data_frame = clean_data(file)
    X_train, Y_train, X_test, Y_test = X_train_Y_train_X_test_Y_test(data_frame)
    x_test = pd.DataFrame([input],
                          columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                   'gill-attachment',
                                   'gill-spacing',
                                   'gill-size', 'gill-color', 'stalk-shape',
                                   'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                   'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                                   'veil-color', 'ring-number', 'ring-type',
                                   'spore-print-color', 'population', 'habitat'], dtype=int)

    # Get model
    file_model = os.path.join(PROJECT_ROOT, "app\\Scripts\\model\\nn_model.sav")
    exist_model = os.path.isfile(os.path.join(PROJECT_ROOT, "app\\Scripts\\model\\nn_model.sav"))
    if exist_model:
        clf = joblib.load(file_model)
    else:
        # call model and fit model and training data
        clf = MLPClassifier().fit(X_train, Y_train)
        pickle.dump(clf, open(file_model, 'wb'))

    # predict test data
    output = clf.predict(x_test)
    precision = clf.score(X_test, Y_test)

    return output, precision
