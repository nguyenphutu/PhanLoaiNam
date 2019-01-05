from sklearn.linear_model import LogisticRegression
from app.Scripts.helper import *
import os
from PhanLoaiNam.settings import PROJECT_ROOT


def lr_algorithms(input):
    file = os.path.join(PROJECT_ROOT, "app\\Scripts\\agaricus-lepiota.data.txt")
    data_frame = clean_data(file)
    x_test = pd.DataFrame([input],
                          columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                   'gill-attachment',
                                   'gill-spacing',
                                   'gill-size', 'gill-color', 'stalk-shape',
                                   'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                   'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                                   'veil-color', 'ring-number', 'ring-type',
                                   'spore-print-color', 'population', 'habitat'], dtype=int)

    X_train, Y_train, X_test, Y_test = X_train_Y_train_X_test_Y_test(data_frame)

    # call model and fit model and training data
    clf_lr = LogisticRegression().fit(X_train, Y_train)
    # predict test data
    output = clf_lr.predict(x_test)
    precision = clf_lr.score(X_test, Y_test)

    return output, precision
