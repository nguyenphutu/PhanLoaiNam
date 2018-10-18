from sklearn import svm
from app.Scripts.helper import *
import os
from PhanLoaiNam.settings import PROJECT_ROOT

def svm_algorithm(x_test):
    file = os.path.join(PROJECT_ROOT,  "app\\Scripts\\agaricus-lepiota.data.txt")
    data_frame = clean_data(file)
    x_train, y_train = x_train_y_train(data_frame)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    x_test = pd.DataFrame([x_test],
                          columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                   'gill-attachment',
                                   'gill-spacing',
                                   'gill-size', 'gill-color', 'stalk-shape',
                                   'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                   'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                                   'veil-color', 'ring-number', 'ring-type',
                                   'spore-print-color', 'population', 'habitat'], dtype=int)
    y_pre_test = clf.predict(x_test)
    return y_pre_test
