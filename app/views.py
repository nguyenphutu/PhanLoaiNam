from django.shortcuts import render
from app.Scripts.LogisticRegression_algorithms import lr_algorithms
from app.Scripts.neural_network_algorithms import nn_algorithms
from app.Scripts.svm_algorithms import svm_algorithm


def index(request):
    if request.method == 'POST':
        dict_value = dict(request.POST.items())
        data_input = [int(x) for x in list(dict_value.values())[1:23]]
        algorithm = request.POST["algorithms"]
        if algorithm == 'nn':
            result, precision = nn_algorithms(data_input)
        elif algorithm == 'lr':
            result, precision = lr_algorithms(data_input)
        else:
            result, precision = svm_algorithm(data_input)
        return render(request, 'app/result.html', {"result": int(result), "precision": round(precision*100, 2)})
    return render(request, 'app/index.html')
