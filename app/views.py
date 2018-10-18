from django.shortcuts import render, HttpResponse
from app.Scripts.svm_algorithms import svm_algorithm


def index(request):
    if request.method == 'POST':
        dict_value = dict(request.POST.items())
        x_test = [int(x) for x in list(dict_value.values())[1:]]
        result = svm_algorithm(x_test)
        return render(request, 'app/result.html', {"result": int(result)})
    return render(request, 'app/index.html')
