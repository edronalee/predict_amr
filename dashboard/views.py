from django.shortcuts import render

# Create your views here.
def about(request):
    return render(request, 'about.html', {})

def demo(request):
    return render(request, 'demo.html', {})

def antibiogram(request):
    return render(request, 'antibiogram.html', {})

def modeldetails(request):
    return render(request, 'modeldetails.html', {})