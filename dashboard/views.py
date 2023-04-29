from django.shortcuts import render
from dashboard.forms import *

# Create your views here.
def about(request):
    return render(request, 'about.html', {})

def demo(request):
    return render(request, 'demo.html', {})

def antibiogram(request):
    form = FastaUploadForm()
    if request.method == 'POST':
        form = FastaUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # process the fasta file and return the predictions
            form = FastaUploadForm()
            print('hello')
        else:
            form = FastaUploadForm()
    return render(request, 'antibiogram.html', {'form': form})



def modeldetails(request):
    return render(request, 'modeldetails.html', {})