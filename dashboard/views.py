from django.shortcuts import render
from dashboard.forms import *
import subprocess

import os
from django.conf import settings

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
            # Save the uploaded file to the local machine
            uploaded_file = request.FILES['fasta_file']
            filename = handle_uploaded_file(uploaded_file)
            
            # print('testinggggg ' + filename)

            # Run IntegronFinder
            subprocess.check_call(['integron_finder', '--local-max', filename], cwd = '/home/vboxuser/predict_amr/media')
            
            # Run Abricate
            abricate_output = subprocess.check_output(['./abricate/bin/abricate', '--db', 'resfinder', '/home/vboxuser/predict_amr/media/'+filename], cwd = '/home/vboxuser')
            abricate_output = abricate_output.decode('utf-8')

            print(abricate_output)

            abricate_lines = abricate_output.strip().split('\n')
            abricate_gene_products = set(abricate_line.split()[13] for abricate_line in abricate_lines[1:])

            for abricate_gene_product in abricate_gene_products:
                print(abricate_gene_product)

            
            # Process the output data and return the predictions
            # predictions = process_output(integron_output, abricate_output)
            
            # Render the template with the predictions
            # return render(request, 'prediction.html', {'predictions': predictions})
    return render(request, 'antibiogram.html', {'form': form})

def modeldetails(request):
    return render(request, 'modeldetails.html', {})

def handle_uploaded_file(uploaded_file):
    """
    Save the uploaded file to the MEDIA_ROOT directory with a unique name.
    Return the filename.
    """
    filename = uploaded_file.name
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    # Write the uploaded file data to the new file
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    return filename