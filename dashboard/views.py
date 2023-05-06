from django.shortcuts import render
from dashboard.forms import *

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
            
            print('testinggggg')
            # Run IntegronFinder and Abricate subprocesses on the saved file
            # integron_output = subprocess.check_output(['integron_finder', '-i', filename])
            # abricate_output = subprocess.check_output(['abricate', filename])
            
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
    
    # Check if a file with the same name already exists
    #if os.path.exists(file_path):
        # If the file exists, add a suffix to the filename to make it unique
    #    filename = f'{os.path.splitext(filename)[0]}_{uuid.uuid4().hex}{os.path.splitext(filename)[1]}'
    #    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    # Write the uploaded file data to the new file
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    return filename