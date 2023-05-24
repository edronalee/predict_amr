from django.shortcuts import render
from dashboard.forms import *
import subprocess

import os
from django.conf import settings

#Import and load the ML model
import joblib
import pickle
cefotaxime_svm_model = joblib.load('cefotaxime_svm_model.pkl')
ceftriaxone_rf_model = joblib.load('ceftriaxone_rf_model.pkl')
ciprofloxacin_xgboost_model = joblib.load('ciprofloxacin_xgboost_model.pkl')
gentamicin_xgboost_model = joblib.load('gentamicin_xgboost_model.pkl')
levofloxacin_xgboost_model = joblib.load('levofloxacin_xgboost_model.pkl')

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, matthews_corrcoef, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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

            # Run IntegronFinder
            subprocess.check_call(['integron_finder', '--local-max', filename], cwd = '/home/vboxuser/predict_amr/media')
            
            # Check if IntegronFinder found an integron
            integron_file_path = '/home/vboxuser/predict_amr/media/Results_Integron_Finder_' + os.path.splitext(filename)[0] + '/' + os.path.splitext(filename)[0] + '.integrons'
            with open(integron_file_path, 'r') as integron_file:
                integron_lines = integron_file.readlines()
                if len(integron_lines) > 1 and integron_lines[1].startswith("# No Integron found"):
                    integron_presence = 0  # Integron not found
                else:
                    integron_presence = 1  # Integron found

            # Create a DataFrame for integron presence
            integron_df = pd.DataFrame({'Integron_Presence': [integron_presence]})

            # Run Abricate
            abricate_output = subprocess.check_output(['./abricate/bin/abricate', '--db', 'resfinder', '/home/vboxuser/predict_amr/media/'+filename], cwd = '/home/vboxuser')
            abricate_output = abricate_output.decode('utf-8')

            abricate_lines = abricate_output.strip().split('\n')
            abricate_gene_products = set(abricate_line.split()[13] for abricate_line in abricate_lines[1:])

            for abricate_gene_product in abricate_gene_products:
                print(abricate_gene_product)
                
            # Create a feature vector using one-hot encoding
            mlb = MultiLabelBinarizer()
            encoded_features = mlb.fit_transform([abricate_gene_products])

            # Convert the encoded features to a DataFrame if needed
            features_df = pd.DataFrame(encoded_features, columns=mlb.classes_)
            # print(features_df)

            # Concatenate integron presence DataFrame with features DataFrame
            processed_data = pd.concat([features_df, integron_df], axis=1)
            # print(processed_data)

            # Get the list of AMR genes used during training from the SVM model excluding the integron_presence column
            amr_genes = [
                        "ARR-3", "aac(3)-IIa", "aac(3)-IId", "aac(3)-Ia", "aac(6')-31", "aac(6')-IIa", "aac(6')-Iaf", "aac(6')-Ib", "aac(6')-Ib-cr", "aadA", "aadA1", "aadA2", "aadA4", "aadA5", "aadA6",
                        "ant(2'')-Ia", "ant(3'')-Ia", "ant(3'')-Ii-aac(6')-IId", "aph(3'')-Ib", "aph(3')-Ia", "aph(3')-VI", "aph(3')-VIa", "aph(6)-Id", "armA", "blaADC-25", "blaCMY-2",
                        "blaCMY-4", "blaCTX-M-1", "blaCTX-M-122", "blaCTX-M-139", "blaCTX-M-14", "blaCTX-M-14b", "blaCTX-M-15", "blaCTX-M-2", "blaCTX-M-27", "blaCTX-M-3", "blaCTX-M-55",
                        "blaGES-14", "blaIMP-1", "blaIMP-19", "blaIMP-68", "blaKPC-3", "blaNDM-1", "blaNDM-4", "blaNDM-5", "blaOXA-1", "blaOXA-10", "blaOXA-100", "blaOXA-106", "blaOXA-120",
                        "blaOXA-121", "blaOXA-124", "blaOXA-126", "blaOXA-2", "blaOXA-208", "blaOXA-23", "blaOXA-235", "blaOXA-314", "blaOXA-407", "blaOXA-430", "blaOXA-510", "blaOXA-64",
                        "blaOXA-65", "blaOXA-66", "blaOXA-69", "blaOXA-9", "blaOXA-90", "blaOXA-94", "blaOXA-98", "blaSHV-101", "blaSHV-106", "blaSHV-108", "blaSHV-11", "blaSHV-110",
                        "blaSHV-12",  "blaSHV-14", "blaSHV-182", "blaSHV-187", "blaSHV-27", "blaSHV-30", "blaSHV-33", "blaSHV-76", "blaTEM-141", "blaTEM-1A", "blaTEM-1B", "blaTEM-1D", "blaVIM-1",
                        "blaVIM-11", "catA1", "catB8", "cmlA1", "dfrA1", "dfrA12", "dfrA14", "dfrA15", "dfrA17", "dfrA26", "dfrA5", "erm(B)", "floR", "fosA", "fosA6", "fosA7", "mdf(A)", "mph(A)",
                        "mph(E)", "msr(E)", "npmA", "oqxA", "oqxB", "qepA1", "qnrA1", "qnrB1", "qnrB19", "qnrB4", "qnrB6", "qnrB9", "qnrS1", "rmtB", "rmtC", "rmtF", "sul1", "sul2", "tet(A)", "tet(B)"
                        ]
            
            # Initialize an empty dictionary to store the feature values
            feature_values = {}

            # Iterate over each AMR gene and check if it is present in the input fasta file
            for amr_gene in amr_genes:
                if amr_gene in abricate_gene_products:
                    feature_values[amr_gene] = 1  # Gene is present
                else:
                    feature_values[amr_gene] = 0  # Gene is missing

            # Create a DataFrame from the feature values
            features_df = pd.DataFrame([feature_values])

            # Add the integron presence feature to the DataFrame
            features_df['Integron_Presence'] = integron_presence

            # Load the ML models
            cefotaxime_svm_model = joblib.load('cefotaxime_svm_model.pkl')
            ceftriaxone_rf_model = joblib.load('ceftriaxone_rf_model.pkl')
            ciprofloxacin_xgboost_model = joblib.load('ciprofloxacin_xgboost_model.pkl')
            gentamicin_xgboost_model = joblib.load('gentamicin_xgboost_model.pkl')
            levofloxacin_xgboost_model = joblib.load('levofloxacin_xgboost_model.pkl')

            # Perform prediction using the models
            cefotaxime_prediction = cefotaxime_svm_model.predict(features_df)
            ceftriaxone_prediction = ceftriaxone_rf_model.predict(features_df)
            ciprofloxacin_prediction = ciprofloxacin_xgboost_model.predict(features_df)
            gentamicin_prediction = gentamicin_xgboost_model.predict(features_df)
            levofloxacin_prediction = levofloxacin_xgboost_model.predict(features_df)

            context = {'cefotaxime_prediction':cefotaxime_prediction, 'ceftriaxone_prediction':ceftriaxone_prediction,
                        'ciprofloxacin_prediction':ciprofloxacin_prediction, 'gentamicin_prediction':gentamicin_prediction,
                        'levofloxacin_prediction':levofloxacin_prediction, 'integron_presence':integron_presence, 'form': form,
                        'abricate_gene_products':abricate_gene_products, 'filename':filename}
            # Render the template with the predictions
            return render(request, 'antibiogram.html', context)
    return render(request, 'antibiogram.html', {'form': form})

def modeldetails(request):
    # Load the test data
    features_data = pd.read_excel('processed_data.xlsx')
    target = pd.read_excel('combined_dataset.xlsx')

    cefotaxime_target = target['Cefotaxime_Resistance']
    ceftriaxone_target = target['Ceftriaxone_Resistance']
    ciprofloxacin_target = target['Ciprofloxacin_Resistance'] 
    gentamicin_target = target['Gentamicin_Resistance']
    levofloxacin_target = target['Levofloxacin_Resistance']

    # ----- Cefotaxime -----
    # Split the data into training and testing sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(features_data, cefotaxime_target, test_size=0.3, random_state=42)

    # Make predictions on the test set using the loaded model
    y_pred1 = cefotaxime_svm_model.predict(X_test1)

    # Calculate the metrics using the test data and predictions
    accuracy1 = accuracy_score(y_test1, y_pred1)
    precision1 = precision_score(y_test1, y_pred1)
    recall1 = recall_score(y_test1, y_pred1)

    # Calculate the ROC curve and AUC
    fpr1, tpr1, thresholds1 = roc_curve(y_test1, y_pred1)
    auc1 = roc_auc_score(y_test1, y_pred1)

    # Calculate the MCC
    mcc1 = matthews_corrcoef(y_test1, y_pred1)

    # Total number of samples in the dataset
    num_resistant1 = target['Cefotaxime_Resistance'].value_counts()[1]
    num_susceptible1 = target['Cefotaxime_Resistance'].value_counts()[0]
    total_samples1 = len(cefotaxime_target)

    # Number of samples in the training set
    num_train_samples1 = len(X_train1)

    # Number of samples in the testing set
    num_test_samples1 = len(X_test1)


    # Plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr1, tpr1, label='ROC curve (area = %.2f)' % auc1)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)
    ax.legend()
    plt.savefig('static/cefotaxime_roc_curve.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure

    # Generate confusion matrix
    cm1 = confusion_matrix(y_test1, y_pred1)
    fig, ax = plt.subplots()
    sns.heatmap(cm1, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig('static/cefotaxime_confusion_matrix.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure


    # ----- Ceftriaxone -----
    # Split the data into training and testing sets
    X_train2, X_test2, y_train2, y_test2 = train_test_split(features_data, ceftriaxone_target, test_size=0.3, random_state=42)

    # Make predictions on the test set using the loaded model
    y_pred2 = ceftriaxone_rf_model.predict(X_test2)

    # Calculate the metrics using the test data and predictions
    accuracy2 = accuracy_score(y_test2, y_pred2)
    precision2 = precision_score(y_test2, y_pred2)
    recall2 = recall_score(y_test2, y_pred2)

    # Calculate the ROC curve and AUC
    fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_pred2)
    auc2 = roc_auc_score(y_test2, y_pred2)

    # Calculate the MCC
    mcc2 = matthews_corrcoef(y_test2, y_pred2)

    # Total number of samples in the dataset
    num_resistant2 = target['Ceftriaxone_Resistance'].value_counts()[1]
    num_susceptible2 = target['Ceftriaxone_Resistance'].value_counts()[0]
    total_samples2 = len(ceftriaxone_target)

    # Number of samples in the training set
    num_train_samples2 = len(X_train2)

    # Number of samples in the testing set
    num_test_samples2 = len(X_test2)

    # Plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr2, tpr2, label='ROC curve (area = %.2f)' % auc2)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)
    ax.legend()
    plt.savefig('static/ceftriaxone_roc_curve.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure

    # Generate confusion matrix
    cm2 = confusion_matrix(y_test2, y_pred2)
    fig, ax = plt.subplots()
    sns.heatmap(cm2, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig('static/ceftriaxone_confusion_matrix.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure


    # ----- Ciprofloxacin -----
    # Split the data into training and testing sets
    X_train3, X_test3, y_train3, y_test3 = train_test_split(features_data, ciprofloxacin_target, test_size=0.3, random_state=4)

    # Make predictions on the test set using the loaded model
    y_pred3 = ciprofloxacin_xgboost_model.predict(X_test3)

    # Calculate the metrics using the test data and predictions
    accuracy3 = accuracy_score(y_test3, y_pred3)
    precision3 = precision_score(y_test3, y_pred3)
    recall3 = recall_score(y_test3, y_pred3)

    # Calculate the ROC curve and AUC
    fpr3, tpr3, thresholds3 = roc_curve(y_test3, y_pred3)
    auc3 = roc_auc_score(y_test3, y_pred3)

    # Calculate the MCC
    mcc3 = matthews_corrcoef(y_test3, y_pred3)
   
    # Total number of samples in the dataset
    num_resistant3 = target['Ciprofloxacin_Resistance'].value_counts()[1]
    num_susceptible3 = target['Ciprofloxacin_Resistance'].value_counts()[0]
    total_samples3 = len(ciprofloxacin_target)

    # Number of samples in the training set
    num_train_samples3 = len(X_train3)

    # Number of samples in the testing set
    num_test_samples3 = len(X_test3)

    # Plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr3, tpr3, label='ROC curve (area = %.2f)' % auc3)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)
    ax.legend()
    plt.savefig('static/ciprofloxacin_roc_curve.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure

    # Generate confusion matrix
    cm3 = confusion_matrix(y_test3, y_pred3)
    fig, ax = plt.subplots()
    sns.heatmap(cm3, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig('static/ciprofloxacin_confusion_matrix.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure


    # ----- Gentamicin -----
    # Split the data into training and testing sets
    X_train4, X_test4, y_train4, y_test4 = train_test_split(features_data, gentamicin_target, test_size=0.3, random_state=3)

    # Make predictions on the test set using the loaded model
    y_pred4 = gentamicin_xgboost_model.predict(X_test4)

    # Calculate the metrics using the test data and predictions
    accuracy4 = accuracy_score(y_test4, y_pred4)
    precision4 = precision_score(y_test4, y_pred4)
    recall4 = recall_score(y_test4, y_pred4)

    # Calculate the ROC curve and AUC
    fpr4, tpr4, thresholds4 = roc_curve(y_test4, y_pred4)
    auc4 = roc_auc_score(y_test4, y_pred4)

    # Calculate the MCC
    mcc4 = matthews_corrcoef(y_test4, y_pred4)

    # Total number of samples in the dataset
    num_resistant4 = target['Gentamicin_Resistance'].value_counts()[1]
    num_susceptible4 = target['Gentamicin_Resistance'].value_counts()[0]
    total_samples4 = len(gentamicin_target)

    # Number of samples in the training set
    num_train_samples4 = len(X_train4)

    # Number of samples in the testing set
    num_test_samples4 = len(X_test4)

    # Plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr4, tpr4, label='ROC curve (area = %.2f)' % auc4)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)
    ax.legend()
    plt.savefig('static/gentamicin_roc_curve.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure

    # Generate confusion matrix
    cm4 = confusion_matrix(y_test4, y_pred4)
    fig, ax = plt.subplots()
    sns.heatmap(cm4, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig('static/gentamicin_confusion_matrix.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure


    # ----- Levofloxacin -----
    # Split the data into training and testing sets
    X_train5, X_test5, y_train5, y_test5 = train_test_split(features_data, levofloxacin_target, test_size=0.3, random_state=42)

    # Make predictions on the test set using the loaded model
    y_pred5 = levofloxacin_xgboost_model.predict(X_test5)

    # Calculate the metrics using the test data and predictions
    accuracy5 = accuracy_score(y_test5, y_pred5)
    precision5 = precision_score(y_test5, y_pred5)
    recall5 = recall_score(y_test5, y_pred5)

    # Calculate the ROC curve and AUC
    fpr5, tpr5, thresholds5 = roc_curve(y_test5, y_pred5)
    auc5 = roc_auc_score(y_test5, y_pred5)

    # Calculate the MCC
    mcc5 = matthews_corrcoef(y_test5, y_pred5)

    # Total number of samples in the dataset
    num_resistant5 = target['Levofloxacin_Resistance'].value_counts()[1]
    num_susceptible5 = target['Levofloxacin_Resistance'].value_counts()[0]
    total_samples5 = len(levofloxacin_target)

    # Number of samples in the training set
    num_train_samples5 = len(X_train5)

    # Number of samples in the testing set
    num_test_samples5 = len(X_test5)

    # Plot the ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr5, tpr5, label='ROC curve (area = %.2f)' % auc5)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)
    ax.legend()
    plt.savefig('static/levofloxacin_roc_curve.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure

    # Generate confusion matrix
    cm5 = confusion_matrix(y_test5, y_pred5)
    fig, ax = plt.subplots()
    sns.heatmap(cm5, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig('static/levofloxacin_confusion_matrix.png')  # Save the figure as a file
    plt.close(fig)  # Close the figure


    # Pass the metrics scores to the template
    context = {
        'accuracy1': accuracy1, 'precision1': precision1, 'recall1': recall1, 'mcc1':mcc1, 'auc1':auc1, 'num_test_samples1':num_test_samples1, 'num_train_samples1':num_train_samples1, 'num_resistant1':num_resistant1,'num_susceptible1':num_susceptible1, 'total_samples1':total_samples1,
        'accuracy2': accuracy2, 'precision2': precision2, 'recall2': recall2, 'mcc2':mcc2, 'auc2':auc2, 'num_test_samples2':num_test_samples2, 'num_train_samples2':num_train_samples2, 'num_resistant2':num_resistant2,'num_susceptible2':num_susceptible2, 'total_samples2':total_samples2,
        'accuracy3': accuracy3, 'precision3': precision3, 'recall3': recall3, 'mcc3':mcc3, 'auc3':auc3, 'num_test_samples3':num_test_samples3, 'num_train_samples3':num_train_samples3, 'num_resistant3':num_resistant3,'num_susceptible3':num_susceptible3, 'total_samples3':total_samples3,
        'accuracy4': accuracy4, 'precision4': precision4, 'recall4': recall4, 'mcc4':mcc4, 'auc4':auc4, 'num_test_samples4':num_test_samples4, 'num_train_samples4':num_train_samples4, 'num_resistant4':num_resistant4,'num_susceptible4':num_susceptible4, 'total_samples4':total_samples4,
        'accuracy5': accuracy5, 'precision5': precision5, 'recall5': recall5, 'mcc5':mcc5, 'auc5':auc5, 'num_test_samples5':num_test_samples5, 'num_train_samples5':num_train_samples5, 'num_resistant5':num_resistant5,'num_susceptible5':num_susceptible5, 'total_samples5':total_samples5
    }
    return render(request, 'modeldetails.html', context)

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