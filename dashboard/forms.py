from django import forms

class FastaUploadForm(forms.Form):
    fasta_file = forms.FileField(label='Upload Fasta File', widget=forms.ClearableFileInput(attrs={'multiple': True}))
