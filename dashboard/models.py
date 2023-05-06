from django.db import models

class FastaFile(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='fasta_files/')

    def __str__(self):
        return self.title
