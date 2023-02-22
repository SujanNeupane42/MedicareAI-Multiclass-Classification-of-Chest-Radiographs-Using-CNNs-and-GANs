from django.db import models

# Create your models here.

## Image -> Name of the class is Image which will be the name of our database

class Image(models.Model):
    name= models.CharField(max_length=500)
    videofile= models.FileField(upload_to='images/', null=True, verbose_name="")

    def __str__(self):
        return self.name + ": " + str(self.imagefile)