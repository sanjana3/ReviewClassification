from django.db import models

class ReviewTab(models.Model):
    rid = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, blank=False)
    review_data = models.CharField(max_length=1000, blank=False)
    gender = models.CharField(max_length=25, blank=True)
    rating = models.CharField(max_length=25, blank=True)