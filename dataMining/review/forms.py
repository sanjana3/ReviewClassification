from django import forms
from django.forms import ModelForm
from review.models import *

class ReviewForm(ModelForm):
    class Meta:
        model = ReviewTab
        fields = ['name','review_data']