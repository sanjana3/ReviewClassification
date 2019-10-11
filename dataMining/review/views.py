from django.shortcuts import render, redirect
from django.http import HttpResponse
from review.forms import ReviewForm
from .models import *
from django.contrib import messages

def home(request):
    form = ReviewForm(request.POST)
    if request.method == 'POST':
        # print("request POST")
        if form.is_valid():
            # form.save()
            # print("Validation Success")
            f_name = form.cleaned_data.get('name')
            f_review = form.cleaned_data.get('review_data')
            instance = ReviewTab.objects.create(name=f_name,review_data=f_review)
            instance.save()
            return render(request, 'review/result.html')
        else:
            print(form.errors)
            form = ReviewForm()
    return render(request, 'review/home.html', {'form': form})