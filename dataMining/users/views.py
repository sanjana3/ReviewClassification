from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm
import sqlite3 as db
import csv
import xlwt, xlrd
from review.models import ReviewTab
from django.http import HttpResponse
from final import _functionTorun

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})

def usermain(request):
    if(request.GET.get('mybtn')):
        # print("I am in IF")
        export_to_csv(request)
        _functionTorun()
        wb = xlrd.open_workbook("review.xls")
        sheet = wb.sheet_by_index(0)
        rows = []
        for i in range(1,(sheet.nrows)):
            row = sheet.row_values(i)
            row[0]=int(row[0])
            rows.append(row)
        context = {
            'rows': rows,
        }
        return render(request, 'users/home.html',context)
    else:
        # print("I am in else")
        return render(request, 'users/main.html')

# def userhome(request):
#     if(request.GET.get('mybtn')):
#         print("I am in IF")
#         export_to_csv(request)
#         _functionTorun()
#         wb = xlrd.open_workbook("review.xls")
#         sheet = wb.sheet_by_index(0)
#         rows = []
#         for i in range(1,(sheet.nrows)):
#             row = sheet.row_values(i)
#             row[0]=int(row[0])
#             rows.append(row)
#         context = {
#             'rows': rows,
#         }
#         return render(request, 'users/home.html',context)
#     else:
#         print("I am in else")
#         return render(request, 'users/main.html')

def export_to_csv(request):
    # with open('review.csv', 'w', newline='') as f_handle:
    #     writer = csv.writer(f_handle)
    #     writer.writerow(['ID','name','review','gender','rating'])
    #     r_data = ReviewTab.objects.all().values_list('rid','name','review_data','gender','rating')
    #     for r in r_data:
    #         writer.writerow(r)
    wb = xlwt.Workbook()
    ws = wb.add_sheet('reviews')
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True
    columns = ['ID','name','review','gender','rating']
    for col_num in range(len(columns)):
        ws.write(row_num, col_num, columns[col_num], font_style)
    font_style = xlwt.XFStyle()
    rows = ReviewTab.objects.all().values_list('rid','name','review_data','gender','rating')
    for row in rows:
        row_num += 1
        for col_num in range(len(row)):
            ws.write(row_num, col_num, row[col_num], font_style)
    wb.save('review.xls')
    # print("Done creating CSV")

@login_required
def profile(request):
    return render(request, 'users/profile.html')