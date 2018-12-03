# Generated by Django 2.0.3 on 2018-12-02 07:47

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ReviewTab',
            fields=[
                ('rid', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('review_data', models.CharField(max_length=1000)),
                ('gender', models.CharField(blank=True, max_length=25)),
                ('rating', models.CharField(blank=True, max_length=25)),
            ],
        ),
    ]
