# Generated by Django 5.1.1 on 2024-11-02 05:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diagnostics', '0007_remove_prediction_diagnosed_at_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='radiography',
            name='descripcion',
            field=models.CharField(blank=True, max_length=400, null=True),
        ),
        migrations.AlterField(
            model_name='radiography',
            name='diagnostico',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]