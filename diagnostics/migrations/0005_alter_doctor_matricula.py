# Generated by Django 5.1.1 on 2024-10-12 01:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diagnostics', '0004_alter_doctor_matricula'),
    ]

    operations = [
        migrations.AlterField(
            model_name='doctor',
            name='matricula',
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
    ]
