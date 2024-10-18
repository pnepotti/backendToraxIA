# Generated by Django 5.1.1 on 2024-10-13 21:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diagnostics', '0006_alter_doctor_matricula'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='prediction',
            name='diagnosed_at',
        ),
        migrations.AlterField(
            model_name='prediction',
            name='prediction_confidence',
            field=models.DecimalField(decimal_places=3, max_digits=6),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='prediction_entropy',
            field=models.DecimalField(decimal_places=3, max_digits=6),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='prediction_probability',
            field=models.DecimalField(decimal_places=3, max_digits=6),
        ),
    ]