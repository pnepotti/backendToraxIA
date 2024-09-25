from django.db import models
"""
    Modelo para almacenar información sobre radiografías"""


class Radiography(models.Model):
    patient_name = models.CharField(max_length=100)
    patient_dni = models.CharField(max_length=10)
    doctor_name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='radiographies/')
    # Agrega campo de fecha automática
    # uploaded_at = models.DateTimeField(auto_now_add=True)
    desease = models.CharField(max_length=100, default="Unknown")

    def __str__(self):
        return f"Radiografía de {self.patient_name}"
