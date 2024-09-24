from django.db import models

"""
    Modelo para almacenar información sobre radiografías.

    Attributes:
        image: El archivo de imagen de la radiografía.
        patient_name: El nombre del paciente.
        patient_dni: El DNI del paciente.
        doctor_name: El nombre del médico que realizó el diagnóstico.
        uploaded_at: Fecha y hora en la que se subió la imagen.
    """


class Radiography(models.Model):
    patient_name = models.CharField(max_length=100)
    patient_dni = models.CharField(max_length=10)
    doctor_name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='radiographies/')
    # Agrega campo de fecha automática
    uploaded_at = models.DateTimeField(auto_now_add=True)
    desease = models.CharField(max_length=100)

    def __str__(self):
        return f"Radiografía de {self.patient_name}"
