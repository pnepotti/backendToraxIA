from django.db import models


class Radiography(models.Model):
    # Información del paciente
    patient_name = models.CharField(max_length=100)
    patient_dni = models.CharField(max_length=10)

    # Información del médico
    doctor_name = models.CharField(max_length=100)

    # Información del diagnóstico
    disease = models.CharField(max_length=100, default="Unknown")
    prediccion_probabilidad = models.FloatField()
    prediccion_confianza = models.FloatField()
    prediccion_entropia = models.FloatField()
    diagnosed_at = models.DateTimeField(
        auto_now_add=True)  # Fecha automática al guardar

    # Imagen de la radiografía
    image = models.ImageField(upload_to='radiographies/')

    def __str__(self):
        return f"Radiografía de {self.patient_name} (Diagnóstico: {self.disease})"
