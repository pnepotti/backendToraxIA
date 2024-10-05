from django.db import models

# Modelo para almacenar información sobre el médico


class Doctor(models.Model):
    name = models.CharField(max_length=100)
    specialty = models.CharField(
        max_length=100, blank=True, null=True)  # Especialidad opcional

    def __str__(self):
        return f"Dr. {self.name}"


# Modelo para almacenar información sobre el paciente
class Patient(models.Model):
    name = models.CharField(max_length=100)
    # DNI único para cada paciente
    dni = models.CharField(max_length=10, unique=True)
    # Fecha de nacimiento opcional
    date_of_birth = models.DateField(blank=True, null=True)

    def __str__(self):
        return f"{self.name} (DNI: {self.dni})"


# Modelo para almacenar la imagen radiográfica
class RadiographyImage(models.Model):
    image = models.ImageField(upload_to='radiographies/')
    uploaded_at = models.DateTimeField(
        auto_now_add=True)  # Fecha de subida automática

    def __str__(self):
        return f"Imagen radiográfica subida en {self.uploaded_at}"


# Modelo para almacenar el diagnóstico asociado a una imagen radiográfica
class RadiographyDiagnosis(models.Model):
    # Relación con la imagen radiográfica
    radiography_image = models.ForeignKey(
        RadiographyImage, on_delete=models.CASCADE)

    # Relación con el médico y el paciente
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)

    # Información del diagnóstico
    disease = models.CharField(max_length=100, default="Unknown")
    prediction_probability = models.FloatField()  # Probabilidad entre 0 y 1
    prediction_confidence = models.DecimalField(
        max_digits=5, decimal_places=2)  # Ejemplo: 95.50%
    # Fecha automática del diagnóstico
    diagnosed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Diagnóstico: {self.disease} para {self.patient.name}"
