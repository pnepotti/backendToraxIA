from django.db import models

# Modelo para almacenar información sobre el médico


class Doctor(models.Model):
    name = models.CharField(max_length=100)
    matricula = models.CharField(
        max_length=10, blank=True, null=True, unique=True)
    specialty = models.CharField(
        max_length=100, blank=True, null=True)  # Especialidad opcional

    def __str__(self):
        return f"Dr. {self.name}"

# Modelo para almacenar información sobre el paciente


class Patient(models.Model):
    name = models.CharField(max_length=100)
    dni = models.CharField(max_length=10, unique=True)  # DNI único
    # Fecha de nacimiento opcional
    date_of_birth = models.DateField(blank=True, null=True)

    def __str__(self):
        return f"{self.name} (DNI: {self.dni})"

# Modelo para almacenar la radiografía


class Radiography(models.Model):
    # Cambiado de image a radiography
    radiography = models.ImageField(upload_to='radiographies/')
    uploaded_at = models.DateTimeField(
        auto_now_add=True)  # Fecha de subida automática
    # Un médico puede tener muchas imágenes
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    # Un paciente puede tener muchas imágenes
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    # Descripcion opcional
    descripcion = models.CharField(max_length=400, blank=True, null=True)
    # Diagnóstico opcional
    diagnostico = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Radiografía de {self.patient.name} subida en {self.uploaded_at}"

# Modelo para almacenar cada predicción asociada a una radiografía


class Prediction(models.Model):
    radiography_image = models.ForeignKey(
        Radiography, on_delete=models.CASCADE, related_name='predictions')
    # Enfermedad diagnosticada
    disease = models.CharField(max_length=100, default="Unknown")
    prediction_probability = models.DecimalField(
        max_digits=6, decimal_places=3)
    prediction_confidence = models.DecimalField(max_digits=6, decimal_places=3)
    prediction_entropy = models.DecimalField(max_digits=6, decimal_places=3)


# Fecha automática del diagnóstico
diagnosed_at = models.DateTimeField(auto_now_add=True)


def __str__(self):
    return f"Predicción: {self.disease} para radiografía: {self.radiography_image}"
