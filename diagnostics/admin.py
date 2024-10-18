from django.contrib import admin
from .models import Doctor, Patient, Radiography, Prediction

# Registra los modelos en el administrador de Django
admin.site.register(Doctor)
admin.site.register(Patient)
admin.site.register(Radiography)
admin.site.register(Prediction)
