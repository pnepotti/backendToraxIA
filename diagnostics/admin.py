from django.contrib import admin
from .models import Doctor, Patient, RadiographyImage, RadiographyPredictions

# Registra los modelos en el administrador de Django
admin.site.register(Doctor)
admin.site.register(Patient)
admin.site.register(RadiographyImage)
admin.site.register(RadiographyPredictions)
