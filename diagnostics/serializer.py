from rest_framework import serializers
from .models import Radiography


class RadiographySerializer(serializers.ModelSerializer):
    class Meta:
        model = Radiography
        fields = ['patient_name', 'patient_dni', 'doctor_name',
                  'image', 'uploaded_at']  # Incluir uploaded_at
        read_only_fields = ['uploaded_at']  # Este campo es solo de lectura
