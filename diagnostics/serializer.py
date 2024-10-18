from rest_framework import serializers
from .models import Doctor, Patient, Radiography, Prediction

# Serializer para el modelo Doctor


class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Doctor
        fields = ['id', 'name', 'matricula', 'specialty']

# Serializer para el modelo Patient


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ['id', 'name', 'dni', 'date_of_birth']


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'radiography_image', 'disease',
                  'prediction_probability', 'prediction_confidence', 'prediction_entropy']


class RadiographySerializer(serializers.ModelSerializer):
    # Incluir predicciones relacionadas
    predictions = PredictionSerializer(many=True, read_only=True)

    class Meta:
        model = Radiography
        fields = ['id', 'radiography', 'uploaded_at',
                  'doctor', 'patient', 'diagnostico', 'predictions']
