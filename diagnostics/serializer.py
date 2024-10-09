from rest_framework import serializers
from .models import Doctor, Patient, RadiographyImage, RadiographyPredictions

# Serializer para el modelo Doctor


class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Doctor
        fields = ['id', 'name', 'specialty']

# Serializer para el modelo Patient


class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ['id', 'name', 'dni', 'date_of_birth']

# Serializer para el modelo RadiographyPredictions


class RadiographyPredictionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = RadiographyPredictions
        fields = ['id', 'radiography_image', 'disease',
                  'prediction_probability', 'prediction_confidence', 'diagnosed_at']

# Serializer para el modelo RadiographyImage, incluyendo las predicciones


class RadiographyImageSerializer(serializers.ModelSerializer):
    predictions = RadiographyPredictionsSerializer(
        many=True, read_only=True)  # Incluir predicciones relacionadas

    class Meta:
        model = RadiographyImage
        fields = ['id', 'image', 'uploaded_at',
                  'doctor', 'patient', 'predictions']
