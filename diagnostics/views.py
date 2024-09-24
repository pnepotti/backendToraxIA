import os
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
import numpy as np
from django.http import JsonResponse
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

from .models import Radiography
from django.conf import settings

# Rutas relativas a los modelos .h5
TOXIC_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'diagnostics', 'ia_models', 'ModeloToraxIAValidacionMuchasImgv2.h5')
DISEASE_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'diagnostics', 'ia_models', 'ModeloToraxIA4Clases2024-09-19_16-58-29.h5')

# Carga perezosa del modelo (solo cuando sea necesario)
torax_model = None
disease_model = None

# Función para cargar modelos si aún no están cargados


def load_models():
    global torax_model, disease_model
    try:
        if torax_model is None:
            torax_model = load_model(TOXIC_MODEL_PATH)
        if disease_model is None:
            disease_model = load_model(DISEASE_MODEL_PATH)
    except Exception as e:
        raise Exception(f"Error al cargar los modelos: {e}")

# Función para preprocesar la imagen


def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalizar
    return img

# Vista para manejar la predicción de imágenes


class DiagnosticView(APIView):
    """Vista para el diagnóstico de radiografías de tórax."""
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        # Capturar datos del formulario
        patient_name = request.data.get('patientName')
        patient_dni = request.data.get('patientDni')
        doctor_name = request.data.get('doctorName')

        # Validar que todos los campos estén presentes
        if not patient_name or not patient_dni or not doctor_name:
            return Response({'error': 'Todos los campos son obligatorios.'}, status=status.HTTP_400_BAD_REQUEST)

        # Capturar la imagen
        img_file = request.FILES.get('image')
        if img_file is None:
            return Response({'error': 'No se ha enviado una imagen.'}, status=status.HTTP_400_BAD_REQUEST)

        try:

            img = Image.open(img_file)
        except Exception as e:
            return Response({'error': f'Error al procesar la imagen: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Cargar modelos si no están cargados aún
            load_models()

            # Preprocesar la imagen para el modelo de validación de tórax
            preprocessed_img = preprocess_image(img, target_size=(224, 224))

            # Validar si es una radiografía de tórax
            is_torax = torax_model.predict(preprocessed_img)
            if is_torax[0] < 0.5:  # Ajusta este umbral según tu modelo
                return Response({'error': 'La imagen no es una radiografía de tórax.'}, status=status.HTTP_400_BAD_REQUEST)

            # Si es tórax, hacer la predicción de la enfermedad
            prediction = disease_model.predict(preprocessed_img)
            class_index = np.argmax(prediction[0])

            # Mapear el índice de predicción a una clase de enfermedad
            classes = ['NORMAL', 'COVID19', 'PNEUMONIA',
                       'TUBERCULOSIS']  # Nombres de las clases
            result = classes[class_index]

            # Guardar los datos de la radiografía en la base de datos
            Radiography.objects.create(
                patient_name=patient_name,
                patient_dni=patient_dni,
                doctor_name=doctor_name,
                image=img_file,
                desease=result
            )

            # Retornar el diagnóstico
            return Response({'diagnosis': result}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': f'Error durante el procesamiento: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Vista para manejar la obtención de imágenes


class ImagesView(APIView):
    def get(self, request, format=None):
        patient_dni = request.query_params.get('dni')
        if not patient_dni:
            return Response({'error': 'Debe proporcionar un DNI.'}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar imágenes por DNI del paciente
        images = Radiography.objects.filter(patient_dni=patient_dni)
        if not images:
            return Response({'error': 'No se encontraron imágenes para el DNI proporcionado.'}, status=status.HTTP_404_NOT_FOUND)

        # Serializar los datos de las imágenes
        image_data = [{'id': img.id, 'image_url': img.image.url}
                      for img in images]

        return Response({'images': image_data}, status=status.HTTP_200_OK)
