import os
import numpy as np
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image
from django.conf import settings
from .models import Radiography
from keras.models import load_model
from keras.utils import img_to_array

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
    img = img.convert('RGB')  # Convertir a RGB si no lo está
    img = img.resize(target_size)  # Redimensionar la imagen
    img = img_to_array(img)  # Convertir a un array NumPy
    # Añadir la dimensión extra para lotes (batch)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalizar los valores de la imagen (0-1)
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
                return Response({'message': 'La imagen no es una radiografía de tórax.'}, status=status.HTTP_200_OK)

            # Si es tórax, hacer la predicción de la enfermedad
            prediction = disease_model.predict(preprocessed_img)
            class_index = np.argmax(prediction[0])

            # Mapear el índice de predicción a una clase de enfermedad
            classes = ['COVID19', 'NORMAL', 'PNEUMONIA',
                       'TUBERCULOSIS']  # Nombres de las clases
            result = classes[class_index]
            sorted_probabilities = np.sort(prediction[0])[
                ::-1]  # Ordenar de mayor a menor

            # Guardar los datos de la radiografía en la base de datos
            Radiography.objects.create(
                patient_name=patient_name,
                patient_dni=patient_dni,
                doctor_name=doctor_name,
                image=img_file,
                disease=result,  # Corrección aquí
                # Agregar probabilidad
                prediccion_probabilidad=prediction[0][class_index],
                # Agregar confianza
                prediccion_confianza=(
                    sorted_probabilities[0] - sorted_probabilities[1]),
                # Agregar entropía
                prediccion_entropia=(-np.sum(prediction[0]
                                     * np.log(prediction[0] + 1e-9)))
            )

            # Retornar el diagnóstico
            return Response({
                'diagnosis': result,
                'probability': prediction[0][class_index],
                'entropy': (-np.sum(prediction[0] * np.log(prediction[0] + 1e-9))),
                'confidence': (sorted_probabilities[0] - sorted_probabilities[1])
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': f'Error durante el procesamiento: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Vista para obtener imágenes guardadas


class ImagesView(APIView):
    """Vista para obtener las imágenes guardadas en la base de datos."""

    def get(self, request, patient_dni=None):
        if patient_dni is not None:
            images = Radiography.objects.filter(patient_dni=patient_dni)
        else:
            images = Radiography.objects.all()

        if not images:
            return Response({'message': 'No se encontraron imágenes.'}, status=status.HTTP_404_NOT_FOUND)

        # Convertir los resultados a un formato serializable
        serialized_images = [{'id': img.id, 'patient_name': img.patient_name,
                              'doctor_name': img.doctor_name} for img in images]

        return Response(serialized_images, status=status.HTTP_200_OK)
