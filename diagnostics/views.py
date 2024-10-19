import os
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
from django.conf import settings
from .models import Radiography, Prediction, Doctor, Patient
from decimal import Decimal

# Rutas relativas a los modelos .h5
TOXIC_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'diagnostics', 'ia_models', 'ModeloToraxIAValidacionMuchasImgv2.h5')
DISEASE_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'diagnostics', 'ia_models', 'ModeloToraxIA5Clases2024-10-02_16-36-52.h5')

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
        doctor_matricula = request.data.get('doctorMatricula')

        # Validar que todos los campos estén presentes
        if not patient_name or not patient_dni or not doctor_name or not doctor_matricula:
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
            if is_torax[0][0] < 0.5:  # Ajusta este umbral según tu modelo
                return Response({'message': 'La imagen no es una radiografía de tórax.'}, status=status.HTTP_200_OK)

            # Si es tórax, hacer la predicción de la enfermedad
            prediction = disease_model.predict(preprocessed_img)
            class_index = np.argmax(prediction[0])
            classes = ['COVID19', 'NORMAL', 'PNEUMONIA',
                       'PNEUMOTHORAX', 'TUBERCULOSIS']

            probability = prediction[0][class_index]
            entropy = -np.sum(prediction[0] * np.log(prediction[0] + 1e-9))
            confidence = np.max(prediction[0]) - \
                np.partition(prediction[0], -2)[-2]
            # Umbrales para confianza y entropía
            confidence_threshold = 0.7  # Ajusta según el modelo
            entropy_threshold = 0.5  # Ajusta según el modelo

            # Verificar confianza y entropía antes de aceptar la predicción
            if confidence >= confidence_threshold and entropy <= entropy_threshold:
                result = classes[class_index]  # Predicción aceptada
            else:
                result = 'DESCONOCIDO'  # Caso de enfermedad no conocida

        except Exception as e:
            return Response({'error': f'Error en la predicción: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            # Buscar o crear doctor y paciente
            doctor, _ = Doctor.objects.get_or_create(
                name=doctor_name, matricula=doctor_matricula)
            patient, _ = Patient.objects.get_or_create(
                name=patient_name, dni=patient_dni)

            # Guardar los datos de la radiografía en la base de datos
            radiography = Radiography.objects.create(
                radiography=img_file,
                doctor=doctor,
                patient=patient
            )

            # Guardar la predicción en la base de datos
            Prediction.objects.create(
                radiography_image=radiography,
                disease=result,
                prediction_probability=Decimal(str(probability)),
                prediction_confidence=Decimal(str(confidence)),
                prediction_entropy=Decimal(str(entropy))
            )

            # Retornar el diagnóstico
            return Response({
                'diagnosis': result,
                'probability': probability,
                'entropy': entropy,
                'confidence': confidence
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': f'Error al guardar la información del doctor o paciente: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Vistas para manejar la obtención de imágenes


class ImagesView(APIView):
    def get(self, request, format=None):
        patient_dni = request.query_params.get('dni')

        # Validar que se proporcione un DNI
        if not patient_dni:
            return Response({'error': 'Debe proporcionar un DNI.'}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar el paciente por DNI
        try:
            patient = Patient.objects.get(dni=patient_dni)
        except Patient.DoesNotExist:
            return Response({'error': 'No se encontró un paciente con el DNI proporcionado.'}, status=status.HTTP_404_NOT_FOUND)

        # Obtener las últimas 5 radiografías del paciente, incluyendo las predicciones
        radiographies = Radiography.objects.filter(patient=patient).prefetch_related(
            'predictions').order_by('-uploaded_at')[:5]

        # Si no se encuentran radiografías
        if not radiographies.exists():
            return Response({'error': 'No se encontraron radiografías para el paciente proporcionado.'}, status=status.HTTP_404_NOT_FOUND)

        # Serializar los datos de las radiografías y sus predicciones
        image_data = []
        for radiography in radiographies:
            # Obtener las predicciones asociadas a la radiografía
            predictions = radiography.predictions.all()

            # Agrupar predicciones
            prediction_data = []
            for prediction in predictions:
                prediction_data.append({
                    'disease': prediction.disease,
                    'probability': prediction.prediction_probability,
                    'confidence': prediction.prediction_confidence,
                    'entropy': prediction.prediction_entropy
                })

            # Añadir la información de cada radiografía con sus predicciones
            image_data.append({
                'radiography_id': radiography.id,
                'image_url': request.build_absolute_uri(radiography.radiography.url),
                'uploaded_at': radiography.uploaded_at,
                'doctor_name': radiography.doctor.name,
                'patient_name': radiography.patient.name,
                'predictions': prediction_data
            })

        return Response({'radiographies': image_data}, status=status.HTTP_200_OK)


class ImagesViewPorMatriYDni(APIView):

    def get(self, request, format=None):
        # Capturar DNI del paciente y matrícula del médico de los parámetros de consulta
        patient_dni = request.query_params.get('dniInputMedico')
        doctor_matricula = request.query_params.get('matricula')

        # Validar que se proporcionen ambos parámetros
        if not patient_dni or not doctor_matricula:
            return Response({'error': 'Debe proporcionar tanto el DNI del paciente como la matrícula del médico.'}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar el paciente por DNI
        try:
            patient = Patient.objects.get(dni=patient_dni)
        except Patient.DoesNotExist:
            return Response({'error': 'No se encontró un paciente con el DNI proporcionado.'}, status=status.HTTP_404_NOT_FOUND)

        # Buscar el doctor por matrícula
        try:
            doctor = Doctor.objects.get(matricula=doctor_matricula)
        except Doctor.DoesNotExist:
            return Response({'error': 'No se encontró un doctor con la matrícula proporcionada.'}, status=status.HTTP_404_NOT_FOUND)

        # Filtrar radiografías por paciente y doctor
        radiographies = Radiography.objects.filter(patient=patient, doctor=doctor).prefetch_related(
            'predictions').order_by('-uploaded_at')[:5]

        # Si no se encuentran radiografías
        if not radiographies.exists():
            return Response({'error': 'No se encontraron radiografías para el paciente y médico proporcionados.'}, status=status.HTTP_404_NOT_FOUND)

        # Serializar los datos de las radiografías y sus predicciones
        image_data = []
        for radiography in radiographies:
            # Obtener las predicciones asociadas a la radiografía
            predictions = radiography.predictions.all()

            # Agrupar predicciones
            prediction_data = []
            for prediction in predictions:
                prediction_data.append({
                    'disease': prediction.disease,
                    'probability': prediction.prediction_probability,
                    'confidence': prediction.prediction_confidence,
                    'entropy': prediction.prediction_entropy
                })

            # Añadir la información de cada radiografía con sus predicciones
            image_data.append({
                'radiography_id': radiography.id,
                'image_url': request.build_absolute_uri(radiography.radiography.url),
                'uploaded_at': radiography.uploaded_at,
                'patient_name': radiography.patient.name,
                'diagnostico': radiography.diagnostico,
                'predictions': prediction_data
            })

        return Response({'radiographies': image_data}, status=status.HTTP_200_OK)


class ImagesViewPorMatriYDiagNull(APIView):

    def get(self, request, format=None):
        # Capturar la matrícula del médico de los parámetros de consulta
        doctor_matricula = request.query_params.get('matricula')

        # Validar que se proporcione la matrícula
        if not doctor_matricula:
            return Response({'error': 'Debe proporcionar la matrícula del médico.'}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar el doctor por matrícula
        try:
            doctor = Doctor.objects.get(matricula=doctor_matricula)
        except Doctor.DoesNotExist:
            return Response({'error': 'No se encontró un doctor con la matrícula proporcionada.'}, status=status.HTTP_404_NOT_FOUND)

        # Filtrar radiografías por doctor y diagnostico nulo
        radiographies = Radiography.objects.filter(doctor=doctor, diagnostico__isnull=True).prefetch_related(
            'predictions').order_by('-uploaded_at')[:5]

        # Si no se encuentran radiografías
        if not radiographies.exists():
            return Response({'error': 'No se encontraron radiografías sin diagnosticar para el médico proporcionado.'}, status=status.HTTP_404_NOT_FOUND)

        # Serializar los datos de las radiografías y sus predicciones
        image_data = []
        for radiography in radiographies:
            # Obtener las predicciones asociadas a la radiografía
            predictions = radiography.predictions.all()

            # Agrupar predicciones
            prediction_data = []
            for prediction in predictions:
                prediction_data.append({
                    'disease': prediction.disease,
                    'probability': prediction.prediction_probability,
                    'confidence': prediction.prediction_confidence,
                    'entropy': prediction.prediction_entropy
                })

            # Añadir la información de cada radiografía con sus predicciones
            image_data.append({
                'radiography_id': radiography.id,
                'image_url': request.build_absolute_uri(radiography.radiography.url),
                'uploaded_at': radiography.uploaded_at,
                'patient_name': radiography.patient.name,
                'diagnostico': radiography.diagnostico,
                'predictions': prediction_data
            })

        return Response({'radiographies': image_data}, status=status.HTTP_200_OK)
