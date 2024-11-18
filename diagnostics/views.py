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

import shutil
from datetime import date
from rest_framework.views import APIView

# Rutas configuradas para guardar las imágenes validadas y el archivo CSV
FUTURO_SET_PATH = "./media/futuro_dataset/"
METADATA_CSV_PATH = os.path.join(FUTURO_SET_PATH, "metadatos.csv")


# Rutas relativas a los modelos .h5
TOXIC_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'diagnostics', 'ia_models', 'ModeloToraxIAValidacionMuchasImgv2.h5')
DISEASE_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'diagnostics', 'ia_models', 'ModeloToraxIA4Clases2024-09-19_16-58-29.h5')

# Carga perezosa del modelo (solo cuando sea necesario)
torax_model = None
disease_model = None


# pylint: disable=no-member


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
        descripcion = request.data.get('descripcion')

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
            classes = ['COVID19', 'NORMAL', 'NEUMONIA', 'TUBERCULOSIS']

            probability = prediction[0][class_index]
            entropy = -np.sum(prediction[0] * np.log(prediction[0] + 1e-9))
            confidence = np.max(prediction[0]) - \
                np.partition(prediction[0], -2)[-2]
            # Umbrales para confianza y entropía
            confidence_threshold = 0.85  # Ajusta según el modelo
            entropy_threshold = 0.40  # Ajusta según el modelo

            # Verificar confianza y entropía antes de aceptar la predicción
            if confidence >= confidence_threshold and entropy <= entropy_threshold:
                result = classes[class_index]  # Predicción aceptada
            else:
                result = 'ENFERMEDAD DESCONOCIDA POR EL MODELO'  # Caso de enfermedad no conocida

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
                patient=patient,
                descripcion=descripcion
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
                'radiography_id': radiography.id,
                'patient_name': radiography.patient.name,
                'patient_dni': radiography.patient.dni,
                'doctor_name': radiography.doctor.name,
                'descripcion': radiography.descripcion,
                'prediccion': result,
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
                'predictions': prediction_data,
                'diagnostico': radiography.diagnostico,
                'descripcion': radiography.descripcion
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
                'predictions': prediction_data,
                'descripcion': radiography.descripcion
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
                'predictions': prediction_data,
                'descripcion': radiography.descripcion
            })

        return Response({'radiographies': image_data}, status=status.HTTP_200_OK)


class ImagesViewPorMatri(APIView):

    def get(self, request, format=None):

        doctor_matricula = request.query_params.get('matricula')

        # Validar que se proporcionen ambos parámetros
        if not doctor_matricula:
            return Response({'error': 'Debe proporcionar la matrícula del médico.'}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar el doctor por matrícula
        try:
            doctor = Doctor.objects.get(matricula=doctor_matricula)
        except Doctor.DoesNotExist:
            return Response({'error': 'No se encontró un doctor con la matrícula proporcionada.'}, status=status.HTTP_404_NOT_FOUND)

        # Filtrar radiografías por paciente y doctor
        radiographies = Radiography.objects.filter(doctor=doctor).prefetch_related(
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
                'doctor_name': radiography.doctor.name,
                'diagnostico': radiography.diagnostico,
                'predictions': prediction_data,
                'descripcion': radiography.descripcion
            })

        return Response({'radiographies': image_data}, status=status.HTTP_200_OK)


class ImagesViewPorIdRx(APIView):

    def get(self, request, format=None):
        # Capturar el ID de la radiografía de los parámetros de consulta
        id_rx = request.query_params.get('idRx')

        # Validar que se proporcione el ID de la radiografía
        if not id_rx:
            return Response({'error': 'Debe proporcionar el Id de la radiografía.'}, status=status.HTTP_400_BAD_REQUEST)

        # Validar si id_rx es un entero
        if not id_rx.isdigit():
            return Response({'error': 'El ID debe ser un número entero válido.'}, status=400)

        # Buscar la radiografía por ID
        try:
            radiography = Radiography.objects.get(id=id_rx)
        except Radiography.DoesNotExist:
            return Response({'error': 'No se encontró una radiografía con el ID proporcionado.'}, status=status.HTTP_404_NOT_FOUND)

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

        # Añadir la información de la radiografía con sus predicciones
        image_data = {
            'radiography_id': radiography.id,
            'image_url': request.build_absolute_uri(radiography.radiography.url),
            'uploaded_at': radiography.uploaded_at,
            'doctor_name': radiography.doctor.name,
            'patient_name': radiography.patient.name,
            'diagnostico': radiography.diagnostico,
            'predictions': prediction_data,
            'descripcion': radiography.descripcion
        }

        return Response({'radiography': image_data}, status=status.HTTP_200_OK)


# class DiagnosticPorIdRx(APIView):

    def post(self, request, format=None):
        # Capturar id de la radiografía y el diagnóstico desde el cuerpo de la solicitud
        id_rx = request.data.get('idRx')
        diagnostico = request.data.get('diagnostico')

        # Validar que ambos parámetros fueron proporcionados
        if not id_rx or not diagnostico:
            return Response({'error': 'Debe proporcionar el ID de la radiografía y el diagnóstico.'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Buscar la radiografía por ID
        try:
            radiography = Radiography.objects.get(id=id_rx)
        except Radiography.DoesNotExist:
            return Response({'error': 'No se encontró una radiografía con el ID proporcionado.'},
                            status=status.HTTP_404_NOT_FOUND)

        # Asignar el diagnóstico al campo correspondiente y guardar
        radiography.diagnostico = diagnostico
        radiography.save()

        return Response({'success': 'Diagnóstico asignado exitosamente.'}, status=status.HTTP_200_OK)


# Función para copiar la imagen y guardar los metadatos
def guardar_imagen_y_metadatos(ruta_origen, nombre_archivo, diagnostico):
    try:
        # Asegurarse de que la carpeta FUTURO_SET_PATH existe
        if not os.path.exists(FUTURO_SET_PATH):
            os.makedirs(FUTURO_SET_PATH)
            print(f"Carpeta '{FUTURO_SET_PATH}' creada exitosamente.")

        # Copiar la imagen a la carpeta destino
        ruta_destino = os.path.join(FUTURO_SET_PATH, nombre_archivo)
        shutil.copy(ruta_origen, ruta_destino)
        print(f"Imagen copiada a {ruta_destino}")

        # Guardar metadatos en el archivo CSV
        fecha_validacion = date.today().isoformat()
        if not os.path.exists(METADATA_CSV_PATH):
            # Crear archivo con encabezado si no existe
            with open(METADATA_CSV_PATH, "w") as csv_file:
                csv_file.write("nombre_archivo,diagnostico,fecha_validacion\n")
                print(f"Archivo de metadatos creado en {METADATA_CSV_PATH}")

        # Agregar datos al archivo CSV
        with open(METADATA_CSV_PATH, "a") as csv_file:
            csv_file.write(
                f"{nombre_archivo},{diagnostico},{fecha_validacion}\n")
            print(
                f"Metadatos guardados: {nombre_archivo}, {diagnostico}, {fecha_validacion}")

    except Exception as e:
        print(f"Error en guardar_imagen_y_metadatos: {str(e)}")
        raise


class DiagnosticPorIdRx(APIView):
    def post(self, request, format=None):
        # Capturar ID de la radiografía y diagnóstico desde la solicitud
        id_rx = request.data.get('idRx')
        diagnostico = request.data.get('diagnostico')

        if not id_rx or not diagnostico:
            return Response({'error': 'Debe proporcionar el ID de la radiografía y el diagnóstico.'},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            radiography = Radiography.objects.get(id=id_rx)
        except Radiography.DoesNotExist:
            return Response({'error': 'No se encontró una radiografía con el ID proporcionado.'},
                            status=status.HTTP_404_NOT_FOUND)

        try:
            # Guardar el diagnóstico en la base de datos
            radiography.diagnostico = diagnostico
            radiography.save()

            # Copiar imagen y guardar metadatos
            guardar_imagen_y_metadatos(
                ruta_origen=radiography.radiography.path,
                nombre_archivo=os.path.basename(radiography.radiography.name),
                diagnostico=diagnostico
            )
        except Exception as e:
            return Response({'error': f'Error al procesar la imagen o guardar metadatos: {str(e)}'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({'success': 'Diagnóstico asignado y datos guardados correctamente.'}, status=status.HTTP_200_OK)
