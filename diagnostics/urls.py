from django.urls import path
from .views import DiagnosticView, ImagesView, ImagesViewPorMatriYDni

urlpatterns = [
    path('api/diagnostic/', DiagnosticView.as_view(), name='diagnostic'),
    path('api/images/', ImagesView.as_view(), name='search_by_dni'),
    path('api/images-by-matricula-dni/',
         ImagesViewPorMatriYDni.as_view(), name='images_by_matricula_dni')

]
