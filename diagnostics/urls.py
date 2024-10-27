from django.urls import path
from .views import DiagnosticView, ImagesView, ImagesViewPorMatriYDni, ImagesViewPorMatriYDiagNull, ImagesViewPorMatri, ImagesViewPorIdRx

urlpatterns = [
    path('api/diagnostic/', DiagnosticView.as_view(), name='diagnostic'),
    path('api/images/', ImagesView.as_view(), name='search_by_dni'),
    path('api/images-by-matricula-dni/',
         ImagesViewPorMatriYDni.as_view(), name='images_by_matricula_dni'),
    path('api/images-by-matricula-null-diagnostic/',
         ImagesViewPorMatriYDiagNull.as_view(), name='images_by_matricula_null_diagnostic'),
    path('api/images-by-matricula/',
         ImagesViewPorMatri.as_view(), name='images_by_matricula'),
    path('api/images-by-idRx/',
         ImagesViewPorIdRx.as_view(), name='images_by_idRx')
]
