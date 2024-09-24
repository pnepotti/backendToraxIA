from django.urls import path
from .views import DiagnosticView, ImagesView

urlpatterns = [
    path('api/diagnostic/', DiagnosticView.as_view(), name='diagnostic'),
    path('api/images/', ImagesView.as_view(), name='search_by_dni'),
]
