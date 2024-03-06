from django.urls import path
from .views import OCRImageView
from . import views

urlpatterns = [
    path('api/ocr/', OCRImageView.as_view(), name='ocr_api'),
    path("", views.index, name="index"),
]
