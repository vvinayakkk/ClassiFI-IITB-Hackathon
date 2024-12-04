# urls.py in your app directory
from django.urls import path
from .views import train_classification_view, predict_classification_view,predict_single_document_view

urlpatterns = [
    path('train1/', train_classification_view, name='train_classification'),
    path('predict1/', predict_classification_view, name='predict_classification'),
    path('uploadsingle/', predict_single_document_view, name='predict_classification'),
]