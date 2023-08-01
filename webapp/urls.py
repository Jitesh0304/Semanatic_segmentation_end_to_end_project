from django.urls import path
from . import views


urlpatterns = [
    path('train/', views.trainingmodel, name= 'trainingmodel'),
]
