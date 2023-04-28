from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.about, name='about'),
    path('demo/', views.demo, name='demo'),
    path('antibiogram/', views.antibiogram, name='antibiogram'),
    path('modeldetails/', views.modeldetails, name='modeldetails'),
]