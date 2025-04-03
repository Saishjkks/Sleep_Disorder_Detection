from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('index1/', views.index1, name='index1'),  # Define URL for index1.html

]
