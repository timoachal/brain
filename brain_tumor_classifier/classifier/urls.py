from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_and_predict, name='upload_and_predict'),
    path('scan/<int:scan_id>/', views.scan_detail, name='scan_detail'),
    path('history/', views.scan_history, name='scan_history'),
    path('gradcam/<int:scan_id>/', views.get_gradcam, name='get_gradcam'),
]

