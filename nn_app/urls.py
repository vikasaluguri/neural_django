from django.urls import path
from . import views

urlpatterns = [
    path('',                      views.dashboard,       name='dashboard'),
    path('create/',               views.create_network,  name='create_network'),
    path('train/<int:pk>/',       views.train_network,   name='train_network'),
    path('results/<int:pk>/',     views.results,         name='results'),
    path('delete/<int:pk>/',      views.delete_session,  name='delete_session'),

    # AJAX / API
    path('api/train/<int:pk>/',   views.api_train,       name='api_train'),
    path('api/predict/<int:pk>/', views.api_predict,     name='api_predict'),
]
