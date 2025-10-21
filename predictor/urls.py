from django.urls import path
from . import views

urlpatterns = [
    path('predict/',views.ConcretePredictAPIView.as_view()),
    path('optimize/',views.ConcreteOptimizeAPIView.as_view()),
]
