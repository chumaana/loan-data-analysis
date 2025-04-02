from django.urls import path
from .views import home, predict_loan

urlpatterns = [
    path("", home, name="home"),
    path("predict/", predict_loan, name="predict"),
]
