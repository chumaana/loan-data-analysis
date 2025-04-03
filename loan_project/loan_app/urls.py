from django.urls import path
from .views import home, predict_loan, graphs

urlpatterns = [
    path("", home, name="home"),
    path("predict/", predict_loan, name="predict_loan"),
    path("graphs/", graphs, name="graphs"),
]
