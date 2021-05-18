from django.urls import path

from . import views

app_name = 'interface'
urlpatterns = [
    path('', views.index, name='index'),
    path('search', views.search, name='search'),
    path('show_query_graph', views.show_query_graph, name='show_query_graph'),
]