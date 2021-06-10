from django.urls import path

from . import views

app_name = 'interface'
urlpatterns = [
    path('', views.index, name='index'),
    path('search', views.search, name='search'),
    path('form', views.form, name='form'),
    path('index', views.index, name='index'),
    path('show_query_graph_redirect', views.show_query_graph_redirect, name='show_query_graph_redirect'),
    path('result', views.result, name='result'),
    path('make_test_formset', views.make_test_formset, name='make_test_formset'),
    path('export/', views.PostExport, name='export')
]