from django.urls import path

from . import views

app_name = 'interface'
urlpatterns = [
    path('', views.index, name='index'),
    path('search', views.search, name='search'),
    path('form', views.form, name='form'),
    path('index', views.index, name='index'),
    path('make_test_formset', views.make_test_formset, name='make_test_formset'),
    path('export/', views.PostExport, name='export'),
    path('workflowgraph/', views.showresultworkflowgraph, name='showresultworkflowgraph'),
    path('exportData/', views.exportData, name='exportData'),
]