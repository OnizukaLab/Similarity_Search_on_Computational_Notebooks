from django.contrib import admin
from .models import QueryNode, QueryEdge, QueryLibrary

admin.site.register(QueryNode)
admin.site.register(QueryEdge)
admin.site.register(QueryLibrary)

