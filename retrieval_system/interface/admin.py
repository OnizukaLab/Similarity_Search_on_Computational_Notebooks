from django.contrib import admin
from .models import QueryNode, QueryEdge, QueryLibrary, QueryJson

admin.site.register(QueryNode)
admin.site.register(QueryEdge)
admin.site.register(QueryLibrary)
admin.site.register(QueryJson)

