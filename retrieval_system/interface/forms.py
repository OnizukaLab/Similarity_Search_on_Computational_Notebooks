from django import forms
from .models import QueryLibrary, QueryNode, QueryEdge, QueryJson

class HelloForm(forms.Form):
    data=[
        ('サンプル1', '1'),
        ('サンプル2', '2'),
        ('サンプル3', '3'),
    ]
    choice = forms.ChoiceField(label='sample_label',choices=data)


class SelectNodeForm(forms.Form):
    data=[]
    for node in QueryNode.objects.all():
        data.append((node,node.node_id))
    choice = forms.ChoiceField(label='node id',choices=data)
    #def __init__(self, label, choices):
    #    self.choice = forms.ChoiceField(label=label,choices=choices)

class SelectEdgeForm(forms.Form):
    data=[]
    for edge in QueryEdge.objects.all():
        data.append((edge,f"node{edge.parent_node_id}-node{edge.successor_node_id}"))
    choice = forms.ChoiceField(label='Edge',choices=data)
    #def __init__(self, label, choices):
    #    self.choice = forms.ChoiceField(label=label,choices=choices)

class SelectTypeForm(forms.Form):
    data=[
        ('code', 'code'),
        ('data', 'data'),
        ('output', 'output'),
    ]
    choice = forms.ChoiceField(label='Type',choices=data)

class SelectSavedQueryForm(forms.Form):
    data=[]
    for query in QueryJson.objects.all():
        data.append((query, query.query_name))
    choice = forms.ChoiceField(label='Query',choices=data)
