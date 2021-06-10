from django import forms
from .models import QueryLibrary, QueryNode, QueryEdge, QueryJson

class HelloForm(forms.Form):
    data=[
        ('サンプル1', '1'),
        ('サンプル2', '2'),
        ('サンプル3', '3'),
    ]
    choice = forms.ChoiceField(label='sample_label',choices=data) #choice: nameフィールド. HTMLでname="choice"と同じ.


class SelectNodeForm(forms.Form):
    data=[]
    for node in QueryNode.objects.all():
        data.append((node.node_id,node.node_id))
    selected_node = forms.ChoiceField(label='node id',choices=data)
    def append_choice(self):
        data=[]
        for node in QueryNode.objects.all():
            data.append((node.node_id,node.node_id))
        self.fields['selected_node'].choices = data

class SelectParentNodeForm(forms.Form):
    data=[]
    for node in QueryNode.objects.all():
        data.append((node.node_id,node.node_id))
    input_parent_node_id = forms.ChoiceField(label='parent node id',choices=data)
    input_successor_node_id = forms.ChoiceField(label='successor node id',choices=data)
    def append_choice(self):
        data=[]
        for node in QueryNode.objects.all():
            data.append((node.node_id,node.node_id))
        self.fields['input_parent_node_id'].choices = data
        self.fields['input_successor_node_id'].choices = data


class SelectEdgeForm(forms.Form):
    data=[]
    for edge in QueryEdge.objects.all():
        data.append((f"{edge.parent_node_id}, {edge.successor_node_id}",f"node{edge.parent_node_id}-node{edge.successor_node_id}"))
    selected_edge = forms.ChoiceField(label='Edge',choices=data)
    def append_choice(self):
        data=[]
        for edge in QueryEdge.objects.all():
            data.append((f"{edge.parent_node_id}, {edge.successor_node_id}",f"node{edge.parent_node_id}-node{edge.successor_node_id}"))
        self.fields['selected_edge'].choices = data

class SelectTypeForm(forms.Form):
    data=[
        ('code', 'code'),
        ('data', 'data'),
        ('output', 'output'),
    ]
    input_node_type = forms.ChoiceField(label='Type',choices=data)

class SelectSavedQueryForm(forms.Form):
    data=[]
    for query in QueryJson.objects.all():
        data.append((query.query_name, query.query_name))
    selected_query = forms.ChoiceField(label='Query',choices=data)
    def append_choice(self):
        data=[]
        for query in QueryJson.objects.all():
            data.append((query.query_name, query.query_name))
        self.fields['selected_query'].choices = data
