from django import forms
from .models import QueryLibrary, QueryNode, QueryEdge, QueryJson

class SelectNodeForm(forms.Form):
    data=[]
    for node in QueryNode.objects.all():
        data.append((node.node_id,node.node_id))
    selected_node = forms.ChoiceField(label='node id',choices=data)
    def append_choice(self):
        # 参考：https://qiita.com/44d/items/897e5bb20113315af006
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
        ('text_output', 'text output'),
        ('figure_output', 'figure output'),
        ('table_output', 'table output'),
        ('reachability', 'reachability'),
    ]
    input_node_type = forms.ChoiceField(label='type',choices=data)

class SelectSavedQueryForm(forms.Form):
    data=[]
    for query in QueryJson.objects.all():
        data.append((query.query_name, query.query_name))
    selected_query = forms.ChoiceField(label='query',choices=data)
    def append_choice(self):
        data=[]
        for query in QueryJson.objects.all():
            data.append((query.query_name, query.query_name))
        self.fields['selected_query'].choices = data


class UploadQueryFileForm(forms.Form):
    # 参考：https://qiita.com/ekzemplaro/items/07abd9a941bcd0eb5834
    query_name = forms.CharField(max_length=50, label='Query name')
    file = forms.FileField(label='Upload query file')

class UploadTableDataFileForm(forms.Form):
    file = forms.FileField(label='Upload data file')

