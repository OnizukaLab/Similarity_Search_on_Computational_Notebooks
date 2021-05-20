from django.db import models
import networkx as nx

#from mymodule.workflow_matching import WorkflowMatching

class QueryNode(models.Model):
    node_id = models.PositiveIntegerField()
    node_type = models.CharField(max_length=200)
    node_contents = models.TextField()
    def __str__(self):
        return "QueryNode" + str(self.node_id)

class QueryEdge(models.Model):
    parent_node_id = models.PositiveBigIntegerField()
    successor_node_id = models.PositiveBigIntegerField()
    def __str__(self):
        return "QueryEdge(" + str(self.parent_node_id) + "," + str(self.successor_node_id) + ")"

class QueryLibrary(models.Model):
    library_name = models.CharField(max_length=200)
    def __str__(self):
        return "Library\"" + str(self.library_name) + "\""

class QueryJson(models.Model):
    """
    QueryNode, QueryEdge, QueryLibraryをまとめてjson形式で保存したもの．
    log用．
    """
    query_name = models.CharField(max_length=200)
    query_contents = models.TextField()
    def __str__(self):
        return self.query_name