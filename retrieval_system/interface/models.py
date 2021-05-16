from django.db import models
import networkx as nx

#from mymodule.workflow_matching import WorkflowMatching

"""
# Create your models here.
def make_sample_query_nx2_3_another2_with_wildcard_2(self): #論文用
        # DataSet16, mobile-phone-pricing-predictions.ipynb
        self.QueryGraph=nx.DiGraph()

        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell")
        for i in range(1,3):
            self.QueryGraph.add_node(f"wildcard_{i}", node_type="AnyWildcard")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="text")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="png")

        self.QueryGraph.add_edge("cell_query_1", "wildcard_1")
        self.QueryGraph.add_edge("wildcard_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "wildcard_2")
        self.QueryGraph.add_edge("wildcard_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_1", "query_var1")
        self.QueryGraph.add_edge("cell_query_2", "query_display1")
        self.QueryGraph.add_edge("cell_query_3", "query_display2")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("32_X2_lephonepricingpredictions")

        code_table=self.fetch_source_code_table("lephonepricingpredictions")
        code=code_table[code_table["cell_id"]==32]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==33]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==35]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("lephonepricingpredictions")
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {"text":1, "png":1}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()
"""