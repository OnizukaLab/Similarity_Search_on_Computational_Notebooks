import networkx as nx
import pandas as pd
import sys
import logging
import json
import os
import timeit

from django import http
from django.http.response import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.utils import timezone

from .models import QueryLibrary, QueryNode, QueryEdge

current_dir=os.getcwd()
#search_engine_path="/Users/misato/Desktop/my_code" # 以前Jupyterで動かしていた方（卒論用の実験）, コピー時点で内容は同じ
search_engine_path=f"{current_dir}/interface/retrieval_engine_module"
juneau_file_path="/Users/misato/Desktop/my_code/juneau_copy" # 以前Jupyterで動かしていた方（卒論用の実験）, コピー時点で内容は同じ
#juneau_file_path=f"{current_dir}/interface/retrieval_engine_module/juneau_copy"
sys.path.append(search_engine_path)
sys.path.append(f"{search_engine_path}/mymodule")
sys.path.append(juneau_file_path)

from juneau.db.table_db import connect2db_engine, connect2gdb
from juneau.config import config

from workflow_matching import WorkflowMatching
#from mymodule.workflow_matching import WorkflowMatching


# Global variable
G_in_this_nb=None


# ***** initialization *****

#valid_nb_name_file_path="/Users/misato/Desktop/データセット/valid_nb_name.txt" # 以前Jupyterで動かしていた方（卒論用の実験）, コピー時点で内容は同じ
valid_nb_name_file_path=f"{current_dir}/interface/retrieval_engine_module/valid_nb_name.txt"
with open(f"{search_engine_path}/dict_nb_name_and_cleaned_nb_name.json", mode="r") as f:
    load_json=f.read()
    dict_nb_name_and_cleaned_nb_name=json.loads(load_json)
with open(f"{search_engine_path}/dict_nb_name_and_cleaned_nb_name2.json", mode="r") as f:
    load_json=f.read()
    dict_nb_name_and_cleaned_nb_name2=json.loads(load_json)
with open(f"{search_engine_path}/dict_nb_name_dir.json", mode="r") as f:
    load_json=f.read()
    dict_nb_name_dir=json.loads(load_json)


# neo4jに接続
try:
    graph_db = connect2gdb()
    logging.info("Connecting to neo4j is successful!")
except:
    print("Connecting to neo4j is failed.")
    sys.exit(1)

# postgresqlに接続
try:
    psql_engine = connect2db_engine(config.sql.dbname)
    logging.info("Connecting to PostgreSQL is successful!")
except:
    print("Connecting to PostgreSQL is failed.")
    sys.exit(1)



def get_db_graph(wm):
    logging.info("Getting workflow graphs from neo4j now...")
    set_graph_start_time = timeit.default_timer()    
    wm.set_db_graph()
    set_graph_end_time = timeit.default_timer()    
    set_graph_time=set_graph_end_time-set_graph_start_time
    logging.info(f"Completed!: Getting workflow graphs from neo4j ({set_graph_time} sec).")
    G_in_this_nb=wm.G
    return wm, G_in_this_nb

def get_db_graph2(wm):
    wm.G=G_in_this_nb
    return wm

    
wm = WorkflowMatching(psql_engine, graph_db, valid_nb_name_file_path=valid_nb_name_file_path)
wm, G_in_this_nb = get_db_graph(wm)




def index(request):
    return render(request, 'interface/index.html', {'w_C':0, 'w_D':0, 'w_L':0, 'w_O':0})
    #return HttpResponse("index page.")

def search(request):
    try:
        arg_list={
            'w_C': request.POST['w_C'],
            'w_D': request.POST['w_D'],
            'w_L': request.POST['w_L'],
            'w_O': request.POST['w_O'],
            'library': request.POST['library'],
            'source_code': request.POST['source_code'],
            'data': request.POST['data'],
            'output': request.POST['output'],
            'debug_data':request.POST
        }
    except:
        return render(request, 'interface/index.html', {'w_C':0, 'w_D':2, 'w_L':0, 'w_O':0})
    else:
        #return HttpResponseRedirect(reverse('interface:index', args=(w_C)))
        return render(request, 'interface/index.html', arg_list)


def show_query_graph(request):
    """
    try:
        arg_list={
            'w_C': request.POST['w_C'],
            'w_D': request.POST['w_D'],
            'w_L': request.POST['w_L'],
            'w_O': request.POST['w_O'],
            'library': request.POST['library'],
            'source_code': request.POST['source_code'],
            'data': request.POST['data'],
            'output': request.POST['output'],
            'debug_data':request.POST
        }
    except:
        return render(request, 'interface/index.html', {'w_C':0, 'w_D':2, 'w_L':0, 'w_O':0})
    else:




    self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
    self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
    self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
    self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")
    self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {"DataFrame": 1}, "max_indegree": 1, "max_outdegree": 2}
    self.set_query_workflow_info_Display_data()
    """
        

    wm = WorkflowMatching(psql_engine, graph_db, valid_nb_name_file_path=valid_nb_name_file_path)
    wm.set_db_graph2(G_in_this_nb)
    #wm, G_in_this_nb = get_db_graph(wm)

    QueryGraph, query_root, query_lib, query_cell_code, query_table, node_id_to_node_name = build_QueryGraph()

    wm.QueryGraph = QueryGraph
    wm.query_root=query_root
    wm.query_lib=query_lib
    wm.query_cell_code=query_cell_code
    wm.attr_of_q_node_type = nx.get_node_attributes(wm.QueryGraph, "node_type")
    wm.attr_of_q_real_cell_id=nx.get_node_attributes(wm.QueryGraph, "real_cell_id")
    wm.attr_of_q_display_type=nx.get_node_attributes(wm.QueryGraph, "display_type")
    #wm.query_workflow_info={"Cell": 3, "Var": 0, "Display_data": {"text": 1}, "max_indegree": 1, "max_outdegree": 1}
    wm.set_query_workflow_info()
    wm.set_query_workflow_info_Display_data()
        
    top_k_result, nb_score = search(wm, w_c=8, w_v=1, w_l=1, w_d=1, k=5)

    node_object_list = QueryNode.objects.all()
    output = '<br>'.join([node.node_contents for node in node_object_list])
    output += '<br><br>*** Graph\'s Information ***<br>' + '<br>'.join([f"{node_name} ----- {node_id}" for node_id, node_name in node_id_to_node_name.items()])
    #output += '<br><br>*** Libraries\' Information ***<br>' + '<br>'.join([library_name for library_name in library_list])
    output += '<br><br>*** Result ***<br>' + '<br>'.join([f"{dict_nb_name_and_cleaned_nb_name[items[0]]}, {items[1]}" for items in top_k_result])
    return HttpResponse(output)



def build_QueryGraph():
    QueryGraph = nx.DiGraph()
    node_object_list = QueryNode.objects.all()
    query_cell_code={}
    query_table={}

    # ノード名設定用の変数3つ. int.
    code_id=0
    data_id=0
    output_id=0

    # edge設定用の{ノード番号:ノード名}の辞書. {int: string}.
    node_id_to_node_name={}

    for node in node_object_list:
        if node.node_type=="code":
            node_name = f"cell_query_{code_id}"
            QueryGraph.add_node(node_name, node_type="Cell", node_id=f"{node.node_id}")
            query_cell_code[node_name] = node.node_contents
            node_id_to_node_name[node.node_id]=node_name
            code_id+=1
        elif node.node_type=="data":
            node_name = f"query_var{data_id}"
            QueryGraph.add_node(node_name, node_type="Var", node_id=f"{node.node_id}")
            query_table[node_name] = node.node_contents
            node_id_to_node_name[node.node_id]=node_name
            data_id+=1
        elif node.node_type=="output":
            # TODO:display_type="text"を正しい内容に変更．
            node_name = f"query_display{data_id}"
            QueryGraph.add_node(node_name, node_type="Display_data", display_type="text", node_id=f"{node.node_id}")
            node_id_to_node_name[node.node_id]=node_name
            output_id+=1
        logging.info(f"{node_name} appended to QueryGraph.")
    
    library_list=[]
    for library in QueryLibrary.objects.all():
        library_list.append(library.library_name)
    query_lib=pd.Series(library_list)
    
    for edge in QueryEdge.objects.all():
        QueryGraph.add_edge(node_id_to_node_name[edge.parent_node_id], node_id_to_node_name[edge.successor_node_id])

    query_root=None
    root_count=0
    for n in QueryGraph.nodes():
        if len(list(QueryGraph.predecessors(n)))==0:
            logging.info(f"{n} is root node.")
            query_root = n
            root_count+=1
            #break
        #logging.info(f"{n} is not root node.")
    if root_count>1:
        print("Only a node is allowed in query graph.")
        sys.exit(1)
    
    logging.info("Completed!: Building a query graph.")
    return QueryGraph, query_root, query_lib, query_cell_code, query_table, node_id_to_node_name



def search(wm, w_c, w_v, w_l, w_d, k, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_calc_data_sim_approximately=False, flg_cache_query_table=False, save_running_time=False):
    return searching_top_k_notebooks(wm, w_c, w_v, w_l, w_d, k, flg_chk_invalid_by_workflow_structure=flg_chk_invalid_by_workflow_structure, flg_flg_prune_under_sim=flg_flg_prune_under_sim, flg_optimize_calc_order=flg_optimize_calc_order, flg_caching=flg_caching, flg_calc_data_sim_approximately=flg_calc_data_sim_approximately, flg_cache_query_table=flg_cache_query_table, save_running_time=save_running_time)
   

# 時間計測あり 提案手法
#def new_proposal_method_bench_mark_4_2_with_wildcard_in_query(wm, w_c, w_v, w_l, w_d, k, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_calc_data_sim_approximately=False, flg_cache_query_table=False):
def searching_top_k_notebooks(wm, w_c, w_v, w_l, w_d, k, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_calc_data_sim_approximately=False, flg_cache_query_table=False, save_running_time=False):
    """
    Search similar notebooks and return top-k result.

    Args:
        wm : a instance of searching component
        w_c (float): weight of code similarity
        w_v (float): weight of table data similarity
        w_l (float): weight of library similarity
        w_d (float): weight of cell's outputs similarity
        k (int): k of top-k
        flg_chk_invalid_by_workflow_structure (bool)
        flg_flg_prune_under_sim (bool)
        flg_optimize_calc_order (bool)
        flg_caching (bool)
        flg_calc_data_sim_approximately (bool)
        flg_cache_query_table (bool)
    """

    wm.set_k(k)
    wm.get_db_workflow_info()
    #wm.load_calculated_sim(calculated_sim_path)
    wm.set_each_w(w_c=w_c, w_v=w_v, w_l=w_l, w_d=w_d)
    wm.set_flg_running_faster(flg_chk_invalid_by_workflow_structure, flg_flg_prune_under_sim, flg_optimize_calc_order, flg_caching, flg_calc_data_sim_approximately, flg_cache_query_table=flg_cache_query_table)

    logging.info(f"k: {wm.k}")
    logging.info(f"Running subgraph_matching_with_wildcard_in_query...")
    subgraph_matching_time, _1, _2, _3, _4, detected_count, nb_count=wm.subgraph_matching_with_wildcard_in_query(tflg=True)


    # *************************************
    # ノートブック類似度計算ベンチマーク
    logging.info(f"Running calc_nb_score_for_new_proposal_method_2_3...")
    calc_nb_score_start_time = timeit.default_timer()    
    count, nb_count=wm.calc_nb_score_for_new_proposal_method_2_3()
    calc_nb_score_end_time = timeit.default_timer()   

    logging.info(f"Running top_k_nb(k)...")
    top_k_nb_start_time = timeit.default_timer() 
    top_k_result=wm.top_k_nb(k)
    top_k_nb_end_time = timeit.default_timer()    

    calc_nb_score_time=calc_nb_score_end_time-calc_nb_score_start_time
    top_k_nb_time=top_k_nb_end_time-top_k_nb_start_time

    # *************************************
    if save_running_time:
        method="newproposal_method_4"

        with open(f"{sta_dir}/running_time.txt", mode="r") as f:
            load_json=f.read()

        if load_json=="":
            time_sta={}
        else:
            time_sta=json.loads(load_json)
        if method not in time_sta:
            time_sta[method]=[]

        print(f"num of nb is {num_of_nb} and pruned is {num_of_nb-len(wm.nb_score)}")
        store_row=[wm.w_c, wm.w_v, wm.w_l, wm.w_d, subgraph_matching_time+calc_nb_score_time+top_k_nb_time, subgraph_matching_time, calc_nb_score_time, top_k_nb_time, num_of_nb, detected_count, k, flg_chk_invalid_by_workflow_structure, flg_flg_prune_under_sim, wm.calc_v_count, wm.each_calc_time_sum["Cell"], wm.each_calc_time_sum["Var"], wm.each_calc_time_sum["Display_data"], wm.each_calc_time_sum["Library"], flg_optimize_calc_order, flg_caching, flg_calc_data_sim_approximately]
        if store_row not in time_sta[method]:
            time_sta[method].append(store_row)
            with open(f"{sta_dir}/running_time.txt", mode="w") as f:
                f.write(json.dumps(time_sta))

        # *************************************
        store_nb_score_path=f"{sta_dir}/nb_score_{method}_{wm.w_c}_{wm.w_v}_{wm.w_l}_{wm.w_d}_{k}_{flg_chk_invalid_by_workflow_structure}_{flg_flg_prune_under_sim}.txt"
        with open(store_nb_score_path, mode="w") as f:
            f.write(json.dumps(wm.nb_score))
        
    return top_k_result, wm.nb_score
