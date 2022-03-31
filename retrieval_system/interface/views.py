import networkx as nx
import pandas as pd
import sys
import logging
import json
import os
import timeit
import io

from django import http, forms
from django.http.response import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse
from django.views import generic
from django.utils import timezone

from .models import QueryLibrary, QueryNode, QueryEdge, QueryJson
from .forms import SelectNodeForm, SelectEdgeForm, SelectSavedQueryForm, SelectParentNodeForm, UploadQueryFileForm, UploadTableDataFileForm, SelectTypeForm


flg_get_db_graph = True # For development (run search parts or not). 検索部分を動かすかどうか. 


# ***** Run the followings automatically when this system is started. システム起動時に実行される - ここから *****

# パスを通す
current_dir=os.getcwd()
search_engine_path=f"{current_dir}/interface/retrieval_engine_module"
upper_dir=current_dir[:current_dir.rfind("/Similarity_Search_on_Computational_Notebooks/")]
flg_loading=False
if os.path.exists(f"{upper_dir}/juneau_copy"):
    juneau_file_path=f"{upper_dir}/juneau_copy"
    flg_loading=True
    sys.path.append(juneau_file_path)
elif os.path.exists(f"{current_dir}/interface/retrieval_engine_module/juneau_copy"):
    juneau_file_path=f"{current_dir}/interface/retrieval_engine_module/juneau_copy"
    flg_loading=True
    sys.path.append(juneau_file_path)
elif os.path.exists(f"{upper_dir}/juneau"):
    juneau_file_path=f"{upper_dir}/juneau" 
    flg_loading=True
    sys.path.append(juneau_file_path)
sys.path.append(search_engine_path)
sys.path.append(f"{search_engine_path}/mymodule")

##
#  Import the required classes. 各種必要なクラスをインポートする
##
# Import Juneau. Juneauのインポート
from juneau.db.table_db import connect2db_engine, connect2gdb
from mymodule.config import config
# Import our proposed method. 提案検索手法インポート
from workflow_matching import WorkflowMatching

if flg_loading:
    logging.info(f"Loading Juneau is successful!: {juneau_file_path}")
else:
    logging.error("Please arrange your files with the search system and Juneau. The details is written in README.md.")

# Global variables for the search engine (holding a instance for the search). データベース読み込み短縮のための，インスタンス保持変数
G_in_this_nb=None
# Setting port id for Jupyter Notebook. Jupyter Notebookを起動時のポート設定
jupyter_notebook_localhost_number=8888


# ***** Initialization *****
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


# Connect to Neo4j. neo4jに接続
try:
    graph_db = connect2gdb()
    logging.info("Connecting to neo4j is successful!")
except:
    print("Connecting to neo4j is failed.")
    sys.exit(1)

# Connect to PostgreSQL. postgresqlに接続
try:
    psql_engine = connect2db_engine(config.sql.dbname)
    logging.info("Connecting to PostgreSQL is successful!")
except:
    print("Connecting to PostgreSQL is failed.")
    sys.exit(1)



def get_db_graph(wm):
    """Load workflow graphs from Neo4j."""
    logging.info("Getting workflow graphs from neo4j now...")
    set_graph_start_time = timeit.default_timer()    
    wm.set_db_graph()
    set_graph_end_time = timeit.default_timer()    
    set_graph_time=set_graph_end_time-set_graph_start_time
    logging.info(f"Getting workflow graphs from neo4j is successful! ({set_graph_time} sec).")
    G_in_this_nb=wm.G
    return wm, G_in_this_nb

def set_db_graph2(wm, G_in_this_nb, flg_get_db_graph):
    """Load workflow graphs using holding instance."""
    if flg_get_db_graph:
        return wm.set_db_graph2(G_in_this_nb)
    else:
        return wm

def delete_node_safely(node_id):
    err_msg=""
    if QueryEdge.objects.filter(parent_node_id=node_id).exists() or QueryEdge.objects.filter(successor_node_id=node_id).exists():
        logging.error("error: Please delete related edges first.")
        err_msg = "Error: This node has any edges. Please delete the edges first."
    else:
        QueryNode.objects.filter(node_id=node_id).delete()
    return err_msg
        

def delete_edge_safely(edge):
    QueryEdge.objects.filter(parent_node_id=edge[0], successor_node_id=edge[1]).delete()
    

# Initialize a instance for the search.
wm = WorkflowMatching(psql_engine, graph_db, valid_nb_name_file_path=valid_nb_name_file_path)
if flg_get_db_graph:
    wm, G_in_this_nb = get_db_graph(wm)
# Initialize parameters
code_weight=1
data_weight=1
library_weight=1
output_weight=1
k=10

# ***** Run the above automatically when this system is started. システム起動時に実行される - ここまで *****



def index(request, *args, **kwargs):
    """
    The main page.
    最初にアクセスする，メインページ．
    """
    logging.info(f"Loaded \'index\' page.")

    #wm = WorkflowMatching(psql_engine, graph_db, valid_nb_name_file_path=valid_nb_name_file_path)
    request_data=request.POST

    uploadfile={"filename":"sample_file_name"}

    #wm, node_id_to_node_name = build_QueryGraph(wm)

    send_node_object_list=arrange_node_object_list(QueryNode.objects.all())
    edges=arrange_edge_object_list(QueryEdge.objects.all())
    libraries_list=[]
    for item in QueryLibrary.objects.all():
        libraries_list.append(item.library_name)
    #msg["libraries_list"]=json.dumps(libraries_list)
    #msg["libraries_list"]="\n".join(libraries_list)
    msg={
        'node_object_list': send_node_object_list, 
        'edges':edges, 
        #'node_id_to_node_name': node_id_to_node_name, 
        "debug_data":request_data, 
        "code_weight":code_weight, 
        "data_weight":data_weight, 
        "library_weight":library_weight, 
        "output_weight":output_weight,
        "uploadfile": uploadfile,
        "form_setting_node": SelectNodeForm(),
        "form_setting_parent_node": SelectParentNodeForm(),
        "form_delete_edge": SelectEdgeForm(),
        "form_setting_type": SelectTypeForm(),
        "form_setting_query": SelectSavedQueryForm(),
        "query_name": "",
        "arranged_result": "",
        "form_upload_query": UploadQueryFileForm(),
        "form_upload_data": UploadTableDataFileForm(),
        "libraries_list": libraries_list,
        }
    

    return render(request, 'interface/index.html', msg)

uploadfile={"filename":"sample_file_name"}
def PostExport(request):
    """
    役職テーブルを全件検索して、CSVファイルを作成してresponseに出力します。
    参考：http://houdoukyokucho.com/2020/06/29/post-1296/，https://qiita.com/t-iguchi/items/d2862e7ef7ec7f1b07e5
    """
    export_file_name = request.POST["export_file_name"]
    json_file = dump_to_json(QueryNode.objects.all(), QueryEdge.objects.all(), QueryLibrary.objects.all())
    #response = HttpResponse(open('/path/to/pdf/marketista.pdf', 'rb').read(), content_type='application/json; charset=Shift-JIS' )
    response = HttpResponse(content_type='application/json; charset=Shift-JIS' )
    response['Content-Disposition'] = f'attachment; filename="{export_file_name}.json"'
    response.write(json_file)
    return response


    response = HttpResponse(content_type='text/csv; charset=Shift-JIS')
    filename = urllib.parse.quote((u'CSVファイル.csv').encode("utf8"))
    response['Content-Disposition'] = 'attachment; filename*=UTF-8\'\'{}'.format(filename)
    writer = csv.writer(response)
    for post in Post.objects.all():
        writer.writerow([post.pk, post.name])
    return response
    
def exportData(request):
    """
    参考：http://houdoukyokucho.com/2020/06/29/post-1296/，https://qiita.com/t-iguchi/items/d2862e7ef7ec7f1b07e5
    """
    savetablename = request.POST["export_name"]
    if savetablename == "":
        savetablename = "download_data"
    tablename = request.POST["export_data_name"]
    export_file = get_data_as_csv(tablename)
    #response = HttpResponse(open('/path/to/pdf/marketista.pdf', 'rb').read(), content_type='application/json; charset=Shift-JIS' )
    response = HttpResponse(content_type='application/json; charset=Shift-JIS' )
    response['Content-Disposition'] = f'attachment; filename="{savetablename}.csv"'
    response.write(export_file)
    return response

def get_data_as_csv(tablename):
    export_table=wm.fetch_var_table(tablename)
    export_file = export_table.to_csv()
    return export_file


def form(request, *args, **kwargs):
    """
    メインページから遷移するページ．このページからもこのページに遷移する．
    """
    logging.info(f"Loaded \'form\' page.")

    request_data=request.POST
    err_msg=""
    search_time=0
    uploadfile={"filename":"sample_file_name"}
    #デバッグ用
    debug_data=request_data
    #if (request.method == 'POST'):

    #func_loading_button(request_data) 検証後，以下を左に置き換え．
    if "loading_button" in request_data:
        json_file = QueryJson.objects.filter(query_name=request_data["selected_query"])[0].query_contents
        replace_all_using_json_log(json_file)
    else:
        pass


    if "setting_button" in request_data:
        if request_data["setting_button"] == "Reset":
            delete_all()

        if request_data["setting_button"] == "Delete":
            if "selected_node" in request_data:
                try:
                    node_id = int(request_data["selected_node"])
                    err_msg = delete_node_safely(node_id)
                except:
                    pass
            if "selected_edge" in request_data:
                try:
                    edge = request_data["selected_edge"].strip(' ').split(',')
                    logging.info(f"Delete edge: parent_node_id {edge[0]}")
                    logging.info(f"Delete edge: parent_node_id {edge[1]}")
                    delete_edge_safely(edge)
                except:
                    pass
            if "libraries_contents" in request_data:
                libraries_contents = request_data["libraries_contents"]
                library_list=libraries_contents.strip(" ").split(",")
                for item in library_list:
                    QueryLibrary.objects.filter(library_name=item).delete()
            if "selected_query" in request_data:
                QueryJson.objects.filter(query_name=request_data["selected_query"]).delete()

        if request_data["setting_button"] in ["Set", "Add", "Change"]: 
            try:
                node_id = int(request_data["input_node_id"])
                node_type = request_data["input_node_type"]
                node_contents = request_data["input_node_contents"]
                parent_node_id = int(request_data["input_parent_node_id"])
            except:
                pass
            else:
                new_node = QueryNode(node_id=node_id, node_type=node_type, node_contents=node_contents)
                QueryNode.objects.filter(node_id=node_id).delete()
                new_node.save()
                if parent_node_id == node_id and node_id == 0:
                    pass
                else:
                    new_edge = QueryEdge(parent_node_id=parent_node_id, successor_node_id=node_id)
                    new_edge.save()
            if "libraries_contents" in request_data:
                if "add_library_mode" in request_data:
                    logging.info("add_library_mode in request_data")
                    libraries_contents = request_data["libraries_contents"]
                    extracted_libraries_contents = extract_library_name(libraries_contents, request_data["add_library_mode"])
                    save_libraries(extracted_libraries_contents)
                else:
                    logging.err("err: 'add_library_mode' is null!")
                    libraries_contents = request_data["libraries_contents"]
                    extracted_libraries_contents = extract_library_name(libraries_contents, "mode_codes")
                    save_libraries(extracted_libraries_contents)

        if request_data["setting_button"] == "Add":
            try:
                parent_node_id = int(request_data["input_parent_node_id"])
                successor_node_id = int(request_data["input_successor_node_id"])
            except:
                pass
            else:
                new_edge = QueryEdge(parent_node_id=parent_node_id, successor_node_id=successor_node_id)
                new_edge.save()

    if "setting_weight_button" in request_data:
        try:
            code_weight=float(request_data["code_weight"])
        except:
            code_weight=1.
        try:
            data_weight=float(request_data["data_weight"])
        except:
            data_weight=1.
        try:
            library_weight=float(request_data["library_weight"])
        except:
            library_weight=1.
        try:
            output_weight=float(request_data["output_weight"])
        except:
            output_weight=1.
    else:
        code_weight=1.
        data_weight=1.
        library_weight=1.
        output_weight=1.
    
    if "saving_button" in request_data:
        saving_query_name = request_data["query_name"]
        save_query(saving_query_name, QueryNode.objects.all(), QueryEdge.objects.all(), QueryLibrary.objects.all())
        form_setting_query = SelectSavedQueryForm()
        form_setting_query.append_choice()



    if "search_button" in request_data:
        wm = WorkflowMatching(psql_engine, graph_db, valid_nb_name_file_path=valid_nb_name_file_path)
        node_id_to_node_name, nb_score, send_node_object_list, arranged_result, search_time = get_result(wm, w_c=code_weight, w_v=data_weight, w_l=library_weight, w_d=output_weight, k=k)
        debug_data=json.dumps(wm.query_workflow_info)
        #arranged_result_json = json.dumps(arranged_result) #デバッグ用
    else:
        wm = WorkflowMatching(psql_engine, graph_db, valid_nb_name_file_path=valid_nb_name_file_path)
        wm, node_id_to_node_name = build_QueryGraph(wm)
        debug_data=json.dumps(wm.query_workflow_info)
        arranged_result=[]
        arranged_result_json=""
        send_node_object_list=arrange_node_object_list(QueryNode.objects.all())

        
    edges=arrange_edge_object_list(QueryEdge.objects.all())


    SelectNodeForm().append_choice()
    SelectParentNodeForm().append_choice()
    SelectEdgeForm().append_choice()
    SelectSavedQueryForm().append_choice()
    libraries_list=[]
    for item in QueryLibrary.objects.all():
        libraries_list.append(item.library_name)
    #msg["libraries_list"]=json.dumps(libraries_list)
    #msg["libraries_list"]="\n".join(libraries_list)
    msg = {
        'node_object_list': send_node_object_list, 
        'edges':edges,
        #'node_id_to_node_name': node_id_to_node_name, 
        #"debug_data":request_data, 
        "debug_data":debug_data, 
        "code_weight":code_weight, 
        "data_weight":data_weight, 
        "library_weight":library_weight, 
        "output_weight":output_weight,
        "uploadfile": uploadfile,
        "form_setting_node": SelectNodeForm(),
        "form_setting_parent_node": SelectParentNodeForm(),
        "form_delete_edge": SelectEdgeForm(),
        "form_setting_type": SelectTypeForm(),
        "form_setting_query": SelectSavedQueryForm(),
        "query_name": "",
        "err_msg": err_msg,
        "arranged_result": arranged_result,
        "libraries_list": libraries_list,
        "search_time": "",
        "form_upload_query": UploadQueryFileForm(),
        }
    if search_time != 0:
        msg["search_time"]=f"({round(search_time,1)} sec.)"

    return render(request, 'interface/index.html', msg)


def showresultworkflowgraph(request, *args, **kwargs):
    logging.info(f"Loaded \'showresultworkflowgraph\' page.")
    request_data=request.POST
    err_msg=""
    debug_data=request_data #デバッグ用
    nb_name = request_data["nb_name"]
    original_nb_name = dict_nb_name_and_cleaned_nb_name[nb_name]
    #nb_name = "edaonindiancuisine" #開発用

    send_node_object_list, send_edge_object_list = arrange_nodes_and_edges_of_result_notebooks_to_object_list(wm, nb_name)
    #send_node_object_list, send_edge_object_list = sample_nodes_and_edges_of_result_notebooks_to_object_list()#開発用
    # 24_data_iencetutorialforbeginnersでエラー
    msg = {
        'node_object_list': send_node_object_list, 
        'edges':send_edge_object_list,
        #'node_id_to_node_name': node_id_to_node_name, data_sample
        #"debug_data":request_data, 
        "debug_data":debug_data, 
        "err_msg": err_msg,
        "original_nb_name": original_nb_name,
        }
    return render(request, 'interface/result.html', msg)

def get_result(wm, w_c=1, w_v=1, w_l=1, w_d=1, k=10):
    wm = WorkflowMatching(psql_engine, graph_db, valid_nb_name_file_path=valid_nb_name_file_path) #検索のためにobjectを初期化
    wm.set_db_graph2(G_in_this_nb) # 検索対象を設定
    wm, node_id_to_node_name = build_QueryGraph(wm) # クエリを設定
        
    top_k_result, nb_score, search_time = search(wm, w_c=w_c, w_v=w_v, w_l=w_l, w_d=w_d, k=k) #検索

    send_node_object_list=arrange_node_object_list(QueryNode.objects.all())
    arranged_result = arrange_result_dict_for_html(jupyter_notebook_localhost_number, top_k_result, dict_nb_name_and_cleaned_nb_name)
    return node_id_to_node_name, nb_score, send_node_object_list, arranged_result, search_time

def arrange_node_object_list(node_object_list):
    send_node_object_list=[]
    for item in node_object_list:
        send_node_object_list.append({"node_id":item.node_id, "node_type": item.node_type, "node_contents": item.node_contents})
        #send_node_object_list.append([item.node_id, item.node_type, item.node_contents])
    return send_node_object_list

def arrange_edge_object_list(edge_object_list):
    send_edge_object_list=[]
    for item in edge_object_list:
        send_edge_object_list.append({"parent_node_id":item.parent_node_id, "successor_node_id":item.successor_node_id})
        #send_edge_object_list.append([item.parent_node_id, item.successor_node_id])
    return send_edge_object_list


def arrange_nodes_and_edges_of_result_notebooks_to_object_list(wm, nb_name):
    send_node_object_list=[]
    send_edge_object_list=[]
    nodesinlist = []
    edgesinlist = []
    for n in wm.G.nodes:
        if wm.attr_of_db_nb_name[n] == nb_name:
            node_start = n
    while(1):
        for n in wm.G.predecessors(node_start):
            node_start = n
            break
        #if len(wm.G.predecessors(node_start)) == 0:
        i=0
        for n in wm.G.predecessors(node_start):
            i+=1
            break
        if i==0:
            break

    def get_node_type_and_contents(wm, n):
        if wm.attr_of_db_node_type[n] == "Cell":
            #node_contents = "sample node contents"
            cleaned_nb_name=wm.attr_of_db_nb_name[n]
            n2_real_cell_id=wm.attr_of_db_real_cell_id[n]
            code_table_B=wm.fetch_source_code_table(cleaned_nb_name)
            codeN=code_table_B[code_table_B["cell_id"]==n2_real_cell_id]["cell_code"].values
            node_contents="".join(list(codeN))
            node_type = "code"
        elif wm.attr_of_db_node_type[n] == "Var":
            #node_contents = "sample node contents"
            node_type = "data"
            node_contents = get_data_as_csv(n)
        elif wm.attr_of_db_node_type[n] == "Display_data":
            #node_contents = "sample node contents"
            node_contents=""
            if n not in wm.attr_of_db_display_type:
                node_type = "text_output"
            elif wm.attr_of_db_display_type[n] == "text":
                node_type = "text_output"
            elif wm.attr_of_db_display_type[n] == "png":
                node_type = "figure_output"
            elif wm.attr_of_db_display_type[n] == "DataFrame":
                node_type = "table_output"
        return node_type, node_contents
    
    

    stack = [node_start]
    i = 0
    id_dict = {}
    while(len(stack)!=0):
        node_cur = stack.pop()
        id_dict.setdefault(node_cur, i)
        i+=1
        if id_dict[node_cur] not in nodesinlist:
            node_type, node_contents = get_node_type_and_contents(wm, node_cur)
            send_node_object_list.append({"node_id":id_dict[node_cur], "node_type": node_type, "node_contents": node_contents, "node_name":node_cur})
            nodesinlist.append(id_dict[node_cur])
        for n in wm.G.successors(node_cur):
            stack.append(n)
            id_dict.setdefault(n, i)
            i+=1
            if id_dict[n] not in nodesinlist:
                node_type, node_contents = get_node_type_and_contents(wm, n)
                send_node_object_list.append({"node_id":id_dict[n], "node_type": node_type, "node_contents": node_contents, "node_name":n})
                nodesinlist.append(id_dict[n])
            if (id_dict[node_cur], id_dict[n]) not in edgesinlist:
                send_edge_object_list.append({"parent_node_id":id_dict[node_cur], "successor_node_id":id_dict[n]})
                edgesinlist.append((id_dict[node_cur], id_dict[n]))
    return send_node_object_list, send_edge_object_list


def sample_nodes_and_edges_of_result_notebooks_to_object_list():
    """Sample workflow graph for development."""
    send_node_object_list = [{'node_id': 0, 'node_type': 'data', 'node_contents': 'sample node contents', "node_name":"10_df_housepriceprediction"}]
    send_edge_object_list=[]
    return send_node_object_list, send_edge_object_list


def build_QueryGraph(wm):
    """
    Initialize and return the instance of search engine.

    wm: a instance of WorkflowMatching class.
    """
    QueryGraph = nx.DiGraph()
    query_cell_code={}
    query_table={}

    # ノード名設定用の変数3つ. int.
    code_id=0
    data_id=0
    output_id=0
    reachability_id=0

    # edge設定用の{ノード番号:ノード名}の辞書. {int: string}.
    node_id_to_node_name={}
    query_workflow_info={"Cell": 0, "Var": 0, "Display_data": {"all":0}, "max_indegree": 0, "max_outdegree": 0}

    for node in QueryNode.objects.all():
        if node.node_type=="code":
            node_name = f"cell_query_{code_id}"
            QueryGraph.add_node(node_name, node_type="Cell", node_id=f"{node.node_id}", real_cell_id=f"{code_id}")
            query_cell_code[node_name] = node.node_contents
            node_id_to_node_name[node.node_id]=node_name
            code_id+=1
            query_workflow_info["Cell"]+=1
        elif node.node_type=="data":
            node_name = f"query_var{data_id}"
            QueryGraph.add_node(node_name, node_type="Var", node_id=f"{node.node_id}", data_type="pandas.core.frame.DataFrame")
            query_table[node_name] = pd.read_csv(io.StringIO(str(node.node_contents)), header=0)
            logging.info(query_table.keys())
            node_id_to_node_name[node.node_id]=node_name
            data_id+=1
            query_workflow_info["Var"]+=1
        elif node.node_type=="text_output":
            node_name = f"query_display{output_id}"
            QueryGraph.add_node(node_name, node_type="Display_data", display_type="text", node_id=f"{node.node_id}")
            node_id_to_node_name[node.node_id]=node_name
            output_id+=1
            if "text" not in query_workflow_info["Display_data"]:
                query_workflow_info["Display_data"]["text"]=0
            query_workflow_info["Display_data"]["text"]+=1
            query_workflow_info["Display_data"]["all"]+=1
        elif node.node_type=="figure_output":
            node_name = f"query_display{output_id}"
            QueryGraph.add_node(node_name, node_type="Display_data", display_type="png", node_id=f"{node.node_id}")
            node_id_to_node_name[node.node_id]=node_name
            output_id+=1
            if "png" not in query_workflow_info["Display_data"]:
                query_workflow_info["Display_data"]["png"]=0
            query_workflow_info["Display_data"]["png"]+=1
            query_workflow_info["Display_data"]["all"]+=1
        elif node.node_type=="table_output":
            node_name = f"query_display{output_id}"
            QueryGraph.add_node(node_name, node_type="Display_data", display_type="DataFrame", node_id=f"{node.node_id}")
            node_id_to_node_name[node.node_id]=node_name
            output_id+=1
            if "DataFrame" not in query_workflow_info["Display_data"]:
                query_workflow_info["Display_data"]["DataFrame"]=0
            query_workflow_info["Display_data"]["DataFrame"]+=1
            query_workflow_info["Display_data"]["all"]+=1
        elif node.node_type=="reachability":
            node_name = f"wildcard_{reachability_id}"
            QueryGraph.add_node(f"wildcard_{reachability_id}", node_type="AnyWildcard", node_id=f"{node.node_id}")
            node_id_to_node_name[node.node_id]=node_name
            reachability_id+=1
        #logging.info(f"{node_name} appended to QueryGraph.")
    

    library_list=[]
    for library in QueryLibrary.objects.all():
        library_list.append(library.library_name)
    query_lib=pd.Series(library_list)
    
    for edge in QueryEdge.objects.all():
        QueryGraph.add_edge(node_id_to_node_name[edge.parent_node_id], node_id_to_node_name[edge.successor_node_id])

    # Set max_indegree and max_outdegree
    for n in QueryGraph.nodes():
        indegree=0
        for i in QueryGraph.predecessors(n):
            indegree+=1
        query_workflow_info["max_indegree"]=max(indegree, query_workflow_info["max_indegree"])
        
        outdegree=0
        for i in QueryGraph.successors(n):
            outdegree+=1
        query_workflow_info["max_outdegree"]=max(outdegree, query_workflow_info["max_outdegree"])

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
    
    wm.QueryGraph = QueryGraph
    wm.query_root=query_root
    wm.query_lib=query_lib
    wm.query_cell_code=query_cell_code
    wm.query_table=query_table
    wm.attr_of_q_node_type = nx.get_node_attributes(wm.QueryGraph, "node_type")
    wm.attr_of_q_real_cell_id=nx.get_node_attributes(wm.QueryGraph, "real_cell_id")
    wm.attr_of_q_display_type=nx.get_node_attributes(wm.QueryGraph, "display_type")
    #wm.query_workflow_info={"Cell": 3, "Var": 0, "Display_data": {"text": 1}, "max_indegree": 1, "max_outdegree": 1}
    #wm.set_query_workflow_info()
    wm.query_workflow_info=query_workflow_info
    #wm.set_query_workflow_info_Display_data()

    logging.info("Completed!: Building a query graph.")
    return wm, node_id_to_node_name


def dump_to_json(QueryNode_object_list, QueryEdge_object_list, QueryLibrary_object_list):
    """
    QueryNode, QueryEdge, QueryLibraryをまとめて一つのjsonファイルにする．
    """
    list2json = {"querynode":[], "queryedge":[], "querylibrary":[]}
    for QueryNode_object in QueryNode_object_list:
        list2json["querynode"].append({"node_id": str(QueryNode_object.node_id), "node_type": QueryNode_object.node_type, "node_contents":str(QueryNode_object.node_contents)})
    for QueryEdge_object in QueryEdge_object_list:
        list2json["queryedge"].append({"parent_node_id": str(QueryEdge_object.parent_node_id), "successor_node_id": QueryEdge_object.successor_node_id})
    for QueryLibrary_object in QueryLibrary_object_list:
        list2json["querylibrary"].append({"library_name": str(QueryLibrary_object.library_name)})
    return json.dumps(list2json)

def replace_all_using_json_log(json_file:str):
    """
    json形式のクエリを読み込む．
    """
    delete_all()
    dictionary = json.loads(json_file)
    for item in dictionary["querynode"]:
        q = QueryNode(node_id=item["node_id"], node_type=item["node_type"], node_contents=item["node_contents"])
        q.save()
    for item in dictionary["queryedge"]:
        q = QueryEdge(parent_node_id=item["parent_node_id"], successor_node_id=item["successor_node_id"])
        q.save()
    for item in dictionary["querylibrary"]:
        q = QueryLibrary(library_name=item["library_name"])
        q.save()


def delete_all():
    """Flush the current query."""
    QueryNode.objects.all().delete()
    QueryEdge.objects.all().delete()
    QueryLibrary.objects.all().delete()



def save_query(saving_query_name, QueryNode_object_list, QueryEdge_object_list, QueryLibrary_object_list):
    """Savee the current query to SQLite."""
    save_json_file = dump_to_json(QueryNode_object_list, QueryEdge_object_list, QueryLibrary_object_list)
    q_json = QueryJson(query_name=saving_query_name, query_contents=save_json_file)
    q_json.save()

def search(wm, w_c, w_v, w_l, w_d, k, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_cache_query_table=False, save_running_time=False):
    """The same as searching_top_k_notebooks"""
    return searching_top_k_notebooks(wm, w_c=w_c, w_v=w_v, w_l=w_l, w_d=w_d, k=k, flg_chk_invalid_by_workflow_structure=flg_chk_invalid_by_workflow_structure, flg_flg_prune_under_sim=flg_flg_prune_under_sim, flg_optimize_calc_order=flg_optimize_calc_order, flg_caching=flg_caching, flg_cache_query_table=flg_cache_query_table, save_running_time=save_running_time)
   

# 時間計測あり 提案手法
#def new_proposal_method_bench_mark_4_2_with_wildcard_in_query(wm, w_c, w_v, w_l, w_d, k, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_calc_data_sim_approximately=False, flg_cache_query_table=False):
def searching_top_k_notebooks(wm, w_c, w_v, w_l, w_d, k, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_cache_query_table=False, save_running_time=False):
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
        flg_cache_query_table (bool)
    """

    search_time_start = timeit.default_timer()    
    wm.set_k(k)
    wm.get_db_workflow_info()
    #wm.load_calculated_sim(calculated_sim_path)
    wm.set_each_w(w_c=w_c, w_v=w_v, w_l=w_l, w_d=w_d)
    wm.set_flg_running_faster(flg_chk_invalid_by_workflow_structure, flg_flg_prune_under_sim, flg_optimize_calc_order, flg_caching, flg_cache_query_table=flg_cache_query_table)

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
        
    search_time_end = timeit.default_timer()    
    search_time = search_time_end - search_time_start
    return top_k_result, wm.nb_score, search_time


def create_jupyter_url(jupyter_notebook_localhost_number, nb_name):
    """検索結果のJupyter NotebookへのURLを生成．"""
    created_url = f"http://localhost:{jupyter_notebook_localhost_number}/tree/notebooks_data/{nb_name}"
    return created_url

def arrange_result_dict_for_html(jupyter_notebook_localhost_number, top_k_result, dict_nb_name_and_cleaned_nb_name):
    arranged_result = []
    rank=1
    for result in top_k_result:
        nb_name = dict_nb_name_and_cleaned_nb_name[result[0]]
        nb_score = result[1]
        nb_url = create_jupyter_url(jupyter_notebook_localhost_number, nb_name)
        arranged_result.append({"nb_name":nb_name,"nb_score":nb_score, "nb_url":nb_url, "rank":rank, "cleaned_nb_name":result[0]})
        rank+=1
    return arranged_result

def make_test_formset(request):
    TestFormSet = forms.formset_factory(
            form=QueryNode,
            extra=3,     # default-> 1
            max_num=4    # initial含めformは最大4となる
    )
    # 通常のformと同様に処理できる。
    if request.method == 'POST':
        formset = TestFormSet(request.POST)
        if formset.is_valid():
            # 参考として,cleaned_dataの中身を表示してみます。          
            data = repr(formset.cleaned_data)
            return HttpResponse(data) 
    else:
        formset = TestFormSet() # initialを渡すことができます。
        # formset = TestFormSet(initial=[{'title':'abc', 'date': '2019-01-01'},])
    return render(request, 'interface/form1.html', {'formset': formset})

def extract_library_name(libraries_contents, add_library_mode):
    """
    a wrapper func.
    """
    if add_library_mode == "mode_codes":
        return extract_library_name_modeA(libraries_contents)
    elif add_library_mode == "mode_librarynames":
        return extract_library_name_modeB(libraries_contents)

def extract_library_name_modeA(string_library):
    library_list=[]
    string_library = string_library.replace("\r", "")
    while("\n\n" in string_library):
        string_library = string_library.replace("\n\n", "\n")

    rows = string_library.split("\n")
    for row in rows:
        if "import " in row:
            pass
        else:
            continue
        if "#" in row:
            row = row[:row.find("#")]
        if row[:row.find(" ")+1]=="from":
            row = row[row.find(" import ")+8:]
        else:
            row = row[row.find("import ")+7:]
        if " as " in row:
            row = row[:row.rfind(" as ")+1]

        row = row.replace(" ", "")

        if "," in row:
            r_list=row.split(",")
            for r in r_list:
                library_list.append(r)
        else:
            library_list.append(row)

    return library_list

def extract_library_name_modeB(string_library):
    library_list=[]
    string_library = string_library.replace("\r", "")
    while("\n\n" in string_library):
        string_library = string_library.replace("\n\n", "\n")

    rows = string_library.split("\n")
    for row in rows:
        row = row.replace(" ", "")

        if "," in row:
            r_list=row.split(",")
            for r in r_list:
                library_list.append(r)
        else:
            library_list.append(row)

    return library_list

def save_libraries(libraries_list):
    for lib_name in libraries_list:
        if QueryLibrary.objects.filter(library_name=lib_name).exists():
            continue
        q = QueryLibrary(library_name=lib_name)
        q.save()


def func_loading_button(request_data):
    if "loading_button" in request_data:
        json_file = QueryJson.objects.filter(query_name=request_data["selected_query"])[0].query_contents
        replace_all_using_json_log(json_file)
    else:
        pass









