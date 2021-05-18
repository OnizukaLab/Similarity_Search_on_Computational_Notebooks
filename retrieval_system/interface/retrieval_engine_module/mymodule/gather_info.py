import os
import networkx as nx
import numpy as np
import pandas as pd
import sys
import nbformat as nbf
from  nbformat.v4.nbbase import (new_code_cell, new_markdown_cell)
import zipfile
import matplotlib.pyplot as plt
import ast
import json
import logging
import random
import subprocess
from threading import Lock



sys.path.append('/Users/runa/Desktop/大学/4年/実装/my_code/juneau_copy')
from juneau.config import config
from juneau.db.schemamapping import SchemaMapping
from juneau.db.table_db import generate_graph, pre_vars
from juneau.search.search_prov_code import ProvenanceSearch
from juneau.search.search_withprov import WithProv
from juneau.utils.funclister import FuncLister
from jupyter_client import BlockingKernelClient
from jupyter_client import find_connection_file
from juneau.jupyter import jupyter

jupyter_lock = Lock()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def gather_nbname(dir_path="../../データセット"):
    """
    dir_path内のJupyter Notebookファイル名を収集する．
    dir_pathのディレクトリは"DataSet"から始まるディレクトリで構成され，
    その中にipynbファイルが入っている必要がある．
    
    Args:
        dir_path (str): ディレクトリ名
    
    Returns:
        dict{str: str}: {ディレクトリ名: ファイル名}
    """
    dirs = os.listdir(dir_path)
    nb_list={}
    for i in range(len(dirs)):
        if ("zip" in dirs[i] or "." in dirs[i]):
            continue
        print("dir: ", dirs[i])
        files=os.listdir(dir_path+("/")+dirs[i])
        nb_list_in_the_dir=[]
        for f in files:
            if f[-6:]==".ipynb":
                print("\t- "+f)
                nb_list_in_the_dir.append(f)
        print("---------------- num of nb: ", len(nb_list_in_the_dir))
        nb_list[dirs[i]]=nb_list_in_the_dir
        print("\n")
    return nb_list

def gather_nbname2(dir_path="../../データセット"):
    """
    dir_path内のJupyter Notebookファイル名を収集する．
    dir_pathのディレクトリは"DataSet"から始まるディレクトリで構成され，
    その中にipynbファイルが入っている必要がある．
    
    Args:
        dir_path (str): ディレクトリ名
    
    Returns:
        dict{str: str}: {ディレクトリ名: ファイル名}
    """
    dirs = os.listdir(dir_path)
    nb_list={}
    for i in range(len(dirs)):
        if ("zip" in dirs[i] or "." in dirs[i]):
            continue
        #print("dir: ", dirs[i])
        files=os.listdir(dir_path+("/")+dirs[i])
        nb_list_in_the_dir=[]
        for f in files:
            if f[-6:]==".ipynb":
                #print("\t- "+f)
                nb_list_in_the_dir.append(f)
        #print("---------------- num of nb: ", len(nb_list_in_the_dir))
        nb_list[dirs[i]]=nb_list_in_the_dir
        #print("\n")
    return nb_list


def __parse_code(code_list):
    test = FuncLister()
    all_code = ""
    line2cid = {}
    

    lid = 1
    fflg = False # defとreturnの行の間だけTrue
    for cid, cell in enumerate(code_list):
        codes = cell.split("\\n")
        new_codes = []
        for code in codes:
            if code[:3].lower() == "def":
                fflg = True
                continue

            temp_code = code.strip(" ")
            temp_code = temp_code.strip("\t")

            if temp_code[:6].lower() == "return":
                fflg = False
                continue

            code = code.strip("\n")
            code = code.strip(" ")
            code = code.split('"')
            code = "'".join(code)
            code = code.split("\\")
            code = "".join(code)

            line2cid[lid] = cid
            lid = lid + 1
            if len(code) == 0:
                continue
            if code[0] == "%":
                continue
            if code[0] == "#":
                continue

            try:
                ast.parse(code)
                if not fflg:
                    new_codes.append(code)
            except:
                pass
                #logging.info(code)

        all_code = all_code + "\n".join(new_codes) + "\n"

    all_code = all_code.strip("\n")

    tree = ast.parse(all_code)
    test.visit(tree)
    return test.dependency, line2cid, all_code


def fetch_all_cell_codes(filepath):
    """
    Args:
        filepath (str): ipynbのpath
        
    Returns:
        
    """
    cellcode={}
    cid=1
    if not os.path.exists(filepath):
        raise Exception("File does not exist: "+filepath)
    # inform notebook file
    with open(filepath,"r") as r:
        nb = nbf.read(r,as_version=3) # nbconvert format version = 3
        for x in nb.worksheets:
            for cell in x.cells:
                if cell.cell_type =="code":
                    if not cell.language == 'python':
                        raise ValueError('Code must be in python!')
                    cellcode[cid]=cell.input.split("\n")
                cid+=1
    return cellcode

def fetch_all_display_data(filepath):
    """
    Args:
        filepath (str): ipynbのpath
        
    Returns:
        
    """
    display_data_type={}
    cid=1
    if not os.path.exists(filepath):
        raise Exception("File does not exist: "+filepath)
    # inform notebook file
    with open(filepath,"r") as r:
        nb = nbf.read(r,as_version=3) # nbconvert format version = 3
        for x in nb.worksheets:
            for cell in x.cells:
                if cell.cell_type =="code":
                    if not cell.language == 'python':
                        raise ValueError('Code must be in python!')
                    if cell.outputs:
                        output=cell.outputs
                        display_data_type[cid]=[]
                        for d in output:
                            #print(d.keys())
                            if "output_type" not in d:
                                continue
                            if "html" in d and "class=\"dataframe\"" in d["html"]:
                                display_data_type[cid].append("DataFrame")
                                #print("this datatype is dataframe")
                            elif d["output_type"] == "display_data" and "png" in d:
                                display_data_type[cid].append("png")
                                #print("this is png")
                            else:
                                display_data_type[cid].append("text")
                                #print(d)
                        display_data_type[cid]=list(set(display_data_type[cid]))
                        #print(display_data_type[cid])
                cid+=1
    return display_data_type

# 不使用
def just_exec_code(kid, code): 
    """
    Executes arbitrary `code` in the kernel with id `kid`. 
    カーネルIDが`kid`のカーネルで任意の`code`(引数)を実行．

    Args:
        kid (int): カーネルID.
        code (str): 実行したいpython言語ソースコード．

    Returns:
        error, if any.
    """
    # Load connection info and init communications.
    cf = find_connection_file(kid)

    with jupyter_lock: # ロック(排他制御)
        ##
        # BlockingKernelClientについては以下のドキュメントを参照. To know about BlockingKernelClient, refer to documentation below.
        # https://jupyter-client.readthedocs.io/en/stable/api/client.html?highlight=execute#jupyter_client.KernelClient.execute
        ##
        km = BlockingKernelClient(connection_file=cf)
        km.load_connection_file()
        km.start_channels() # channelを起動
        msg_id = km.execute(code, store_history=True) # 引数`code`を実行. 履歴に記録しない.
        reply = km.get_shell_msg(msg_id, timeout=60) # shell channelからメッセージを受け取る. Accept message from shell channel.
        error = None
        
        km.stop_channels() # channelを終了
        if reply["content"]["status"] != "ok": # エラーが発生した場合
            logging.error(f"Status is {reply['content']['status']}")
            output = None
    # ロック解除(排他制御)
    
    
def gather_var_list2(cellcode_list):
    """
    Args:
        cellcode (dict{int: list[str]}): 含まれる変数が知りたいソースコード(python).内容は{セルID: [改行で区切ったソースコード]}
    Returns:
        dict{int list[str]}: 引数cellcodeに含まれる変数のリスト．
    """
    var_list={}
    for i in cellcode_list:
        var_list[i]=[]
        dep, _1,_2=gather_info.__parse_code(cellcode_list[i])
        for j in dep:
            if type(dep[j][0][0]) is tuple:
                var_list[i].append(dep[j][0][0][0])
            else:
                var_list[i].append(dep[j][0][0])
            #print(dep[j][1])
            for k in range(len(dep[j][1])):
                if type(dep[j][1][k]) is tuple:
                    var_list[i].append(dep[j][1][k][0])
                    #print(dep[j][1][k][0])
                else:
                    var_list[i].append(dep[j][1][k])
                    #print(dep[j][1][k])
        var_list[i]=list(set(var_list[i]))
    return var_list