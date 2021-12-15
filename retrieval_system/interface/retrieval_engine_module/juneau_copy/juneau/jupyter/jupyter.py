# Copyright 2020 Juneau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module is in charge of defining functions that will
interact with the Jupyter kernel.

このモジュールによってJupyterカーネルを変更している．
"""

import logging
import os
import subprocess
from threading import Lock

from jupyter_client import BlockingKernelClient
from jupyter_client import find_connection_file

jupyter_lock = Lock()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# FIXME: Why are we printing a variable and calling `.to_json()`? If the type
#        of var is a list, this will throw an error because `.to_json()` is not a list function.
def request_var(kid, var):
    """
    Requests the contents of a dataframe or matrix by executing some code.
    変数名が引数varが表データか調べ，tupleを返す．
    引数varが，PandasのDataFrameの構造，NumPyのndarrayの構造,リストのいずれかである場合は，
    varをjson形式に変換したものを返す．
    上記のいずれでも無い場合は，errorを返す．

    Args:
        kid (int): カーネルID.
        var (str): 表またはリストの変数名．

    Return:
        output (json-style): 
            形式は{"columns":[(str)], "index":[(str)], "data":[(str)]}の形． 
            エラーがある場合はoutputはNone.
            参考(https://note.nkmk.me/python-pandas-to-json/).
        error: エラーが無い場合はerrorはNone.

    """
    code = f"""
        import pandas as pd
        import numpy as np
        if isinstance({var}, pd.DataFrame) or isinstance({var}, np.ndarray) or isinstance({var}, list):
            print({var}.to_json(orient='split', index=False))
        """
    return exec_code(kid, code)


def exec_code(kid, code):
    """
    Executes arbitrary `code` in the kernel with id `kid`. 
    カーネルIDが`kid`のカーネルで任意の`code`(引数)を実行．

    Args:
        kid (int): カーネルID.
        code (str): 実行したいpython言語ソースコード．

    Returns:
        tuple: the output of the code and the error, if any. (tuple: 引数codeを実行した際のの出力とエラーのタプル．エラーがある場合，出力はNoneになる．)
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
        msg_id = km.execute(code, store_history=False) # 引数`code`を実行. 履歴に記録しない.
        reply = km.get_shell_msg(msg_id, timeout=60) # shell channelからメッセージを受け取る. Accept message from shell channel.
        output, error = None, None

        while km.is_alive(): # カーネルがrunningの間以下を実行する.
            ##
            # IOPubについては以下のドキュメントを参照. To know about IOPub, refer to documentation below.
            # https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-pub-sub-channel
            ##
            msg = km.get_iopub_msg(timeout=10)
            if (
                "content" in msg
                and "name" in msg["content"]
                and msg["content"]["name"] == "stdout"
            ):
                output = msg["content"]["text"]
                break

        km.stop_channels() # channelを終了
        if reply["content"]["status"] != "ok": # エラーが発生した場合
            logging.error(f"Status is {reply['content']['status']}")
            logging.error(output)
            error = output
            output = None
    # ロック解除(排他制御)

    return output, error
    
def request_var_contents(kid, var_list, code_list): #新しく作成
    # Load connection info and init communications.
    cf = find_connection_file(kid)
    var_outputs={}
    #追記
    end_str="hogehoge43794allcodesaredone"
    flg=True

    with jupyter_lock: # ロック(排他制御)
        ##
        # BlockingKernelClientについては以下のドキュメントを参照. To know about BlockingKernelClient, refer to documentation below.
        # https://jupyter-client.readthedocs.io/en/stable/api/client.html?highlight=execute#jupyter_client.KernelClient.execute
        ##
        km = BlockingKernelClient(connection_file=cf)
        km.load_connection_file()
        km.start_channels() # channelを起動
        output, output2, output3, error = None, None, None, None

        #ノートブックから読み込んだセルを1つずつ実行するたびに変数の内容を得る．
        for i in code_list:
            reply1, reply2, reply3 = None, None, None
            logging.info(f"""code num {i} starts""")
            var_outputs[i]={}
            ##
            #ノートブックから読み込んだセルを1つずつ実行
            ##
            if "\n".join(code_list[i]) == "":
                continue
            code1="\n".join(code_list[i]+["""print(\'hogehoge43794allcodesaredone\')"""])
            logging.info(f"""running code num {i} ...""")
            flg=False
            msg_id1 = km.execute(code1, store_history=True) # 引数`code`を実行. 履歴に記録しない.
            #logging.info(f"""running code num {i} ended""")
            #logging.info(f"""getting code num {i} outputs...""")
            #reply1 = km.get_shell_msg(msg_id1, timeout=60) # shell channelからメッセージを受け取る. Accept message from shell channel.
            reply1 = km.get_shell_msg(msg_id1, timeout=None)
            while km.is_alive(): # カーネルがrunningの間以下を実行する.
                ##
                # IOPubについては以下のドキュメントを参照. To know about IOPub, refer to documentation below.
                # https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-pub-sub-channel
                ##
                msg = km.get_iopub_msg(timeout=None)
                #if (
                    #reply1["content"]["status"] == "ok"
                    #and "content" in msg
                    #and not "execution_state" in msg["content"]
                    #and not "execution_count" in msg["content"]
                #):
                if (
                    "content" in msg
                    and "text" in msg["content"]
                    and not "execution_state" in msg["content"]
                    and not "execution_count" in msg["content"]
                    and end_str in msg["content"] ["text"]
                ):
                    output1= msg["content"] ["text"]
                    flg=True
                    break
                    #and "text" in msg["content"]
                    #and end_str in msg["content"]["text"]
                    #):
                    #break
                    #output1 = msg["content"]
                    #break
            #chk_error(reply, output, error)
            #logging.info(f"""getting code num {i} outputs: {output1}""")
            logging.info(f"""running code num {i} done""")

            for j in range(len(var_list[i])):
                while flg==False:
                    pass
                logging.info(f"""code num {i}: collectiong var {var_list[i][j]} outputs starts""")
                var=var_list[i][j]
                ##
                #変数の型の取得
                ##
                code2=f"""
                    import pandas as pd
                    import numpy as np
                    print(type({var}))
                    """
                flg=False
                msg_id2 = km.execute(code2, store_history=False)
                reply2 = km.get_shell_msg(msg_id2, timeout=None)
                while km.is_alive():
                    msg = km.get_iopub_msg(timeout=None)
                    if (
                        reply1["content"]["status"] == "ok"
                        and reply2["content"]["status"] == "ok"
                        and "content" in msg
                        and "text" in msg["content"]
                        and not "execution_state" in msg["content"]
                        and not "execution_count" in msg["content"]
                    ):
                        output2 = msg["content"]
                        flg=True
                        break
                if "name" in output2 and "stdout" in output2["name"]:
                    output2=output2["text"].strip()
                #chk_error(reply, output, error)
                logging.info(f"""{var_list[i][j]} datatype: {output2}""")
                logging.info(f"""code num {i}: got type of var {var_list[i][j]}""")

                ##
                #変数の内容の取得
                ##
                code3=f"""
                    import pandas as pd
                    import numpy as np
                    import json
                    if isinstance({var}, pd.DataFrame) or isinstance({var}, np.ndarray):
                        print({var}.to_json(orient='split', index=False))
                    elif isinstance({var}, list):
                        print(json.dumps({var}))
                    else:
                        print({var})
                    """
                while flg==False:
                    pass
                flg=False
                msg_id3 = km.execute(code3, store_history=False)
                reply3 = km.get_shell_msg(msg_id3, timeout=None)
                while km.is_alive():
                    msg = km.get_iopub_msg(timeout=None)
                    if (
                        reply2["content"]["status"] == "ok"
                        and reply3["content"]["status"] == "ok"
                        and "content" in msg
                        and not "execution_state" in msg["content"]
                        and not "execution_count" in msg["content"]
                    ):
                        output3 = msg["content"]
                        flg=True
                        break
                #logging.info(f"""{var_list[i][j]} outputs: {output3}""")
                logging.info(f"""code num {i}: got contents of var {var_list[i][j]}""")
                var_outputs[i][var_list[i][j]]=[output2, output3]
                #chk_error(reply, output, error)

            logging.info(f"""code num {i} ended""")
        
        logging.info(f"""running all code ended""")
        km.stop_channels() # channelを終了
        
    # ロック解除(排他制御)

    return var_outputs, error
def just_get_var_contents(kid, var_list, code_list): #新しく作成
    # Load connection info and init communications.
    cf = find_connection_file(kid)
    var_outputs={}
    end_str="hogehoge43794allcodesaredone"
    flg=True

    with jupyter_lock: # ロック(排他制御)
        ##
        # BlockingKernelClientについては以下のドキュメントを参照. To know about BlockingKernelClient, refer to documentation below.
        # https://jupyter-client.readthedocs.io/en/stable/api/client.html?highlight=execute#jupyter_client.KernelClient.execute
        ##
        km = BlockingKernelClient(connection_file=cf)
        km.load_connection_file()
        km.start_channels() # channelを起動
        output, output2, output3, error = None, None, None, None

        #ノートブックから読み込んだセルを1つずつ実行するたびに変数の内容を得る．
        for i in code_list:
            reply1, reply2, reply3 = None, None, None
            logging.info(f"""code num {i} starts""")
            var_outputs[i]={}

            for j in range(len(var_list[i])):
                while flg==False:
                    pass
                logging.info(f"""code num {i}: collectiong var {var_list[i][j]} outputs starts""")
                var=var_list[i][j]
                ##
                #変数の型の取得
                ##
                code2=f"""
                    import pandas as pd
                    import numpy as np
                    print(type({var}))
                    """
                flg=False
                msg_id2 = km.execute(code2, store_history=False)
                reply2 = km.get_shell_msg(msg_id2, timeout=None)
                while km.is_alive():
                    msg = km.get_iopub_msg(timeout=None)
                    if (
                        reply1["content"]["status"] == "ok"
                        and reply2["content"]["status"] == "ok"
                        and "content" in msg
                        and "text" in msg["content"]
                        and not "execution_state" in msg["content"]
                        and not "execution_count" in msg["content"]
                    ):
                        output2 = msg["content"]
                        flg=True
                        break
                if "name" in output2 and "stdout" in output2["name"]:
                    output2=output2["text"].strip()
                #chk_error(reply, output, error)
                logging.info(f"""{var_list[i][j]} datatype: {output2}""")
                logging.info(f"""code num {i}: got type of var {var_list[i][j]}""")

                ##
                #変数の内容の取得
                ##
                code3=f"""
                    import pandas as pd
                    import numpy as np
                    import json
                    if isinstance({var}, pd.DataFrame) or isinstance({var}, np.ndarray):
                        print({var}.to_json(orient='split', index=False))
                    elif isinstance({var}, list):
                        print(json.dumps({var}))
                    else:
                        print({var})
                    """
                while flg==False:
                    pass
                flg=False
                msg_id3 = km.execute(code3, store_history=False)
                reply3 = km.get_shell_msg(msg_id3, timeout=None)
                while km.is_alive():
                    msg = km.get_iopub_msg(timeout=None)
                    if (
                        reply2["content"]["status"] == "ok"
                        and reply3["content"]["status"] == "ok"
                        and "content" in msg
                        and not "execution_state" in msg["content"]
                        and not "execution_count" in msg["content"]
                    ):
                        output3 = msg["content"]
                        flg=True
                        break
                #logging.info(f"""{var_list[i][j]} outputs: {output3}""")
                logging.info(f"""code num {i}: got contents of var {var_list[i][j]}""")
                var_outputs[i][var_list[i][j]]=[output2, output3]
                #chk_error(reply, output, error)

            logging.info(f"""code num {i} ended""")
        
        logging.info(f"""running all code ended""")
        km.stop_channels() # channelを終了
        
    # ロック解除(排他制御)

    return var_outputs

def exe_cell_contents(kid, one_cell_code_list): #新しく作成
    # km.start_channels()で開始したカーネルは，km.stop_channels() を実行すると変数の内容などが全てなくなることを確認
    # Load connection info and init communications.
    cf = find_connection_file(kid)
    end_str="hogehoge43794allcodesaredone"

    with jupyter_lock: # ロック(排他制御)
        ##
        # BlockingKernelClientについては以下のドキュメントを参照. To know about BlockingKernelClient, refer to documentation below.
        # https://jupyter-client.readthedocs.io/en/stable/api/client.html?highlight=execute#jupyter_client.KernelClient.execute
        ##
        km = BlockingKernelClient(connection_file=cf)
        km.load_connection_file()
        km.start_channels() # channelを起動
        
        code="\n".join(one_cell_code_list)
        if code != "":
            code="\n".join(one_cell_code_list+["""print(\"hogehoge43794allcodesaredone\")"""])
            msg_id = km.execute(code, store_history=False) # 引数`code`を実行. 履歴に記録しない.
            #reply1 = km.get_shell_msg(msg_id1, timeout=60) # shell channelからメッセージを受け取る. Accept message from shell channel.
            reply = km.get_shell_msg(msg_id, timeout=None)
            while km.is_alive(): # カーネルがrunningの間以下を実行する.
                ##
                # IOPubについては以下のドキュメントを参照. To know about IOPub, refer to documentation below.
                # https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-pub-sub-channel
                ##
                msg = km.get_iopub_msg(timeout=None)
                #logging.info(msg)
                if (
                    reply["content"]["status"] == "ok"
                    and "content" in msg
                    and "text" in msg["content"]
                    and not "execution_state" in msg["content"]
                    and not "execution_count" in msg["content"]
                    and end_str in msg["content"] ["text"]
                ):
                    output=msg["content"]
                    break
                #chk_error(reply, output, error)
            logging.info(f"""output: {output}""")
        logging.info(f"""running code ended""")
        km.stop_channels() # channelを終了  
    # ロック解除(排他制御)

def chk_error(reply, output, error):
    if reply["content"]["status"] != "ok": # エラーが発生した場合
            logging.error(f"Status is {reply['content']['status']}")
            logging.error(output)
            error = output
            output = None

def request_var_test(kid, var): # 試し用
    """
    変数名が引数varが表データか調べ，tripleを返す．

    Args:
        kid (int): カーネルID.
        var (str): 表またはリストの変数名．

    Returns:
        triple(outputs, var_type, error):
            var_typeはテーブルなら"table", テーブルでないなら"nottable".
            (tuple: 引数codeを実行した際のの出力とエラーのタプル．エラーがある場合，出力はNoneになる．)
    """
    code = f"""
        import pandas as pd
        import numpy as np
        if isinstance({var}, pd.DataFrame) or isinstance({var}, np.ndarray) or isinstance({var}, list):
            print({var}.to_json(orient='split', index=False))
            print(\"Hello world1\")
        """
    code2 = f"""
        if isinstance({var}, pd.DataFrame) or isinstance({var}, np.ndarray) or isinstance({var}, list):
            print(\"Hello world2\")
        """
    code3 = f"""
        print(type({var}))
        """
    # Load connection info and init communications.
    cf = find_connection_file(kid)
    var_type="not_table"

    with jupyter_lock: # ロック(排他制御)
        ##
        # BlockingKernelClientについては以下のドキュメントを参照. To know about BlockingKernelClient, refer to documentation below.
        # https://jupyter-client.readthedocs.io/en/stable/api/client.html?highlight=execute#jupyter_client.KernelClient.execute
        ##
        km = BlockingKernelClient(connection_file=cf)
        km.load_connection_file()
        km.start_channels() # channelを起動
        msg_id = km.execute(code, store_history=False) # 引数`code`を実行. 履歴に記録しない.
        reply = km.get_shell_msg(msg_id, timeout=60) # shell channelからメッセージを受け取る. Accept message from shell channel.
        output, output2, output3, error = None, None, None, None

        while km.is_alive(): # カーネルがrunningの間以下を実行する.
            ##
            # IOPubについては以下のドキュメントを参照. To know about IOPub, refer to documentation below.
            # https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-pub-sub-channel
            ##
            msg = km.get_iopub_msg(timeout=10)
            if (
                "content" in msg
                and not "execution_state" in msg["content"]
                and not "execution_count" in msg["content"]
            ):
                output = msg["content"]
                break
        
        msg_id2 = km.execute(code2, store_history=False) # 引数`code`を実行. 履歴に記録しない.
        #reply2 = km.get_shell_msg(msg_id2, timeout=60) # shell channelからメッセージを受け取る. Accept message from shell channel.
        while km.is_alive(): # カーネルがrunningの間以下を実行する.
            ##
            # IOPubについては以下のドキュメントを参照. To know about IOPub, refer to documentation below.
            # https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-pub-sub-channel
            ##
            msg = km.get_iopub_msg(timeout=10)
            if (
                "content" in msg
                and not "execution_state" in msg["content"]
                and not "execution_count" in msg["content"]
            ):
                output2 = msg["content"]
                break
        
        msg_id3 = km.execute(code3, store_history=False) # 引数`code`を実行. 履歴に記録しない.
        reply3 = km.get_shell_msg(msg_id3, timeout=60) # shell channelからメッセージを受け取る. Accept message from shell channel.
        while km.is_alive(): # カーネルがrunningの間以下を実行する.
            ##
            # IOPubについては以下のドキュメントを参照. To know about IOPub, refer to documentation below.
            # https://jupyter-client.readthedocs.io/en/stable/messaging.html#messages-on-the-iopub-pub-sub-channel
            ##
            msg = km.get_iopub_msg(timeout=10)
            if (
                "content" in msg
                and not "execution_state" in msg["content"]
                and not "execution_count" in msg["content"]
            ):
                output3 = msg["content"]
                break
              
        #if (
            #"content" in msg
            #and "name" in msg["content"]
            #and msg["content"]["name"] == "stdout"
        #):
            #var_type="table"
            #output = msg["content"]["text"]

        km.stop_channels() # channelを終了
        if reply["content"]["status"] != "ok": # エラーが発生した場合
            logging.error(f"Status is {reply['content']['status']}")
            logging.error(output)
            error = output
            output = None
    # ロック解除(排他制御)

    return output, output2, output3, var_type, error

def exec_connection_to_psql(kernel_id):
    """
    Runs the `connect_psql.py` script inside the Jupyter kernel.
    "db"ディレクトリの`connect_psql.py`をJupyterのカーネル内部で実行する．
    `connect_psql.py`は，juneau_connect()を定義するコードをkernel_idのカーネルで実行する．

    Args:
        kernel_id: The kernel id.

    Returns:
        tuple - the output of the code, and the error if any.
    """
    with jupyter_lock:
        psql_connection = os.path.join(os.path.join(BASE_DIR, "db"), "connect_psql.py") # juneau_connect()のReturns: engine.connect()
        # Popenについて参考(https://nansystem.com/python-call-vs-check-out-vs-run-vs-popen/)
        msg_id = subprocess.Popen( # msg_id: Popenインスタンス
            ["python3", psql_connection, kernel_id],
            stderr=subprocess.PIPE, # 標準エラー出力
            stdout=subprocess.PIPE, # 標準出力
        )
        output, error = msg_id.communicate()

    output = output.decode("utf-8") # UTF-8でデコード
    error = error.decode("utf-8")
    msg_id.stdout.close()
    msg_id.stderr.close()

    return output, error
