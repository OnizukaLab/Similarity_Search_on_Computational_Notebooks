import copy
import logging
import os
import pickle
import sys
from abc import abstractmethod

import numpy as np
import pandas as pd

from module2.db.table_db import (
    connect2db_engine,
    connect2gdb,
    fetch_all_table_names,
    fetch_all_views,
)


class SearchTables:
    """
    Args:
        dbname: PostgreSQLに接続するためのデータベース名．（= conf.sql.dbname）
        schema: データベースからテーブルを取ってくる際のスキーマ名指定．defaults to None.

    Attributes:
        query (DataFrame)
        eng: PostgreSQLのインスタンス．
        geng: Neo4jの接続．
        real_tables (list)
        already_map (list)

    self.var str:
        self.query (DataFrame)
        self.eng: PostgreSQLのインスタンス．
        self.geng: Neo4jの接続．
        self.real_tables ({str: DataFrame}): (テーブル名: 実際のテーブルの内容)の辞書．
        self.tables ([str]): データベースから取ってきたテーブル名のリスト．
    """
    # クラス変数
    query = None
    eng = None
    geng = None
    real_tables = []
    already_map = []

    def __init__(self, dbname, schema=None):
        self.query = None
        self.eng = connect2db_engine(dbname) # PostgreSQLのインスタンス
        self.geng = connect2gdb() # Neo4jの接続

        self.real_tables = {}
        conn = self.eng.connect() # PostgreSQLの接続

        if schema: # 引数にスキーマ名の指定がある時
            logging.info("Indexing existing tables from data lake")
            # 指定スキーマのすべてのテーブル名をconnで接続されるデータベースからPostgreSQLで取ってくる．
            # スキーマとは， SELECT ... FROM schema.table のように，テーブル名とドットで繋げられる前の部分．テーブルの上位階層に相当．
            self.tables = fetch_all_table_names(schema, conn)

            count = 0
            for i in self.tables:
                try:
                    table_r = pd.read_sql_table(i, conn, schema=schema) # SQLのテーブルをDataFrameに変換.
                    if "Unnamed: 0" in table_r.columns: #列名に不備がある場合はその列を削除
                        table_r.drop(["Unnamed: 0"], axis=1, inplace=True)
                    self.real_tables[i] = table_r # 実際のテーブルを格納
                    count = count + 1
                    if count % 20 == 0: # テーブル読み込み20個ごとにログを書き込む.
                        logging.info("Indexed " + str(count) + " tables...")
                except KeyboardInterrupt:
                    return
                except ValueError:
                    logging.info("Value error, skipping table " + i)
                    continue
                except TypeError:
                    logging.info("Type error, skipping table " + i)
                    continue
                except:
                    logging.info("Error, skipping table " + i)
                    logging.error("Unexpected error:", sys.exc_info()[0])
                    continue
        else: # 引数にschemaの指定がない時
            logging.info("Indexing views from data lake")
            self.tables = fetch_all_views(conn)  # self.eng)
            count = 0
            for i in self.tables:
                try:
                    table_r = pd.read_sql_table(i, conn)  # self.eng)
                    self.real_tables[i] = table_r # 実際のテーブルを格納
                    count = count + 1

                    if count % 20 == 0: # テーブル読み込み20個ごとにログを書き込む.
                        logging.info("Indexed " + str(count) + " tables...")
                except ValueError:
                    logging.info("Error, skipping table " + i)
                    continue
                except TypeError:
                    logging.info("Type error, skipping table " + i)
                    continue
                except KeyboardInterrupt:
                    return
                except:
                    logging.info("Error, skipping table " + i)
                    logging.error("Unexpected error:", sys.exc_info()[0])
                    continue

        conn.close() # PostgreSQLの接続を解除.
        logging.info(
            "%s tables detected in the database." % len(self.real_tables.keys())
        )

        self.init_schema_mapping() # この関数はWithProvでオーバーライドで定義されている．

    @staticmethod
    def line2cid(directory):
        nb_l2c = {}
        files = os.listdir(directory)
        for f in files:
            ind = pickle.load(open(os.path.join(directory, f), "rb"))
            nb_l2c[f[:-2]] = ind
        return nb_l2c

    @staticmethod
    def col_similarity(tableA, tableB, SM, key_factor):
        """
        Args:
            tableA : 表形式データ
            tableB : 表形式データ
            SM (dict): スキーママッピング(2つのテーブルのキー(列名)の対応関係)。おそらく{str: str}で、{tableAのキー: tableBのキー}。
            key_factor (float): key_scoreに相当
        
        Returns:
            float: 列の類似度
        """
        col_sim_upper = 1 + float(len(SM.keys()) - 1) * float(key_factor)
        tableA_not_in_tableB = [] # キーのリスト
        for kyA in tableA.columns.tolist():
            if kyA not in SM:
                tableA_not_in_tableB.append(kyA)
        col_sim_lower = len(tableB.columns.values) + len(tableA_not_in_tableB) # tableAとtableBのキーの重複無し集合
        col_sim = float(col_sim_upper) / float(col_sim_lower)
        return col_sim

    @staticmethod
    def row_similarity(colA, colB):
        colA_value = colA[~pd.isnull(colA)].values
        colB_value = colB[~pd.isnull(colB)].values
        row_sim_upper = len(np.intersect1d(colA_value, colB_value))
        row_sim_lower = len(np.union1d(colA_value, colB_value))
        row_sim = float(row_sim_upper) / float(row_sim_lower)
        return row_sim

    def remove_dup(self, ranked_list, ks):
        """
        3要素リストの重複を取り除く．

        Args:
            ranked_list (list[any, any, any]): 重複を含んでいる可能性がある3要素リスト．
            ks (int): リストの最大サイズ.
        
        Returns:
            ranked_list (list[any, any, any]): 最大サイズksの重複を含まない3要素リスト．
        """
        res = []
        for i, j, l in ranked_list:
            flg = True
            for k, m in res: #i,l
                if self.real_tables[i].equals(self.real_tables[k]): #iとkのテーブルが同じとき
                    flg = False
                    break
            if flg:
                res.append((i, l))

            if len(res) == ks:
                break
        return res

    def remove_dup2(self, ranked_list, ks):
        """
        2要素リストの重複を取り除く．

        Args:
            ranked_list (list[any, any]): 重複を含んでいる可能性がある2要素リスト．
            ks (int): リストの最大サイズ.
        
        Returns:
            ranked_list (list[any, any]): 最大サイズksの重複を含まない2要素リスト．
        """
        res = []
        for i, j in ranked_list:
            flg = True
            for k in res:
                if self.real_tables[i].equals(self.real_tables[k]):
                    flg = False
                    break
            if flg == True:
                res.append(i)

            if len(res) == ks:
                break
        return res

    def dfs(self, snode) -> list:
        """
        引数のノードsnodeを含むセルが属するワークフローグラフに対し，
        そのグラフに含まれるすべての変数ノードを深さ優先探索で調べ，変数ノード名のリストを返す．
        (グループIDの決定に利用)

        Args:
            snode (Node): 変数のノード．
        
        Returns:
            list[str]: 引数のノードとその変数を含むセルが属するグラフ内に含まれるすべての変数ノード名の重複無しリスト．
        """

        # cell_rel: セルのノードsnodeに対しリレーション""Containedby"で接続するノード間のリレーションの1つ．
        # (つまり, ノードsnodeの元となるセルに含まれる変数のノード間のリレーション)
        cell_rel = self.geng.match_one((snode,), r_type="Containedby") #py2neoのmatcher
        if not cell_rel: # 1つも該当ノード(snodeとリレーション""Containedby"で接続するノード)がなかった場合
            return []
        cell_node = cell_rel.end_node
        names = []
        return_names = []
        stack = [cell_node]
        while True:
            if len(stack) != 0:
                node = stack.pop()
                names.append(node["name"])
                for rel in self.geng.match((node,), r_type="Contains"): #nodeからリレーション"Contains"で繋がるノード(変数のノード)のリストの各要素に対して
                    return_names.append(rel.end_node["name"]) # 変数のノードのノード名をリストに追加

                # nodeからリレーション"Successor"または"Parent"で繋がるノードのうち,
                # まだ調べていないノード(リストnamesにまだノード名が入ってないノード)をスタックに追加
                for rel in self.geng.match((node,), r_type="Successor"):
                    if rel.end_node["name"] in names:
                        continue
                    else:
                        stack.append(rel.end_node)

                for rel in self.geng.match((node,), r_type="Parent"):
                    if rel.end_node["name"] in names:
                        continue
                    else:
                        stack.append(rel.end_node)
            else:
                break

        return list(set(return_names))

    @abstractmethod
    def init_schema_mapping(self):
        """
        Abstract function to init the schema mapping.
        """
        return

    def app_common_key(self, tableA, tableB, SM, key, thres_prune):
        kyA = key
        kyB = SM[key]
        key_value_A = tableA[kyA].tolist()
        key_value_B = tableB[kyB].tolist()

        key_estimateA = float(len(set(key_value_A))) / float(len(key_value_A))
        key_estimateB = float(len(set(key_value_B))) / float(len(key_value_B))
        if min(key_estimateA, key_estimateB) <= thres_prune:
            return 0

        mapped_keyA = list(SM.keys())

        if kyA not in self.query_fd:
            self.query_fd[kyA] = {}
            for idv, kv in enumerate(key_value_A):
                if kv not in self.query_fd[kyA]:
                    self.query_fd[kyA][kv] = []
                self.query_fd[kyA][kv].append(
                    ",".join(map(str, tableA[mapped_keyA].iloc[idv].tolist()))
                )
            fd = copy.deepcopy(self.query_fd[key])
        else:
            fd = copy.deepcopy(self.query_fd[key])

        mapped_keyB = list(SM.values())
        for idv, kv in enumerate(key_value_B):
            if kv not in fd:
                fd[kv] = []
            fd[kv].append(",".join(map(str, tableB[mapped_keyB].iloc[idv].tolist())))

        key_score = 0
        for fdk in fd.keys():
            key_score = key_score + float(len(set(fd[fdk]))) / float(
                tableA.shape[0] + tableB.shape[0]
            )

        return key_score

    @staticmethod
    def comp_table_similarity_row(tableA, tableB, key, SM, sample_size):

        value_setb = set(tableB[SM[key]].tolist())

        sim_sum = 0
        sim_count = 0
        for idi, i in enumerate(tableA[key].tolist()):
            if sim_count == sample_size:
                break
            if i in value_setb:
                dfB = tableB.loc[tableB[SM[key]] == i]
                sim_array = []
                for index, row in dfB.iterrows():
                    a = np.array(list(map(str, tableA.iloc[idi].tolist())))
                    b = np.array(list(map(str, row.tolist())))
                    try:
                        sim_array.append(
                            float(len(np.intersect1d(a, b)))
                            / float(len(np.union1d(a, b)))
                        )
                    except KeyboardInterrupt:
                        return 0
                sim_array = np.array(sim_array)
                sim_sum = sim_sum + np.mean(sim_array)
                sim_count += 1
        if sim_count == 0:
            return 0
        else:
            return float(sim_sum) / float(sim_count)

    def comp_table_similarity(
        self, tableA, tableB, beta, SM, gid, thres_key_prune, thres_key_cache
    ):

        key_choice = []
        for kyA in SM.keys():
            flg = False
            if kyA in self.already_map[gid]:
                determined = self.already_map[gid][kyA]
                check_set = set(list(SM.values()))
                for ds in determined:
                    if ds.issubset(check_set):
                        flg = True
                        break
                if flg:
                    key_choice.append(
                        (
                            kyA,
                            self.app_common_key(
                                tableA, tableB, SM, kyA, thres_key_prune
                            ),
                        )
                    )
                    break

                else:
                    key_score = self.app_common_key(
                        tableA, tableB, SM, kyA, thres_key_prune
                    )
                    key_choice.append((kyA, key_score))
                    if key_score == 1:
                        break
            else:
                key_score = self.app_common_key(
                    tableA, tableB, SM, kyA, thres_key_prune
                )
                key_choice.append((kyA, key_score))

        if len(key_choice) == 0:
            return 0
        else:
            key_choice = sorted(key_choice, key=lambda d: d[1], reverse=True) # key_scoreでソート
            key_chosen = key_choice[0][0] # kyA (argmax key_score)に相当
            key_factor = key_choice[0][1] # max key_scoreに相当

            if key_factor >= thres_key_cache:
                if key_chosen not in self.already_map[gid]:
                    self.already_map[gid][key_chosen] = []
                self.already_map[gid][key_chosen].append(set(list(SM.values())))

            row_sim = self.row_similarity(tableA[key_chosen], tableB[SM[key_chosen]])
            col_sim = self.col_similarity(tableA, tableB, SM, key_factor)

            return beta * col_sim + float(1 - beta) * row_sim

    def comp_table_joinable(
        self, tableA, tableB, beta, SM, gid, thres_key_prune, thres_key_cache
    ):

        key_choice = []
        for kyA in SM.keys():
            flg = False
            if kyA in self.already_map[gid]:
                determined = self.already_map[gid][kyA]
                check_set = set(list(SM.values()))
                for ds in determined:
                    if ds.issubset(check_set):
                        flg = True
                        break
                if flg:
                    key_choice.append(
                        (
                            kyA,
                            self.app_common_key(
                                tableA, tableB, SM, kyA, thres_key_prune
                            ),
                        )
                    )
                    break

                else:
                    key_score = self.app_common_key(
                        tableA, tableB, SM, kyA, thres_key_prune
                    )
                    key_choice.append((kyA, key_score))
                    if key_score == 1:
                        break
            else:
                key_score = self.app_common_key(
                    tableA, tableB, SM, kyA, thres_key_prune
                )
                key_choice.append((kyA, key_score))

        key_choice = sorted(key_choice, key=lambda d: d[1], reverse=True)
        key_chosen = key_choice[0][0]
        key_factor = key_choice[0][1]

        if key_factor >= thres_key_cache:
            if key_chosen not in self.already_map[gid]:
                self.already_map[gid][key_chosen] = []
            self.already_map[gid][key_chosen].append(set(list(SM.values())))

        row_sim = self.row_similarity(tableA[key_chosen], tableB[SM[key_chosen]])

        col_sim_upper = 1 + float(len(tableA.columns.values) - 1) * float(key_factor)
        tableA_not_in_tableB = []
        for kyA in tableA.columns.tolist():
            if kyA not in SM:
                tableA_not_in_tableB.append(kyA)
        col_sim_lower = len(tableB.columns.values) + len(tableA_not_in_tableB)
        col_sim = float(col_sim_upper) / float(col_sim_lower)

        col_sim_upper2 = 1 + float(len(tableB.columns.values) - 1) * float(key_factor)
        col_sim2 = float(col_sim_upper2) / float(col_sim_lower)

        score1 = beta * col_sim + float(1 - beta) * row_sim
        score2 = beta * col_sim2 + float(1 - beta) * row_sim

        return max(score1, score2)

    def comp_table_joinable_key(
        self,
        SM_test,
        tableA,
        tableB,
        beta,
        SM,
        gid,
        meta_mapping,
        schema_linking,
        thres_key_prune,
        unmatched,
    ):
        """
        sim^{mu}_{row}(S,T)を求める．
        これは論文の式(17(18?))に相当．
        
        Args:
            SM_test (instance Schemamapping)
            tableA (DataFrame)
            tableB (DataFrame)
            beta (float)
            SM (dict{int: str}): tableAとtableBで共通する列のSID(Schema ID?)と列名のリスト(辞書).内容は{sid: 列名}
            gid (int): グループID
            meta_mapping (dict{int: dict{str: str}}):   
                列Aと列Bのデータ集合のジャカード類似度のうち，  
                閾値self.sim_thresより高い組み合わせの列名のリスト．  
                内容は{グループID: {列名B: 列名A}}
            schema_linking (dict{int: dict{str: int}}):
                テーブルグループそれぞれに対し，列名とsid(列名ごとの固有ID.schema IDの略?)の対応関係が格納されている．
                {テーブルグループID: {列名: sid}}
            thres_key_prune
            unmatched (dict{int: dict{str: dict{str: str}}}):
                閾値よりも類似度の値が小さい列の組み合わせ．
                内容は，{グループID: {列名: {列名: ""}}}
                列名nameAと列名nameBの列のジャカード類似度が閾値より低い場合は，unmatched[group][nameA][nameB] = ""とする．
        Returns:
            max(score1, score2) (float)
            meta_mapping(dict{int: dict{str: str}}):   
                列Aと列Bのデータ集合のジャカード類似度のうち，  
                閾値self.sim_thresより高い組み合わせの列名のリスト．  
                内容は{グループID: {列名B: 列名A}}
            unmatched (dict{int: dict{str: dict{str: str}}}):
                閾値よりも類似度の値が小さい列の組み合わせ．
                内容は，{グループID: {列名: {列名: ""}}}
                列名nameAと列名nameBの列のジャカード類似度が閾値より低い場合は，unmatched[group][nameA][nameB] = ""とする．
            sm_time
            key_chosen (int): 最もスコアが高いSID(列ID)
            
        """

        key_choice = [] #list[int, float]: [SID, score]
        for kyA in SM.keys():
            key_score = self.app_join_key(tableA, tableB, SM, kyA, thres_key_prune)
            key_choice.append((kyA, key_score))

        if len(key_choice) == 0:
            return 0, meta_mapping, unmatched, 0, None
        else:
            key_choice = sorted(key_choice, key=lambda d: d[1], reverse=True) # scoreで降順にソート
            key_chosen = key_choice[0][0]
            key_factor = key_choice[0][1]

            (
                SM_real,
                meta_mapping,
                unmatched,
                sm_time,
            ) = SM_test.mapping_naive_incremental(
                tableA, tableB, gid, meta_mapping, schema_linking, unmatched, mapped=SM
            )  # SM_test.mapping_naive(tableA, tableB, SM)

            #1+tableAの列数*score
            col_sim_upper = 1 + float(len(tableA.columns.values) - 1) * float(
                key_factor
            )
            # tableAにあるがtableBにはない列を探す.
            tableA_not_in_tableB = []
            for kyA in tableA.columns.tolist():
                if kyA not in SM_real:
                    tableA_not_in_tableB.append(kyA)
            col_sim_lower = len(tableB.columns.values) + len(tableA_not_in_tableB) - 1
            # 式(2) (のちに式(17)の計算のために利用)
            col_sim = float(col_sim_upper) / float(col_sim_lower)

            #1+tableBの列数*score
            col_sim_upper2 = 1 + float(len(tableB.columns.values) - 1) * float(
                key_factor
            )
            # 式(2) (のちに式(17)の計算のために利用)
            col_sim2 = float(col_sim_upper2) / float(col_sim_lower)

            row_sim = self.row_similarity(
                tableA[key_chosen], tableB[SM_real[key_chosen]]
            )

            # 式(17)?
            # 疑問: betaと1-beta逆では?
            score1 = beta * col_sim + float(1 - beta) * row_sim
            score2 = beta * col_sim2 + float(1 - beta) * row_sim

            return max(score1, score2), meta_mapping, unmatched, sm_time, key_chosen

    def app_join_key(self, tableA, tableB, SM, key, thres_prune):

        kyA = key
        key_value_A = tableA[key].tolist()
        scoreA = float(len(set(key_value_A))) / float(len(key_value_A))
        if scoreA == 1:
            return 1

        kyB = SM[key]
        key_value_B = tableB[kyB].tolist()
        scoreB = float(len(set(key_value_B))) / float(len(key_value_B))
        if scoreB == 1:
            return 1

        if min(scoreA, scoreB) < thres_prune:
            return 0

        mapped_keyA = list(SM.keys())

        if kyA not in self.query_fd:
            self.query_fd[kyA] = {}
            for idv, kv in enumerate(key_value_A):
                if kv not in self.query_fd[kyA]:
                    self.query_fd[kyA][kv] = []
                self.query_fd[kyA][kv].append(
                    ",".join(map(str, tableA[mapped_keyA].iloc[idv].tolist()))
                )
            fd = copy.deepcopy(self.query_fd[key])
        else:
            fd = copy.deepcopy(self.query_fd[key])

        mapped_keyB = list(SM.values())
        for idv, kv in enumerate(key_value_B):
            if kv in fd:
                fd[kv].append(
                    ",".join(map(str, tableB[mapped_keyB].iloc[idv].tolist()))
                )

        key_scoreAB = 0
        for fdk in fd.keys():
            key_scoreAB += float(len(set(fd[fdk]))) / float(
                tableA.shape[0] + tableB.shape[0]
            )

        temp_fd = {}
        for idv, kv in enumerate(key_value_B):
            if kv not in temp_fd:
                temp_fd[kv] = []
            temp_fd[kv].append(
                ",".join(map(str, tableB[mapped_keyB].iloc[idv].tolist()))
            )

        for idv, kv in enumerate(key_value_A):
            if kv in temp_fd:
                temp_fd[kv].append(
                    ",".join(map(str, tableA[mapped_keyA].iloc[idv].tolist()))
                )

        key_scoreBA = 0
        for fdk in temp_fd.keys():
            key_scoreBA += float(len(set(temp_fd[fdk]))) / float(
                tableA.shape[0] + tableB.shape[0]
            )

        return max(key_scoreAB, key_scoreBA)

    def comp_table_similarity_key(
        self,
        SM_test,
        tableA,
        tableB,
        SM,
        gid,
        meta_mapping,
        schema_linking,
        thres_key_prune,
        thres_key_cache,
        unmatched,
    ):

        key_choice = []
        for kyA in SM.keys():
            flg = False
            if kyA in self.already_map[gid]:
                determined = self.already_map[gid][kyA]
                check_set = set(list(SM.values()))
                for ds in determined:
                    if ds.issubset(check_set):
                        flg = True
                        break
                if flg:
                    key_choice.append(
                        (
                            kyA,
                            self.app_common_key(
                                tableA, tableB, SM, kyA, thres_key_prune
                            ),
                        )
                    )
                    break

                else:
                    key_score = self.app_common_key(
                        tableA, tableB, SM, kyA, thres_key_prune
                    )
                    key_choice.append((kyA, key_score))
                    if key_score == 1:
                        break
            else:
                key_score = self.app_common_key(
                    tableA, tableB, SM, kyA, thres_key_prune
                )
                key_choice.append((kyA, key_score))

        if len(key_choice) == 0:
            return 0, meta_mapping, unmatched, 0, None
        else:
            key_choice = sorted(key_choice, key=lambda d: d[1], reverse=True)
            key_chosen = key_choice[0][0]
            key_factor = key_choice[0][1]

            if key_factor >= thres_key_cache:
                if key_chosen not in self.already_map[gid]:
                    self.already_map[gid][key_chosen] = []
                self.already_map[gid][key_chosen].append(set(list(SM.values())))

            (
                SM_real,
                meta_mapping,
                unmatched,
                sm_time,
            ) = SM_test.mapping_naive_incremental(
                tableA, tableB, gid, meta_mapping, schema_linking, unmatched, mapped=SM
            )

            row_sim = self.row_similarity(
                tableA[key_chosen], tableB[SM_real[key_chosen]]
            )
            col_sim = self.col_similarity(tableA, tableB, SM_real, key_factor)

            return col_sim, row_sim, meta_mapping, unmatched, sm_time, key_chosen
