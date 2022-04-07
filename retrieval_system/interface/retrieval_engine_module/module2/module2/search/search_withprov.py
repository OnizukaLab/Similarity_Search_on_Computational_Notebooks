import logging
import timeit

import numpy as np
from py2neo import NodeMatcher

from module2.db.schemamapping import SchemaMapping
from module2.search.search_tables import SearchTables

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class WithProv(SearchTables):
    """
    関数`init_schema_mapping()`で初期化．
    初期化ではデータベース内のテーブルに対して，テーブルの従属関係によるグループ化のほか，
    テーブル名，列名，列のデータ値などをグループごとに整理し，各インスタンス変数に格納する．

    self.var str:
        self.schema_linking (dict{int: dict{str: int}}):
            テーブルグループそれぞれに対し，列名とsid(列名ごとの固有ID.schema IDの略?)の対応関係が格納されている．
            {テーブルグループID: {列名: sid}}
        self.schema_element (dict{int: dict{str: list[]}):
            テーブルグループごとの，各列名の列の実際のデータ値の重複無し集合．(null値を除く.)
            {テーブルグループID: {列名: [同じ列名を持つグループ内のすべてのテーブルから集めた実際のデータ値]}}
        self.schema_element_count (dict{int: dict{str: int-like}}):
            テーブルグループごとの，各列名が何個のテーブルに含まれているかの個数.
            (いくつのテーブルに共有されている列名か)．
        self.schema_element_dtype (dict{int: dict{str: str}}):
            テーブルの列に対するデータ型．{テーブルグループID: {列名: 列の型}}
        self.table_group (dict{str: int}):
            従属関係にあるテーブルをグループ化し，各テーブルにグループIDが
            割り当てられている．{各テーブル名(変数名): グループID}
    """
    schema_linking = {}
    schema_element = {}
    schema_element_count = {}
    schema_element_dtype = {}
    query_fd = {}
    table_group = {}

    def init_schema_mapping(self):

        logging.info("Start Reading From Neo4j!")
        matcher = NodeMatcher(self.geng)

        tables_touched = [] # list[str]: tables_connectedにすでに入れたテーブル名のリスト.
        tables_connected = [] # list[list[str]]: グループ(ワークフローグラフで接続された関係にあるもののグループ)ごとのテーブル名リストのリスト．
        # self.real_tables ({str: DataFrame}): (テーブル名:実際のテーブルの内容)の辞書. 
        # テーブル名はf"rtable{idi}"またはf"rtable{idi}_{vid}"となっている． 
        # idiはintまたはstr(変数名)か. vidはversion IDのことか.
        for i in self.real_tables.keys(): # i (str): テーブル名
            # テーブルとその従属関係があるテープルをすべて探す.
            if i[6:] not in set(tables_touched): # i[6:]: テーブル名から"rtables"を除いた部分.変数名か.
                logging.info(i)
                current_node = matcher.match("Var", name=i[6:]).first() # current_node (Node): 変数のノード
                connected_tables = self.dfs(current_node) # connected_tables (str): 変数 idi に従属関係のあるテーブルの変数名リスト.
                tables_touched = tables_touched + connected_tables # リストの連結
                tables_connected.append(connected_tables)

        self.schema_linking = {}
        self.schema_element = {}
        self.schema_element_count = {}
        self.schema_element_dtype = {}

        self.table_group = {}

        # assign each table a group id
        for idi, i in enumerate(tables_connected):
            for j in i:
                self.table_group[j] = idi # self.table_group[テーブル名(変数名または変数名_vid?)]=グループID

        for idi, i in enumerate(tables_connected):
            # idi (int): enumerateによる, ループごとにインクリメントする整数.
            # i (str): 変数名または変数名_vid. "rtable"+iでテーブル名になる.
            self.schema_linking[idi] = {}
            self.schema_element[idi] = {}
            self.schema_element_dtype[idi] = {}
            self.schema_element_count[idi] = {}

            for j in i: # 各テーブルに対し
                tname = "rtable" + j
                if tname not in self.real_tables: # テーブル名エラーの時
                    continue
                for col in self.real_tables[tname].columns: # テーブルの列それぞれに対して以下の操作
                    # テーブルの列名ごとに固有のsidを付与.
                    if col not in self.schema_linking[idi]:
                        if len(self.schema_linking[idi].keys()) == 0: # まだself.schema_linking[idi]に要素が入っていないとき(ループ1周目)
                            sid = 0
                        else:
                            sid = max(list(self.schema_linking[idi].values())) + 1

                        self.schema_linking[idi][col] = sid
                        self.schema_element_dtype[idi][col] = self.real_tables[tname][
                            col
                        ].dtype
                        self.schema_element_count[idi][col] = 1
                        self.schema_element[idi][col] = []
                        self.schema_element[idi][col] += self.real_tables[tname][col][
                            self.real_tables[tname][col].notnull() # Null値でないときにTrue
                        ].tolist()
                        self.schema_element[idi][col] = list(
                            set(self.schema_element[idi][col])
                        )
                    else:
                        self.schema_element[idi][col] += self.real_tables[tname][col][
                            self.real_tables[tname][col].notnull()
                        ].tolist()
                        self.schema_element[idi][col] = list(
                            set(self.schema_element[idi][col])
                        )
                        self.schema_element_count[idi][col] += 1

        logging.info("There are %s groups of tables." % len(tables_connected))

    def sketch_meta_mapping(self, sz=10):
        """
        Args:
            sz (int): 最大列数. defaults to 10.

        self.var str:
            self.schema_element_sample (dict{int: dict{}}): {テーブルグループID: }
        """
        self.schema_element_sample = {}
        for i in self.schema_element.keys(): # 各テーブルグループに対して
            self.schema_element_sample[i] = {}
            if len(self.schema_element[i].keys()) <= sz: #列数がsz以下である時
                for sc in self.schema_element[i].keys():
                    self.schema_element_sample[i][sc] = self.schema_element[i][sc]
            else:
                sc_choice = []
                for sc in self.schema_element[i].keys():
                    if sc == "Unnamed: 0" or "index" in sc:
                        continue
                    if self.schema_element_dtype[i][sc] is np.dtype(float):
                        continue
                    sc_value = list(self.schema_element[i][sc])
                    sc_choice.append(
                        (sc, float(len(set(sc_value))) / float(len(sc_value)))
                    )
                sc_choice = sorted(sc_choice, key=lambda d: d[1], reverse=True)

                count = 0
                for sc, v in sc_choice:
                    if count == sz:
                        break
                    self.schema_element_sample[i][sc] = self.schema_element[i][sc]
                    count += 1

    def sketch_query_cols(self, query, sz=10):
        #WithProvOptにそのまま有り．(オーバーライドでもない)
        """
        引数のテーブルqueryに対し，引数szで指定する最大列数以下のテーブルの列名をlistで返す．
        最大列数まで列数を減らすときは、その列の値ができるだけバラバラな値をとる順(|セット集合|/|多重集合|が大きい順)に残す．

        Args:
            query (ndarray): クエリ(テーブル).
            sz (int): 列数の最大サイズ．defaults to 5.

        Returns:
            list [str]: 最大列数以下の，テーブルの列名.
        """
        if query.shape[1] <= sz:
            return query.columns.tolist()
        else:
            q_cols = query.columns.tolist()
            c_scores = []
            for i in q_cols:
                if i == "Unnamed: 0" or "index" in i:
                    continue
                if query[i].dtype is np.dtype(float):
                    continue
                cs_v = query[i].tolist()
                c_scores.append((i, float(len(set(cs_v))) / float(len(cs_v))))
            c_scores = sorted(c_scores, key=lambda d: d[1], reverse=True)

            q_cols_chosen = []
            c_count = 0
            for i, j in c_scores:
                if c_count == sz:
                    break
                q_cols_chosen.append(i)
                c_count += 1
            return q_cols_chosen

    def search_similar_tables(
        self, query, beta, k, thres_key_cache, thres_key_prune, tflag=False
    ):

        self.query = query

        topk_tables = []
        SM_test = SchemaMapping()
        # groups_possibly_matched (list[int]): グループIDのリスト
        groups_possibly_matched = SM_test.mapping_naive_groups(
            self.query, self.schema_element_sample
        )
        self.query_fd = {}

        start_time1 = timeit.default_timer()
        time1 = 0
        start_time = timeit.default_timer()
        meta_mapping = SM_test.mapping_naive_tables(
            self.query, self.schema_element, groups_possibly_matched
        )
        end_time = timeit.default_timer()
        time1 += end_time - start_time

        time2 = 0
        time3 = 0

        for i in self.real_tables.keys():

            tname = i
            gid = self.table_group[tname[6:]]
            if gid not in meta_mapping:
                continue

            start_time = timeit.default_timer()
            SM = self.schema_mapping(
                self.query, self.real_tables[i], meta_mapping, gid
            )
            end_time = timeit.default_timer()
            time2 += end_time - start_time

            if len(SM) == 0:
                continue

            start_time = timeit.default_timer()
            table_sim = self.comp_table_similarity(
                self.query,
                self.real_tables[i],
                beta,
                SM,
                gid,
                thres_key_prune,
                thres_key_cache,
            )
            end_time = timeit.default_timer()
            time3 += end_time - start_time
            topk_tables.append((i, table_sim))

        topk_tables = sorted(topk_tables, key=lambda d: d[1], reverse=True)
        end_time1 = timeit.default_timer()
        time4 = end_time1 - start_time1

        if len(topk_tables) < k:
            k = len(topk_tables)

        rtables_names = self.remove_dup(topk_tables, k)

        if tflag:
            print("Schema Mapping Cost: ", time1 + time2)
            print("Similarity Computation Cost: ", time3)
            print("Totally Cost: ", time4)

        rtables = []
        for i in rtables_names:
            rtables.append((i, self.real_tables[i]))

        return rtables

    def search_similar_tables_threshold1(
        self, query, beta, k, theta, thres_key_cache, thres_key_prune, tflag=False
    ):

        self.query = query
        query_col = self.sketch_query_cols(query)

        self.already_map = {}
        for i in self.schema_linking.keys():
            self.already_map[i] = {}

        SM_test = SchemaMapping()
        groups_possibly_matched = SM_test.mapping_naive_groups(
            self.query, self.schema_element_sample
        )
        self.query_fd = {}

        start_time1 = timeit.default_timer()
        time1 = 0
        start_time = timeit.default_timer()
        meta_mapping = SM_test.mapping_naive_tables(
            self.query, query_col, self.schema_element, groups_possibly_matched
        )
        end_time = timeit.default_timer()
        time1 += end_time - start_time

        top_tables = []
        Cache_MaxSim = {}

        rank2 = []
        rank_candidate = []

        for i in self.real_tables.keys():

            tname = i
            gid = self.table_group[tname[6:]]
            if gid not in meta_mapping:
                continue

            tableS = self.query
            tableR = self.real_tables[i]

            start_time = timeit.default_timer()
            SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gid)
            end_time = timeit.default_timer()
            time1 = time1 + end_time - start_time

            Cache_MaxSim[tname] = ms

            if len(SM.items()) == 0:
                continue

            tableSnotintableR = []
            for sk in tableS.columns.tolist():
                if sk not in SM:
                    tableSnotintableR.append(sk)

            vname_score = float(1) / float(
                len(tableR.columns.values) + len(tableSnotintableR)
            )
            vname_score2 = float(len(SM.keys()) - 1) / float(
                len(tableR.columns.values) + len(tableSnotintableR) - 1
            )
            ubound = beta * vname_score2 + float(1 - beta) * Cache_MaxSim[tname]

            rank2.append(ubound)
            rank_candidate.append((tname, vname_score, SM))

        rank2 = sorted(rank2, reverse=True)
        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)

        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)

        for i in range(ks):
            tableR = self.real_tables[rank_candidate[i][0]]
            SM = rank_candidate[i][2]
            gid = self.table_group[rank_candidate[i][0][6:]]
            top_tables.append(
                (
                    rank_candidate[i][0],
                    self.comp_table_similarity(
                        self.query,
                        tableR,
                        beta,
                        SM,
                        gid,
                        thres_key_prune,
                        thres_key_cache,
                    ),
                )
            )

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
        min_value = top_tables[-1][1]

        ks = ks - 1
        id = 0
        while True:
            if ks + id >= len(rank_candidate):
                break

            threshold = beta * rank_candidate[ks + id][1] + rank2[0]

            if threshold <= min_value * theta:
                break
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    break

                tableR = self.real_tables[rank_candidate[ks + id][0]]
                SM = rank_candidate[ks + id][2]
                gid = self.table_group[rank_candidate[ks + id][0][6:]]
                new_score = self.comp_table_similarity(
                    self.query, tableR, beta, SM, gid, thres_key_cache, thres_key_cache
                )
                if new_score <= min_value:
                    continue
                else:
                    top_tables.append((rank_candidate[ks + id][0], new_score))
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
                    min_value = top_tables[ks][1]

        end_time1 = timeit.default_timer()
        time3 = end_time1 - start_time1

        if tflag == True:
            print("Schema Mapping Cost: ", time1)
            print("Totally Cost: ", time3)

        rtables_names = self.remove_dup(top_tables, ks)

        rtables = []
        for i in rtables_names:
            rtables.append((i, self.real_tables[i]))

        return rtables

    def search_joinable_tables(
        self, query, beta, k, thres_key_cache, thres_key_prune, tflag
    ):

        self.query = query
        self.already_map = {}
        for i in self.schema_linking.keys():
            self.already_map[i] = {}

        topk_tables = []
        SM_test = SchemaMapping()
        groups_possibly_matched = SM_test.mapping_naive_groups(
            self.query, self.schema_element_sample
        )
        self.query_fd = {}

        start_time1 = timeit.default_timer()
        time1 = 0
        start_time = timeit.default_timer()
        meta_mapping = SM_test.mapping_naive_tables(
            self.query, self.schema_element, groups_possibly_matched
        )
        end_time = timeit.default_timer()
        time1 += end_time - start_time
        time2 = 0
        time3 = 0

        for i in self.real_tables.keys():

            tname = i
            gid = self.table_group[tname[6:]]
            if gid not in meta_mapping:
                continue

            start_time = timeit.default_timer()
            SM = self.schema_mapping(
                self.query, self.real_tables[i], meta_mapping, gid
            )
            end_time = timeit.default_timer()
            time2 += end_time - start_time

            if len(SM) == 0:
                continue

            start_time = timeit.default_timer()
            table_sim = self.comp_table_joinable(
                self.query,
                self.real_tables[i],
                beta,
                SM,
                gid,
                thres_key_prune,
                thres_key_cache,
            )
            end_time = timeit.default_timer()
            time3 += end_time - start_time
            topk_tables.append((i, table_sim))

        topk_tables = sorted(topk_tables, key=lambda d: d[1], reverse=True)
        end_time1 = timeit.default_timer()
        time4 = end_time1 - start_time1

        if len(topk_tables) < k:
            k = len(topk_tables)

        rtables_names = self.remove_dup(topk_tables, k)

        if tflag == True:
            print("Schema Mapping Cost: ", time1 + time2)
            print("Joinability Computation Cost: ", time3)
            print("Totally Cost: ", time4)

        rtables = []
        for i in rtables_names:
            rtables.append((i, self.real_tables[i]))

        return rtables

    def search_joinable_tables_threshold1(
        self, query, beta, k, theta, thres_key_cache, thres_key_prune, tflag
    ):

        self.query = query
        self.already_map = {}
        for i in self.schema_linking.keys():
            self.already_map[i] = {}

        SM_test = SchemaMapping()
        groups_possibly_matched = SM_test.mapping_naive_groups(
            self.query, self.schema_element_sample
        )
        self.query_fd = {}

        start_time1 = timeit.default_timer()

        time1 = 0
        start_time = timeit.default_timer()
        meta_mapping = SM_test.mapping_naive_tables(
            self.query, self.schema_element, groups_possibly_matched
        )
        end_time = timeit.default_timer()
        time1 += end_time - start_time

        top_tables = []
        Cache_MaxSim = {}

        rank2 = []
        rank_candidate = []

        for i in self.real_tables.keys():

            tname = i
            gid = self.table_group[tname[6:]]
            if gid not in meta_mapping:
                continue

            tableS = self.query
            tableR = self.real_tables[i]

            start_time = timeit.default_timer()
            SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gid)
            end_time = timeit.default_timer()
            time1 = time1 + end_time - start_time

            Cache_MaxSim[tname] = ms

            if len(SM.items()) == 0:
                continue

            tableSnotintableR = []
            for sk in tableS.columns.tolist():
                if sk not in SM:
                    tableSnotintableR.append(sk)

            vname_score = float(1) / float(
                len(tableR.columns.values) + len(tableSnotintableR)
            )
            vname_score2 = float(
                max(len(tableR.columns.values), len(tableS.columns.values)) - 1
            ) / float(len(tableR.columns.values) + len(tableSnotintableR))
            ubound = beta * vname_score2 + float(1 - beta) * Cache_MaxSim[tname]

            rank2.append(ubound)
            rank_candidate.append((tname, vname_score, SM))

        rank2 = sorted(rank2, reverse=True)
        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)

        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)

        for i in range(ks):
            tableR = self.real_tables[rank_candidate[i][0]]
            SM = rank_candidate[i][2]
            gid = self.table_group[rank_candidate[i][0][6:]]
            top_tables.append(
                (
                    rank_candidate[i][0],
                    self.comp_table_joinable(
                        self.query,
                        tableR,
                        beta,
                        SM,
                        gid,
                        thres_key_prune,
                        thres_key_cache,
                    ),
                )
            )

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
        min_value = top_tables[-1][1]

        ks = ks - 1
        id = 0
        while True:
            if ks + id >= len(rank_candidate):
                break

            threshold = beta * rank_candidate[ks + id][1] + rank2[0]

            if threshold <= min_value * theta:
                break
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    break

                tableR = self.real_tables[rank_candidate[ks + id][0]]
                SM = rank_candidate[ks + id][2]
                gid = self.table_group[rank_candidate[ks + id][0][6:]]
                new_score = self.comp_table_joinable(
                    self.query, tableR, beta, SM, gid, thres_key_cache, thres_key_cache
                )
                if new_score <= min_value:
                    continue
                else:
                    top_tables.append((rank_candidate[ks + id][0], new_score))
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
                    min_value = top_tables[ks][1]

        end_time1 = timeit.default_timer()
        time3 = end_time1 - start_time1

        rtables_names = self.remove_dup(top_tables, k)

        if tflag == True:
            print("Schema Mapping Cost: ", time1)
            print("Totally Cost: ", time3)

        rtables = []
        for i in rtables_names:
            rtables.append((i, self.real_tables[i]))

        return rtables
