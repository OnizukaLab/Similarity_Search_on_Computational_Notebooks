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
Handling of both indexing and search which helps us
understand the relationship between two tables.
"""

import numpy as np
import pandas as pd
import timeit


class SchemaMapping:
    # 関数によってMpairの型が違うことに注意(グループIDが入っているかどうか)
    """
    Args:
        sim_thres (float): ジャカード類似度の閾値.defaults to 0.3.
    
    self.var str:
        self.sim_thres (float): ジャカード類似度の閾値.defaults to 0.3.
    """
    def __init__(self, sim_thres=0.3):
        self.sim_thres = sim_thres

    @staticmethod
    def jaccard_similarity(colA, colB):
        """
        The Jaccard similarity between two sets A and B is defined as
        |intersection(A, B)| / |union(A, B)|.
        colAとcolBのジャカード類似度を計算する．

        Args:
            colA list[any]: データ値のリスト.
            colB list[any]: データ値のリスト.

        Returns:
            float: ジャカード類似度．
        """
        try:
            if min(len(colA), len(colB)) == 0:
                return 0
            colA = np.array(colA) #numpyの型に変換?
            # 疑問: colBは変換しなくて良いのか？
            union = len(np.union1d(colA, colB))
            inter = len(np.intersect1d(colA, colB))
            return float(inter) / float(union)
        except:
            return 0

    def mapping_naive(self, tableA, tableB, mapped=None):
        #疑問: MpairR不要ではないか?
        """
        tableAの列とtableBの列のジャカード類似度が閾値より高い，列の組み合わせを返す．
        類似度計算の際はNullまたはNan値は除いている．

        Args:
            tableA (DataFrame): テーブル.
            tableB (DataFrame): テーブル.
            mapped dict{int: dict{str: str}}:
                すでに列間の類似度が閾値より高いとわかっている列名のペアのリスト(辞書)．
                内容は{列名A: 列名B}
                defaults to None.
        
        Returns:
            dict{str: str}:
                列Aと列Bのデータ集合のジャカード類似度のうち，閾値self.sim_thresより高い組み合わせについて，
                類似度が高い順にソートした列名のリスト．{列名A: 列名B}.
            float: tableAの列とtableBの列のすべての組み合わせの類似度のうち，最大の類似度．

        self.var str:
            self.sim_thres (float): 列間のジャカード類似度の閾値．
        """
        start_time = timeit.default_timer()
        time1 = 0
        c1 = 0

        matching = []
        Mpair = mapped # dict{str: str}: 内容は{列名A: 列名B}
        MpairR = {}
        for i in Mpair.keys():
            MpairR[Mpair[i]] = i

        scma = tableA.columns.values
        scmb = tableB.columns.values
        shmal = len(scma)
        shmbl = len(scmb)

        acol_set = {}

        # tableAの各列とtableBの各列の組み合わせのジャカード類似度の計算
        # 計算結果はmatchingに格納
        for i in range(shmal):

            nameA = scma[i]
            if nameA in Mpair or nameA == "Unnamed: 0" or "index" in nameA:
                continue

            colA = tableA[scma[i]][~pd.isnull(tableA[scma[i]])].values
            if nameA not in acol_set:
                acol_set[nameA] = list(set(colA))

            for j in range(shmbl):

                nameB = scmb[j]
                if nameB in MpairR or nameB == "Unnamed: 0" or "index" in nameB or \
                        tableA[scma[i]].dtype != tableB[scmb[j]].dtype:
                    continue

                colB = tableB[scmb[j]][~pd.isnull(tableB[scmb[j]])].values
                # エラーが発生するので修正
                # 元：colB = colB[~np.isnan(colB)]
                try:
                    colB = colB[~np.isnan(colB)] # Nanの値を除外
                except:
                    pass

                s1 = timeit.default_timer()
                sim_col = SchemaMapping.jaccard_similarity(acol_set[nameA], colB)
                e1 = timeit.default_timer()
                time1 += e1 - s1
                c1 += 1

                matching.append((nameA, nameB, sim_col))

        matching = sorted(matching, key=lambda d: d[2], reverse=True)

        for i in range(len(matching)):
            if matching[i][2] < self.sim_thres:
                break
            else:
                if matching[i][0] not in Mpair and matching[i][1] not in MpairR:
                    Mpair[matching[i][0]] = matching[i][1]
                    MpairR[matching[i][1]] = matching[i][0]
        if len(matching) == 0:
            rv = 0
        else:
            rv = matching[0][2]

        end_time = timeit.default_timer()
        print("raw schema mapping: ", end_time - start_time)
        print("sim schema mapping: ", time1)
        print("sim times: ", c1)
        return Mpair, rv

    # Do full schema mapping
    def mapping_naive_incremental(
            self, tableA, tableB, gid, meta_mapping, schema_linking, unmatched, mapped=None
    ):
        #疑問: Mpairに不備?(閾値より類似度が低い列の組み合わせも入っている?)
        #疑問: MpairR不要ではないか?
        """
        Do full schema mapping.
        tableAの列とtableBの列のジャカード類似度が閾値より高い，列の組み合わせを返す．
        類似度計算の際はNullまたはNan値は除いている．

        Args:
            tableA (DataFrame): テーブル.
            tableB (DataFrame): テーブル.
            gid (int): グループID.
            meta_mapping (dict{int: dict{str: str}}): 
                列Aと列Bのデータ集合のジャカード類似度のうち，
                閾値self.sim_thresより高い組み合わせの列名のリスト．
                内容は{グループID: {列名B: 列名A}}
            schema_linking (dict{int: dict{str: int}}):
                テーブルグループそれぞれに対し，列名とsid(列名ごとの固有ID.schema IDの略?)の対応関係が格納されている．
                {テーブルグループID: {列名: sid}}
            unmatched (dict{int: dict{str: dict{str: str}}}):
                閾値よりも類似度の値が小さい列の組み合わせ．
                内容は，{グループID: {列名: {列名: ""}}}
                列名nameAと列名nameBの列のジャカード類似度が閾値より低い場合は，unmatched[group][nameA][nameB] = ""とする．
            mapped dict{str: str}:
                すでに列間の類似度が閾値より高いとわかっている列名のペアのリスト(辞書)．
                内容は{列名A: 列名B}
                defaults to None.
        
        Returns:
            dict{str: str}:
                列Aと列Bのデータ集合のジャカード類似度のうち，
                閾値self.sim_thresより高い組み合わせの列名のリスト．{列名A: 列名B}.
            dict{int: dict{str: str}}:
                列Aと列Bのデータ集合のジャカード類似度のうち，
                閾値self.sim_thresより高い組み合わせの列名のリスト．
                内容は{グループID: {列名B: 列名A}}
            unmatched (dict{int: dict{str: dict{str: str}}}):
                閾値よりも類似度の値が小さい列の組み合わせ．
                内容は，{グループID: {列名: {列名: ""}}}
                列名nameAと列名nameBの列のジャカード類似度が閾値より低い場合は，unmatched[group][nameA][nameB] = ""とする．
            time_total (float): この関数の実行にかかった時間．(小数点以下がミリ秒.)

        self.var str:
            self.sim_thres (float): 列間のジャカード類似度の閾値．
        """

        start_time = timeit.default_timer() # 時間測定用
        time1 = 0 # 時間測定用(不使用)

        Mpair = mapped # dict{str: str}: 内容は{列名A: 列名B}
        MpairR = {} # dict{str: str}: 内容は{列名B: 列名A}
        for i in Mpair.keys():
            MpairR[Mpair[i]] = i

        matching = []
        t_mapping = {} # {SID: 列名}

        # 以下の2つのforループによって, meta_mappingに含まれている列名のうち, 
        # tableAおよびBに共通する列名を調べる.
        # 共通する列名はMpairおよびMpairRに格納.
        # MpairRはkeyとvalueを逆にしたもの.(必要性->?)
        for i in tableA.columns.tolist(): # tableAの列のうち, meta_mappingに含まれている列について, SIDと列名の組み合わせをt_mappingに追加
            if i in Mpair:
                continue
            if i not in meta_mapping[gid]:
                continue
            t_mapping[schema_linking[gid][meta_mapping[gid][i]]] = i

        for i in tableB.columns.tolist():
            if i in MpairR:
                continue
            if schema_linking[gid][i] in t_mapping: # tableAとtableBの共通列出会った場合(t_mappingに同じSIDが入っているとき)
                if tableB[i].dtype != tableA[t_mapping[schema_linking[gid][i]]].dtype:
                    continue
                Mpair[t_mapping[schema_linking[gid][i]]] = i
                MpairR[i] = t_mapping[schema_linking[gid][i]]

        scma = tableA.columns.tolist()
        scmb = tableB.columns.tolist()
        shmal = len(scma)
        shmbl = len(scmb)

        acol_set = {}

        # tableAの列とtableBの列の全ての組み合わせに対してジャカード類似度を計算.
        # 類似度が閾値より低ければunmatchedに追加.
        for i in range(shmal):

            nameA = scma[i]

            if nameA in Mpair or nameA == "Unnamed: 0" or "index" in nameA:
                continue

            if nameA not in acol_set:
                colA = tableA[scma[i]][~pd.isnull(tableA[scma[i]])].values
                acol_set[nameA] = list(set(colA))
            else:
                colA = acol_set[nameA]

            for j in range(shmbl):

                nameB = scmb[j]
                if nameB in MpairR or nameB == "Unnamed: 0" or "index" in nameB or \
                        tableA[nameA].dtype != tableB[nameB].dtype or nameB in unmatched[gid][nameA]:
                    continue

                colB = tableB[scmb[j]][~pd.isnull(tableB[scmb[j]])].values

                # エラーが発生するので修正
                # 元：colB = colB[~np.isnan(colB)]
                try:
                    colB = colB[~np.isnan(colB)] # Nanの値を除外
                except:
                    pass

                s1 = timeit.default_timer() # 時間測定用
                sim_col = self.jaccard_similarity(colA, colB)
                e1 = timeit.default_timer() # 時間測定用
                time1 += e1 - s1 # 時間測定用(不使用)

                if sim_col < self.sim_thres:
                    unmatched[gid][nameA][nameB] = ""

                matching.append((nameA, nameB, sim_col))

        matching = sorted(matching, key=lambda d: d[2], reverse=True) # 類似度が高い順にソート

        for i in range(len(matching)):
            if matching[i][2] < self.sim_thres:
                break
            else:
                if matching[i][0] not in Mpair and matching[i][1] not in MpairR:
                    Mpair[matching[i][0]] = matching[i][1]
                    MpairR[matching[i][1]] = matching[i][0]

        # meta_mappingに新しく発見した列の組み合わせを追加
        for i in tableA.columns.tolist():
            if i in Mpair:
                if i not in meta_mapping[gid]:
                    meta_mapping[gid][i] = Mpair[i]

                for j in tableB.columns.tolist():
                    if j != Mpair[i]:
                        unmatched[gid][i][j] = ""

        end_time = timeit.default_timer() # 時間測定用
        time_total = end_time - start_time # 時間測定用
        return Mpair, meta_mapping, unmatched, time_total

    # Do schema mapping for tables when looking for similar tables
    def mapping_naive_tables(
            self, tableA, valid_keys, schema_element, schema_dtype, tflag=False
    ):
        #疑問: MpairR不要ではないか?
        """
        Do schema mapping for tables when looking for similar tables.
        tableAの列名のうちvalid_keysに含まれる列に対して，
        schema_elementの列のとジャカード類似度が閾値より高い，列の組み合わせを返す．
        類似度計算の際はNullまたはNan値は除いている．

        Args:
            tableA (DataFrame): テーブル.
            valid_keys (list[str]): tableAの列のうち，schema mappingを調べる対象としたい列の列名リスト．
            schema_element (dict{int: dict{str: list[]}):
                テーブルグループごとの，各列名の列の実際のデータ値の重複無し集合．(null値を除く.)
                {テーブルグループID: {列名: [グループ内のすべてのテーブルから同じ列名のデータ値を集めた実際のデータ値]}}
            schema_dtype (dict{int: dict{str: str}}):
                テーブルの列に対するデータ型．{グループID: {列名: 列の型}}
            tflag (bool):
                Trueならrow schema mappingおよびsim schema mappingに要した時間を出力する．
                (この関数にかかった時間を出力.)
                defaults to False.
        
        Returns:
            dict{int: dict{str: str}}:
                列Aと列Bのデータ集合のジャカード類似度のうち，閾値self.sim_thresより高い組み合わせについて，
                類似度が高い順にソートした列名のリスト．{グループID: {列名A: 列名B}}.

        self.var str:
            self.sim_thres (float): 列間のジャカード類似度の閾値．
        """

        start_time = timeit.default_timer()
        time1 = 0

        # nameA (str): tableAの列名
        # nameB (str): schema_elementの列名
        Mpair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        MpairR = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名B: 列名A}}

        scma = tableA.columns.values # scma (list[str]): tableAの列名リスト
        shmal = len(scma) # 列数
        acol_set = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}

        for group in schema_element.keys(): # 各テーブルグループに対して

            Mpair[group] = {}
            MpairR[group] = {}
            matching = [] # list[str, str, float]: [列名A, schema_elementの列名B, 列Aと列Bの類似度]

            # tableAの各列とschema_elementの各列の組み合わせのジャカード類似度の計算
            # 計算結果はmatchingに格納
            for i in range(shmal):

                nameA = scma[i] # nameA (str): tableAの列名
                if nameA == "Unnamed: 0" or "index" in nameA or nameA not in valid_keys:
                    continue

                if nameA not in acol_set:
                    # colA (list[any]): 列名nameAの列のデータ値のうち,Null値を除いたデータ値のリスト.
                    colA = tableA[scma[i]][~pd.isnull(tableA[scma[i]])].values # ~はビット反転演算子
                    acol_set[nameA] = list(set(colA)) # 重複無しリスト
                else:
                    colA = acol_set[nameA]

                for j in schema_element[group].keys(): # schema_element各列名に対して

                    nameB = j
                    if nameB == "Unnamed: 0" or "index" in nameB:
                        continue

                    colB = np.array(schema_element[group][nameB]) # 列名がnameB列の実データ値のリスト

                    if schema_dtype[group][j] is not tableA[nameA].dtype:
                        continue

                    # エラーが発生するので修正．
                    # 元：colB = colB[~np.isnan(colB)] # Nanの値を除外
                    try:
                        colB = colB[~np.isnan(colB)] # Nanの値を除外
                    except:
                        pass
                    try:
                        colB = colB[~np.isnull(colB)] # Nanの値を除外
                    except:
                        pass

                    s1 = timeit.default_timer() # sim schema mappingの時間計測用.

                    sim_col = self.jaccard_similarity(colA, colB)
                    # ここまでmatching_naive_tables_joinも同じ
                    e1 = timeit.default_timer() # sim schema mappingの時間計測用.
                    time1 += e1 - s1 # sim schema mappingの時間計測用.

                    matching.append((nameA, nameB, sim_col)) # list[str, str, float]: [列名A, 列名B, 列Aと列Bの類似度]

            matching = sorted(matching, key=lambda d: d[2], reverse=True) # 類似度が高い順にソート

            for i in range(len(matching)):
                if matching[i][2] < self.sim_thres: # 類似度が閾値よりも低い場合は枝刈り.
                    break
                else:
                    if (
                            matching[i][0] not in Mpair[group]
                            and matching[i][1] not in MpairR[group]
                    ):
                        Mpair[group][matching[i][0]] = matching[i][1] # Mpair[グループID][nameA]=nameB
                        MpairR[group][matching[i][1]] = matching[i][0] # MpairR[グループID][nameB]=nameA

        end_time = timeit.default_timer()

        if tflag:
            print("Schema Mapping Before Search: %s Seconds." % (end_time - start_time))

        return Mpair

    # Do schema mapping for tables when looking for joinable tables
    def mapping_naive_tables_join(
            self,
            tableA,
            valid_keys,
            schema_element_sample,
            schema_element,
            schema_dtype,
            unmatched,
            tflag=False,
    ):
        #疑問: 最初のforループ要る？(tableAの各列とschema_elementの各列の組み合わせのジャカード類似度の計算)
        """
        Do schema mapping for tables when looking for joinable tables.
        tableAの列名のうちvalid_keysに含まれる列に対して，
        schema_elementの列との組み合わせ，およびschema_element_sampleとの列の組み合わせのうち，
        類似度が閾値より高い列の組み合わせを返す．
        類似度計算の際はNullまたはNan値は除いている．

        Args:
            tableA (DataFrame): テーブル.
            valid_keys (list[str]): tableAの列のうち，schema mappingを調べる対象としたい列の列名リスト．
            schema_element_sample (dict{int: dict{str: list[]}):
                schema_elementの各テーブルグループに対して指定のサイズ以下の列数，行数となるようにサンプリングしたもの．
            schema_element (dict{int: dict{str: list[]}):
                テーブルグループごとの，各列名の列の実際のデータ値の重複無し集合．(null値を除く.)
                内容は，{テーブルグループID: {列名: [グループ内のすべてのテーブルから同じ列名のデータ値を集めた実際のデータ値]}}
            schema_dtype (dict{int: dict{str: str}}):
                テーブルの列に対するデータ型．{テーブルグループID: {列名: 列の型}}
            unmatched (dict{int: dict{str: dict{str: str}}}):
                閾値よりも類似度の値が小さい列の組み合わせ．
                内容は，{グループID: {列名: {列名: ""}}}
                列名nameAと列名nameBの列のジャカード類似度が閾値より低い場合は，unmatched[group][nameA][nameB] = ""とする．
            tflag (bool):
                Trueならrow schema mappingおよびsim schema mappingに要した時間を出力する．
                (この関数にかかった時間を出力.)
                defaults to False.
        
        Returns:
            dict{int: dict{str: str}}:
                列Aと列Bのデータ集合のジャカード類似度のうち，閾値self.sim_thresより高い組み合わせについて，
                類似度が高い順にソートした列名のリスト．{グループID: {列名A: 列名B}}.
            unmatched (dict{int: dict{str: dict{str: str}}}):
                閾値よりも類似度の値が小さい列の組み合わせ．
                内容は，{グループID: {列名: {列名: ""}}}
                列名nameAと列名nameBの列のジャカード類似度が閾値より低い場合は，unmatched[group][nameA][nameB] = ""とする．

        self.var str:
            self.sim_thres (float): 列間のジャカード類似度の閾値．
        """

        start_time = timeit.default_timer() # row schema mappingの時間計測用.
        time1 = 0 # sim schema mappingの時間計測用.

        Mpair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        MpairR = {}

        scma = tableA.columns.values
        shmal = len(scma)
        acol_set = {}

        for group in schema_element.keys():

            Mpair[group] = {}
            MpairR[group] = {}
            matching = [] # list[str, str, float]: [列名A, schema_elementの列名B, 列Aと列Bの類似度]

            # tableAの各列とschema_elementの各列の組み合わせのジャカード類似度の計算
            # 計算結果はmatchingに格納
            for i in range(shmal):

                nameA = scma[i]

                if nameA == "Unnamed: 0" or "index" in nameA:
                    continue
                if nameA not in valid_keys:
                    continue

                if nameA not in acol_set:
                    A_index = ~pd.isnull(tableA[nameA])
                    colA = (tableA[nameA][A_index]).values
                    acol_set[nameA] = list(set(colA))
                else:
                    colA = acol_set[nameA]

                for j in schema_element[group].keys():

                    nameB = j

                    if nameB == "Unnamed: 0" or "index" in nameB:
                        continue

                    if schema_dtype[group][j] is not tableA[nameA].dtype:
                        continue

                    colB = np.array(schema_element[group][nameB])
                    # エラーが発生するので修正
                    # 元：colB = colB[~np.isnan(colB)]
                    try:
                        colB = colB[~np.isnan(colB)] # Nanの値を除外
                    except:
                        pass

                    s1 = timeit.default_timer() # sim schema mappingの時間計測用.
                    sim_col = self.jaccard_similarity(colA, colB)
                    # ここまでmatching_naive_tablesと同じ
                    if sim_col < self.sim_thres: # tableAの列Aとschema_elementの列Bの類似度が閾値より低い場合はunmatchedに格納.
                        unmatched[group][nameA][nameB] = ""

                    e1 = timeit.default_timer() # sim schema mappingの時間計測用.
                    time1 += e1 - s1

                    matching.append((nameA, nameB, sim_col))

            # tableAの各列とschema_element_sampleの各列の組み合わせのジャカード類似度の計算
            # 計算結果はmatchingに格納
            for i in schema_element_sample[group].keys():

                nameB = i # schema_element_sampleの列名

                if nameB == "Unnamed: 0" or "index" in nameB:
                    continue

                colB = np.array(schema_element_sample[group][nameB])
                # エラーが発生するので修正
                # 元：colB = colB[~np.isnan(colB)]
                try:
                    colB = colB[~np.isnan(colB)] # Nanの値を除外
                except:
                    pass

                for j in range(shmal):

                    nameA = scma[j]
                    if nameA == "Unnamed: 0" or "index" in nameA or nameB in unmatched[group][nameA]:
                        continue

                    if nameA not in acol_set:
                        colA = tableA[nameA][~pd.isnull(tableA[nameA])].values
                        acol_set[nameA] = list(set(colA))
                    else:
                        colA = acol_set[nameA]

                    if schema_dtype[group][nameB] is not tableA[nameA].dtype:
                        continue

                    s1 = timeit.default_timer() # sim schema mappingの時間計測用.
                    sim_col = self.jaccard_similarity(colA, colB)
                    e1 = timeit.default_timer() # sim schema mappingの時間計測用.
                    time1 += e1 - s1 # sim schema mappingの時間計測用.

                    if sim_col < self.sim_thres:
                        unmatched[group][nameA][nameB] = ""

                    matching.append((nameA, nameB, sim_col))

            matching = sorted(matching, key=lambda d: d[2], reverse=True) # 類似度が高い順にソート

            for i in range(len(matching)):
                if matching[i][2] < self.sim_thres: # 類似度が閾値よりも低い場合は枝刈り.
                    break
                else:
                    if (
                        matching[i][0] not in Mpair[group]
                        and matching[i][1] not in MpairR[group]
                    ):
                        Mpair[group][matching[i][0]] = matching[i][1]
                        MpairR[group][matching[i][1]] = matching[i][0]

        end_time = timeit.default_timer() # row schema mappingの時間計測用.

        if tflag:
            print("raw schema mapping: ", end_time - start_time)
            print("sim schema mapping: ", time1)

        return Mpair, unmatched

    # Do schema mapping on Groups
    def mapping_naive_groups(self, tableA, tableA_valid, schema_element):
        """
        Do schema mapping on Groups
        tableAの列名のうちtable_validに含まれるいずれかの列に対して，
        schema_elementの各グループのうち，閾値より高い類似度となる列が1つでも存在するグループを調べる．

        Args:
            tableA (DataFrame): テーブル.
            tableA_valid (list[str]): tableAの列のうち，schema mappingを調べる対象としたい列の列名リスト．
            schema_element (dict{int: dict{str: list[]}):
                テーブルグループごとの，各列名の列の実際のデータ値の重複無し集合．(null値を除く.)
                {テーブルグループID: {列名: [グループ内のすべてのテーブルから同じ列名のデータ値を集めた実際のデータ値]}}
        
        Returns:
            list[int]:
                tableAの列名のうちtable_validに含まれるいずれかの列に対して，
                schema_elementの各グループのうち，閾値より高い類似度となる列が1つでも存在するグループIDのリスト．

        self.var str:
            self.sim_thres (float): 列間のジャカード類似度の閾値．
        """

        time1 = 0

        Mpair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        MpairR = {}

        scma = tableA.columns.values
        shmal = len(scma)
        acol_set = {}

        group_list = []
        for group in schema_element.keys():
            Mpair[group] = {}
            MpairR[group] = {}
            matching = []

            for i in range(shmal):

                nameA = scma[i]
                if nameA not in tableA_valid or nameA == "Unnamed: 0" or "index" in nameA:
                    continue

                colA = tableA[scma[i]][~pd.isnull(tableA[scma[i]])].values
                if nameA not in acol_set:
                    acol_set[nameA] = list(set(colA))

                for j in schema_element[group].keys():

                    nameB = j
                    colB = np.array(schema_element[group][nameB])

                    # エラーが発生するので修正
                    # 元：colB = colB[~np.isnan(colB)]
                    try:
                        colB = colB[~np.isnan(colB)] # Nanの値を除外
                    except:
                        pass

                    s1 = timeit.default_timer()

                    sim_col = self.jaccard_similarity(acol_set[nameA], colB)
                    e1 = timeit.default_timer()
                    time1 += e1 - s1
                    matching.append((nameA, nameB, sim_col))

            matching = sorted(matching, key=lambda d: d[2], reverse=True)

            if len(matching) == 0:
                continue

            if matching[0][2] < self.sim_thres:
                continue
            else:
                group_list.append(group)

        return group_list
