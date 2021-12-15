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
Performs provenance search in tables by using the star distance
between rows.
"""


class ProvenanceSearch:
    ##
    # 関数
    # star_distance(s1,s2) : スター型グラフs1, s2のスター編集距離sdt(s1,s2)を計算する.
    # star_mapping_distance : 2つのグラフの近似編集距離ζ(query,root)を計算する.
    # graph_edit_distance : グラフqueryとグラフgraphs_dependenciesの編集距離が小さい順に上位k個をリストとして返す．
    # search_top_k : クエリに対する上位k件の結果を返す関数を呼び出し，戻り値をそのまま返す．
    # search_score_rank : 編集距離で昇順にソートした，クエリとgraphs_dependenciesの近似編集距離のリストを返す．
    ##
    """
    データの周辺のコードをグラフ(Provenance Graphs)として入力し，スター編集距離を利用してコードの類似性を計算する．

    Args:
        graphs (dict{dict{}}): セルのコードの変数間の関係．内容は，{行番号: (変数1, [(変数2: 変数2を入力として変数1に対する操作)])}

    self.var str:
        self.graphs_dependencies (dict{any: dict{int: list[tuple(list[str], list[tuple(str, str)])]}}): セルのコードの変数間の関係．内容は，{グラフごとのID?: {行番号: (変数1, [(変数2: 変数2を入力として変数1に対する操作)])}}
    """

    def __init__(self, graphs):
        self.graphs_dependencies = graphs

    @staticmethod
    def star_distance(listA, listB):
        """
        2つのスター型グラフlistAとlistBのスター編集距離を求める.
        大きいグラフの近似編集距離を計算するのに利用される．

        Args:
            listA (list[str] or list[tuple(str, str)]):
                "変数2を入力として変数1に対する操作"のリスト．
                または，tuple(変数2, 変数2を入力として変数1に対する操作)のリスト．(おそらく後者)
                これは，スター型グラフs1=(r1,L1,f1)に対するf1(辺のラベル集合)に相当．
            listB (list[str] or list[tuple(str, str)]): listAと同様に，スター型グラフs2=(r2,L2,f2)に対するf2(辺のラベル集合)に相当．
        
        Returns:
            int: listAとlistBで与えられるグラフs1およびs2のスター編集距離sdt(s1,s2).
        """

        # listA, listBをソート
        listA, listB = sorted(listA), sorted(listB)
        lenA, lenB = len(listA), len(listB)
        i, j = 0, 0
        # intersection list: listAとlistBの要素のうち, 共通要素．
        intersection = []
        # listAとlistBに共通するコンテンツのみを取り出す
        while i < lenA and j < lenB:
            if listA[i] == listB[j]:
                intersection.append(listA[i])
                i += 1
                j += 1
            elif listA[i] < listB[j]:
                i += 1
            elif listA[i] > listB[j]:
                j += 1
        # 変数 dist : sdt(s1,s2)の計算
        # sdt(s1,s2)= ||f1|-|f2|| - M(f1,f2)
        # M(f1,f2)= max{ |(f1の多重集合)| , |(f2の多重集合)| } - |(f1の多重集合とf2の多重集合の積集合)|
        dist = abs(lenA - lenB) + max(lenA, lenB) - len(intersection)
        return dist

    @staticmethod
    def star_mapping_distance(query, root):
        """
        2つのグラフの近似編集距離ζ(query,root)を計算する．
        それぞれのグラフの部分グラフの組み合わせに対しスター編集距離を利用している．

        Args:
            query (list[tuple(list[str], list[tuple(str, str)])]):
                tuple([変数1],[(変数2: 変数2を入力として変数1に対する操作)])のリスト．
            root (list[tuple(list[str], list[tuple(str, str)])] or dict{str: dict{str: str}}?):
                tuple([変数1],[(変数2: 変数2を入力として変数1に対する操作)])のリスト．
        
        Returns: 
            int-like: 2つのグラフの近似編集距離ζ(query,root)を返す．
        """
        mapo, mapr = {}, {}
        distance_pair = [] # 2つのグラフのノード名とそのスター型グラフの類似度のトリプルを格納
        # siとγ(si)のペアを探すために全組み合わせを総当たりする.
        for i in query.keys(): # i (list[str]?): ノード名.
            stara = query[i].values() # スター型グラフの辺のリストA.
            for j in root.keys(): # j (str): 変数2の名前.
                starb = root[j].values() # スター型グラフの辺のリストB. rootがdict{str1: dict{str2: str3}}のとき, root[j]はdict[str2: str3], root[j].values()はstr3
                # staraとstarbのスター編集距離を求める.
                simAB = ProvenanceSearch.star_distance(stara, starb)
                distance_pair.append((i, j, simAB))

        distance_pair = sorted(distance_pair, key=lambda d: d[2]) # simABで昇順にソート 
        distance_ret = 0
        for i, j, k in distance_pair:
            if i not in mapo and j not in mapr:
                mapo[i] = j
                mapr[j] = i
                distance_ret += k # min(siとγ(si))の和を計算

        return distance_ret

    def graph_edit_distance(self, query, k):
        """
        グラフqueryとグラフself.graphs_dependenciesの編集距離が小さい順に上位k個をリストとして返す．
        k個存在しなければ存在する最大個数返す．

        Args:
            query dict{str: dict{str: str}}?
            k (int): 上位何件の検索結果を返すかを指定する．

        Returns:
            list: 上位k個のgraphs_dependenciesのキー(ノード番号？)のリスト．k個に満たない場合は，最大個数分だけのキーのリスト．
        """
        distance_rank = []
        for i in self.graphs_dependencies.keys(): # 各ノードに対して(おそらく)
            for j in self.graphs_dependencies[i].keys(): #ノードの隣接ノード対して以下の処理
                # クエリグラフと各スター型グラフに対してスター編集距離を計算.
                # self.graph_dependencies[i][j]: tuple(list[str], list[tuple(str, str)])
                dist = ProvenanceSearch.star_mapping_distance(query, self.graphs_dependencies[i][j])
                distance_rank.append((j, dist))
        distance_rank = sorted(distance_rank, key=lambda d: d[1]) # スター編集距離で昇順にソート
        k = min(k, len(distance_rank)) # k個存在しなければ全体を返す
        return [i for i, _ in distance_rank[:k]]

    def search_top_k(self, query, k):
        """
        クエリに対する上位k件の結果を返す関数を呼び出し，戻り値をそのまま返す．
        グラフqueryとグラフgraphs_dependenciesの編集距離が小さい順に上位k個をリストとして返す．

        Args:
            query : クエリグラフ．
            k (int): 上位何件の検索結果を返すかを指定する．

        Returns:
            list: 上位k個のgraphs_dependenciesのキー(ノード番号？)のリスト．k個に満たない場合は，最大個数分だけのキーのリスト．
        """
        return self.graph_edit_distance(query, k)

    def search_score_rank(self, query):
        """
        編集距離で昇順にソートした，クエリとgraphs_dependenciesの近似編集距離のリストを返す．

        Args:
            query (dict{dict{}}):
                クエリグラフ．キーにノード番号，値にそのノードを根としたスター型グラフのリストをとる辞書．

        Returns:
            list:
                編集距離で昇順にソートした，クエリとgraphs_dependenciesの近似編集距離のリスト．
        """
        distance_rank = []
        for i in self.graphs_dependencies.keys():
            dist = ProvenanceSearch.star_mapping_distance(query, self.graphs_dependencies[i])
            distance_rank.append((i, dist))
        return sorted(distance_rank, key=lambda d: d[1])
