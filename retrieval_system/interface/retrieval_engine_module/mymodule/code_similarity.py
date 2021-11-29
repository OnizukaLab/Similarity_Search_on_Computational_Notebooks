import numpy as np

class CodeSimilarity:
    def __init__(self):
        pass

    @staticmethod
    def jaccard_similarity_coefficient(colA:list, colB:list) -> float:
        """
        The Jaccard similarity between two sets A and B is defined as
        |intersection(A, B)| / |union(A, B)|.
        集合Aと集合Bのジャカード類似度を計算する．

        Args:
            colA list[any]: データ値の集合.
            colB list[any]: データ値の集合.

        Returns:
            float: ジャカード類似度．
        """
        try:
            if min(len(colA), len(colB)) == 0:
                return 0
            colA = np.array(colA) #numpyの型に変換?
            # 疑問: colBは変換しなくて良いのか？
            colB = np.array(colB) #numpyの型に変換?
            union = len(np.union1d(colA, colB))
            inter = len(np.intersect1d(colA, colB))
            return float(inter) / float(union)
        except:
            return 0

    @staticmethod
    def code_sim_based_on_levenshtein_dist(seq1:str, seq2:str) -> float:
        """
        レーベンシュタイン距離（編集距離）に基づく類似度（0以上）を0以上1以下に正規化して返す．

        Calculate strings similarity using Levenshtein distance between seq1 and seq2.
        The similarity is normalized in range [0,1].
        """
        return CodeSimilarity.smith_waterman(seq1, seq2)/min(len(seq1), len(seq2))
        
    @staticmethod
    def smith_waterman(seq1:str, seq2:str) -> int:
        """
        レーベンシュタイン距離（編集距離）に基づく類似度（0以上）を計算する．

        Calculate strings similarity using Levenshtein distance between seq1 and seq2.
        """
        m=len(seq1)
        n=len(seq2)

        #weights
        h=1 # hit
        d=1 # insertion or deletion (indel)
        r=1 # replacing
        gaps=0 # affine gap modelのとき

        scoring_matrix = np.zeros(shape=(m+1,n+1), dtype=np.int)
        parent_matrix = np.zeros(shape=(m+1,n+1), dtype=np.int)
        for i in range(1,m+1):
            for j in range(1,n+1):
                if seq1[i-1]==seq2[j-1]:
                    scoring_matrix[i][j] = scoring_matrix[i-1][j-1] + h
                    parent_matrix[i][j]=(i-1)+m*(j-1)
                else:
                    scoring_matrix[i][j] = max(0, max(scoring_matrix[i-1][j]-d, max(scoring_matrix[i][j-1]-d, scoring_matrix[i-1][j-1]-r)))
        #print(scoring_matrix)
        #print(scoring_matrix[m][n])
        return scoring_matrix[m][n]

    if __name__ == "__main__":
        seq1="abcbadbcab"
        seq2="abbdbda"
        levenshtein_dist = smith_waterman(seq1, seq2)
        print(levenshtein_dist/min(len(seq1), len(seq2)))
