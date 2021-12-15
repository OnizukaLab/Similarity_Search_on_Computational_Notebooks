[English](/README.md)

# 計算ノートブックの類似検索

## 概要

![スクリーンショット](/retrieval_system/images/screenshot1.png "Screenshot")

Jupyter notebookのTop-10検索を行う検索システムです．

ブラウザインタフェースでクエリを入力し，クエリに類似したJupyter notebookを10件出力します．

### デモ

![デモ動画](/retrieval_system/images/Demo_gif3_2.gif "Demo(gif)")

[Demo1(old version)](https://drive.google.com/file/d/1x1yiM8xQkwlJtQmQPgIOiSyN2d3QoUBu/view?usp=sharing)

[Demo2(old version)](https://drive.google.com/file/d/19CfahRTEwlbaOSZQLLfiALocrVQ3SNkH/view?usp=sharing)

### 評価実験

[評価実験](/retrieval_system/images/JupySim_experimental_evaluation.pdf "Experimental evaluation(pdf)")


## システムの構成要素

* DBMS: PostgreSQL, Neo4j, SQLite

* Jupyter Notebook

* 当検索システム

検索前に，あらかじめ加工されたJupyter notebookが保存されている必要があります．

## システムの準備

以下を実行し，本システムをクローンします．

```
git clone https://github.com/OnizukaLab/Similarity_Search_on_Computational_Notebooks.git
```

主要なファイルで構成したファイルツリーを以下に示します．

Similarity_Search_on_Computational_Notebooks/

├── retrieval_system/

│   ├── manage.py

│   ├── interface/

│   └── retrieval_system/

└── sample_dataset/

│   ├── neo4j_sample.zip

│   ├── data1.zip

│   ├── data2.zip

│   └── data3.zip

└── README.md

## データセットの準備

* [postgres_sample.sql](https://drive.google.com/file/d/1po-5Z5M4JbojbLjSvGkgMIOQK51_afur/view?usp=sharing)をpostgresにインポートする．

* `sample_dataset/neo4j_sample.zip`を解凍したものをneo4jのデータディレクトリ(e.g. `/usr/local/var/neo4j/data`)に移す．

* `notebooks_data`というディレクトリを場所不問で作成し，そこに`Similarity_Search_on_Computational_Notebooks/sample_dataset/`以下のzipファイルを解凍したものを入れる．

## 検索Webアプリケーションの起動

ディレクトリ`notebooks_data`に移動し，以下を指定のポートでそれぞれを起動します．

* PostgreSQL

* Neo4J (localhost:7474)

* Jupyter Notebook (localhost:8888)

ディレクトリ`Similarity_Search_on_Computational_Notebooks/retrieval_system/`に移動し，以下のコマンドを実行してサーバを起動します．

```
python manage.py runserver 8080
```

http://127.0.0.1:8080/interface/
  
で，インタフェースにアクセスできます．ポート番号は変更可能です．
