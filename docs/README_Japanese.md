# 計算ノートブックの類似検索

![the interface](retrieval_system/images/screenshot1.png "screenshot1")


## デモ

[Demo1](https://drive.google.com/file/d/1x1yiM8xQkwlJtQmQPgIOiSyN2d3QoUBu/view?usp=sharing)

[Demo2](https://drive.google.com/file/d/19CfahRTEwlbaOSZQLLfiALocrVQ3SNkH/view?usp=sharing)



## システムの構成要素

* DBMS: PostgreSQL, Neo4j, SQLite
* Jupyter Notebook
* *[Juneau](https://github.com/juneau-project/juneau.git)

Juneauは計算ノートブックのグラフ化や類似度計算に利用しています．

## 準備

以下を実行し，本システムをクローンします．

```
git clone https://github.com/OnizukaLab/Similarity_Search_on_Computational_Notebooks.git
```

以下を実行し，ディレクトリSimilarity_Search_on_Computational_Notebooksと同じ階層にJuneauをクローンします．

```
git clone https://github.com/juneau-project/juneau.git
```

主要なファイルで構成したファイルツリーを以下に示します．

.

├── Similarity_Search_on_Computational_Notebooks

│   ├── README.md

│   ├── data.zip

│   ├── notebooks_data.zip

│   └── retrieval_system

│       ├── manage.py

│       ├──interface

│       └──retrieval_system

└── juneau


data.zipはNeo4Jのデータ，notebooks_data.zipは.ipynbフォーマットのファイルが入っています．




## Start the web interface

Start PostgreSQL and Neo4J with the databases that have transformed computational notebooks.

Then start Jupyter Notebook on port 8888 by running the following command:

```
jupyter notebook --port 8888
```


To start the web interface, go to the "similarity_retrieval_system/retrieval_system" directory and run the following command:

```
python manage.py runserver
```

You can access the `localhost:8000` and you can use the interface.

If you want to change the server's port, this command starts the server on port 8080:
```
python manage.py runserver 8080
```
