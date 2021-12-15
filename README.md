[日本語版](/docs/README_Japanese.md)


# Similarity Search on Computational notebooks

![the interface](/retrieval_system/images/screenshot1.png "screenshot1")


## Demo

![Demo](/retrieval_system/images/Demo_gif3_2.gif "demo")

[Demo1 old version](https://drive.google.com/file/d/1x1yiM8xQkwlJtQmQPgIOiSyN2d3QoUBu/view?usp=sharing)

[Demo2 old version](https://drive.google.com/file/d/19CfahRTEwlbaOSZQLLfiALocrVQ3SNkH/view?usp=sharing)


### Experimental evaluation

![Experimental evaluation](/retrieval_system/images/JupySim_experimental_evaluation.pdf "Experimental evaluation(pdf)")


## システムの構成要素

* DBMS: PostgreSQL, Neo4j, SQLite

* Jupyter Notebook

* This search system

Jupyter notebooks converted into the particular formats must be stored in Databases.

## Preparation

Clone this system by running following command:

```
git clone https://github.com/OnizukaLab/Similarity_Search_on_Computational_Notebooks.git
```

A file tree consisting of important files is as follows:

Similarity_Search_on_Computational_Notebooks/

├── retrieval_system/

│   ├── manage.py

│   ├── interface/

│   └── retrieval_system/

└── sample_dataset/

│   ├── postgres_sample.sql (in preparation)

│   ├── neo4j_sample (in preparation)

│   ├── data1.zip

│   ├── data2.zip

│   └── data3.zip

└── README.md

Import sample_dataset/postgres_sample.sql into PostgreSQL and sample_dataset/neo4j into neo4j, respectively.

Make a directory notebooks_data, then unzip sample_dataset/ and put '.ipynb' files into notebooks_data.

## 検索Webアプリケーションの起動

Move 'notebooks_data' and start followings:

* PostgreSQL

* Neo4J (localhost:7474)

* Jupyter Notebook (localhost:8888)

Move 'Similarity_Search_on_Computational_Notebooks/retrieval_system/' and run following command to start our system.

```
python manage.py runserver <port>
```

Then use interface by accessing
http://127.0.0.1:<port>/interface/ .

For example, if you want use port 8080, run following command:

```
python manage.py runserver 8080
```

and access
http://127.0.0.1:8080/interface/ .
