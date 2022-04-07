[日本語版](/docs/README_Japanese.md)


# Similarity Search on Computational notebooks

![the interface](/retrieval_system/images/screenshot1.png "screenshot1")

This is a search system on Jupyter notebook.

Input a query through a browser interface and output Top-10 similar Jupyter notebook.

## Demo

![Demo](/retrieval_system/images/Demo_gif3_2.gif "demo")

If the demo is not displayed, see following files on google drive.
[demo 1](https://drive.google.com/file/d/1v2GsEWBz4TRddJAyp10hB_7RxMSBSasu/view?usp=sharing)
[demo 2](https://drive.google.com/file/d/1v2GsEWBz4TRddJAyp10hB_7RxMSBSasu/view?usp=sharing)

### Experimental evaluation

[Experimental evaluation](/retrieval_system/images/JupySim_experimental_evaluation.pdf "Experimental evaluation(pdf)")


## Components

* DBMS: PostgreSQL, Neo4j, SQLite

* Jupyter Notebook

* This search system

Jupyter notebooks converted into the particular formats must be stored in Databases.

## System preparation

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

│   ├── neo4j_sample.zip

│   ├── data1.zip

│   ├── data2.zip

│   └── data3.zip

└── README.md

## Dataset preparation

* Import [postgres_sample.sql](https://drive.google.com/file/d/1po-5Z5M4JbojbLjSvGkgMIOQK51_afur/view?usp=sharing) into postgres.

* Unzip `sample_dataset/neo4j_sample.zip` and move it into neo4j's data directory (e.g. `/usr/local/var/neo4j/data`).

* Make a directory `notebooks_data`, then unzip zipfiles in `Similarity_Search_on_Computational_Notebooks/sample_dataset/` and put '.ipynb' files into `notebooks_data/`.

## Start the interface

Navigate to `notebooks_data/` and start followings:

* PostgreSQL

* Neo4J (localhost:7474)

* Jupyter Notebook (localhost:8888)

Navigate to `Similarity_Search_on_Computational_Notebooks/retrieval_system/` and run following command to start our system.

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
