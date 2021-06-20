# Similarity Search on Computational notebooks

## The components of our system

Our system uses PostgreSQL and Neo4j as DBMS. Our system also uses [Juneau](https://github.com/juneau-project/juneau.git) to transform computational notebooks into graphs and compute some similarities. 

Before searching, transformed computational notebooks must be stored into them.
The sample data is now in preparation.

## Start the web interface

First, make a "similarity_retrieval_system" directory and clone our system into the directory.

This search system uses [Juneau](https://github.com/juneau-project/juneau.git).

Then clone Juneau into the same hierarchy to "similarity_retrieval_system" as following directory structure:

.

├── similarity_retrieval_system
│   └── retrieval_system
│       ├── README.md
│       ├── manage.py
│       ├──db.sqlite3
│       ├──interface
│       ├──retrieval_system
│       └──templates
└── juneau


To start the web interface, go to the "similarity_retrieval_system/retrieval_system" directory and run the following command:
```
python manage.py runserver
```

If you want to change the server's port, this command starts the server on port 8080:
```
python manage.py runserver 8080
```
