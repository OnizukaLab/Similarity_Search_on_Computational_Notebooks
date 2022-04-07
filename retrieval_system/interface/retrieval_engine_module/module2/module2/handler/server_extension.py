from notebook.utils import url_path_join
from module2.config import config
from module2.handler.handler import JuneauHandler
from module2.search.search_withprov_opt import WithProv_Optimized


def load_jupyter_server_extension(nb_server_app):
    """
    Registers the `JuneauHandler` with the notebook server.
    ノートブックサーバに`JuneauHandler`を登録する．(../juneau/__init__.py内部で呼び出される．)
    IPythonHandlerについて詳細は右の通り: "Registering custom handlers" https://jupyter-notebook.readthedocs.io/en/stable/extending/handlers.html

    Args:
        nb_server_app (NotebookWebApplication): handle to the Notebook webserver instance.
    """
    nb_server_app.log.info("Juneau extension loading...")

    # Inject global application variables.
    web_app = nb_server_app.web_app
    web_app.indexed = set()
    web_app.nb_cell_id_node = {}
    web_app.search_test_class = WithProv_Optimized(config.sql.dbname, config.sql.dbs)

    route_pattern = url_path_join(web_app.settings["base_url"], "/juneau")
    web_app.add_handlers(".*$", [(route_pattern, JuneauHandler)])
    nb_server_app.log.info("Juneau extension loaded.")
