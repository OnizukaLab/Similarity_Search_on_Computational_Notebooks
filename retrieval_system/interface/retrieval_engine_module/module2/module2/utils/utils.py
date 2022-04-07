from sys import getsizeof


def clean_notebook_name(nb_name):
    """
    Cleans the notebook name by removing the .ipynb extension, removing hyphens,
    and removing underscores.
    Example:
        >>> nb = "My-Awesome-Notebook.ipynb"
        >>> # Receive a PUT with `nb`
        >>> print(clean_notebook_name(nb))
        >>> # prints "MyAwesomeNotebook"
    Returns:
        A string that is cleaned per the requirements above.
    """
    nb_name = nb_name.replace(".ipynb", "").replace("-", "").replace("_", "")
    nb_name = nb_name.split("/")
    if len(nb_name) > 2:
        nb_name = nb_name[-2:]
    nb_name = "".join(nb_name)
    return nb_name[-25:]


def _getsizeof(x):
    """
    Gets the size of a variable `x`. Amended version of sys.getsizeof
    which also supports ndarray, Series and DataFrame.
    """
    if type(x).__name__ in ["ndarray", "Series"]:
        return x.nbytes
    elif type(x).__name__ == "DataFrame":
        return x.memory_usage().sum()
    else:
        return getsizeof(x)
