import os
from .arff_reader import read_arff

import pandas as pd


__all__ = ['load', 'read_csv', 'read_clipboard', 'read_arff']

read_csv = pd.read_csv
read_clipboard = pd.read_clipboard

_SETS_DIR = os.path.join(os.path.dirname(__file__), 'sets')

def load(set_name, *args, **kwargs):
    """
    Carrega dataset. Se `set_name` tiver extensão (.arff, .csv, .txt) lê o arquivo;
    senão, busca o dataset padrão em sets/.
    """
    _, ext = os.path.splitext(set_name)

    if ext == '.arff':
        return read_arff(set_name)
    elif ext in ['.csv', '.txt']:
        return read_csv(set_name, *args, **kwargs)
    else:
        # dataset padrão (ex: 'iris')
        path = os.path.join(_SETS_DIR, set_name + '.arff')
        return read_arff(path)