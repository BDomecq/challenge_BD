# src/data_merger.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataMerger(BaseEstimator, TransformerMixin):
    def __init__(self, merge_on=['dia', 'mes', 'año']):
        """
        Inicializa el merger.
        
        Parameters:
        merge_on (str): Nombre de la columna en la que se debe fusionar los datasets.
        """
        self.merge_on = merge_on

    def fit(self, X, y=None):
        """
        Ajuste del merger (no se utiliza, solo para cumplir con la interfaz de TransformerMixin).
        
        Parameters:
        X (list): Lista de dos DataFrames para ajustar.
        y (pd.Series, optional): Etiquetas objetivo (no se utilizan aquí).
        
        Returns:
        self: El combinador ajustado.
        """
        return self

    def transform(self, X):
        """
        Fusiona los datasets en los features seleccionados.
        
        Parameters:
        X (list): Lista de dos DataFrames (datos de bicicletas y datos de clima).
        
        Returns:
        pd.DataFrame: Datos combinados.
        """
        bike_data, climate_data = X
        merged_data = pd.merge(bike_data, climate_data, how='left', on=self.merge_on)
        merged_data = merged_data.sort_values(by=['año', 'mes', 'dia'])
        
        return merged_data