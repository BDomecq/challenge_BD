# src/data_preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime
from utils import cuadrantes, features_date, demandaxcuadrante, outliers_nan, estadisticos
import datetime as dt

from sklearn.base import BaseEstimator, TransformerMixin

class BikeDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Inicializa el preprocesador de datos de bicicletas.
        
        Parameters:        
        """
        self.data = None
       

    def fit(self, X, y=None):
        """
        Ajusta el preprocesador a los datos (aquí no hace nada, pero se requiere para la interfaz de TransformerMixin).
        
        Parameters:
        X (pd.DataFrame): Datos de entrada para ajustar el preprocesador.
        y (pd.Series, optional): Etiquetas objetivo (no se utilizan aquí).
        
        Returns:
        self: El preprocesador ajustado.
        """
        # Primero, almacenar una copia de los datos y aplicar cuadrantes
        self.data = X.copy()
        #self.data = outliers_nan(self.data)
        
        return self


    def transform(self, data):
        """
        Transforma los datos de entrada. Manejo de outliers y Nan's
        
        Parameters:
        X (pd.DataFrame): Datos de entrada para transformar.
        
        Returns:
        pd.DataFrame: Datos transformados.
        """
        self.data = outliers_nan(self.data)
        
        self.data = cuadrantes(self.data)
        
        # Luego, aplicar el procesamiento de fechas
        self.data = features_date(self.data)

        # Finalmente, calcular la demanda por cuadrante y preparar el DataFrame final
        self.data = demandaxcuadrante(self.data)
       
        # Calcular algunos features
        self.data = estadisticos(self.data)

        return self.data   
  