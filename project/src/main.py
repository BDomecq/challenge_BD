# src/main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils import normalization, scaled_data, data_split, predict
from joblib import dump, load

from data_preprocessing import BikeDataPreprocessor
from data_merger import DataMerger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def load_data():
    """
    Carga los datasets de bicicletas y clima.
    
    Returns:
    pd.DataFrame, pd.DataFrame: Los datasets de bicicletas y clima.
    """
    bike_data = pd.read_csv('./src/data/trips_2022.csv')
    climate_data = pd.read_csv('./src/data/clima_bsas_2022.csv')
    return bike_data, climate_data

def preprocess_and_merge_data(bike_data, climate_data):
    """
    Preprocesa y fusiona los datasets de bicicletas y clima.
    
    Parameters:
    bike_data (pd.DataFrame): El dataset de bicicletas.
    climate_data (pd.DataFrame): El dataset climático.
    
    Returns:
    pd.DataFrame: Datos combinados y preprocesados.
    """
    # Inicializar los transformadores
    preprocessor = BikeDataPreprocessor()
    
    # Preprocesar datos de bicicletas
    bike_data_clean = preprocessor.fit_transform(bike_data)  
    merger = DataMerger(merge_on=['dia', 'mes', 'año'])
    # Fusionar datos
    combined_data = merger.transform([bike_data_clean, climate_data])

    
    return combined_data

def train_and_evaluate_model(combined_data, cuadrante):
    """
    Entrena el pipeline de modelado y evalúa su rendimiento.
    
    Parameters:
    combined_data (pd.DataFrame): Datos combinados y preprocesados.
    
    Returns:
    None
    """
    df_SE = combined_data[combined_data['QO'] == cuadrante].copy()
    df_SE = df_SE.drop(columns=['QO', 'fecha_origen'])
    # sqr sobre demanda
    df_SE = normalization(df_SE, 3)
  
    X_train, y_train, X_test, y_test = data_split(df_SE, 25)
    # Escalar data
    X_train, X_test, scaler = scaled_data(X_train, X_test)
    

    #SELECCION DE MODELO SI NO TENEMOS ENTRENADO SE ENTRENA

    param_grid = {'n_estimators': [50,80, 120], # number of trees in the ensemble
             'max_depth': [15,20,30],           # maximum number of levels allowed in each tree.
             'min_samples_split': [5,15,20],    # minimum number of samples necessary in a node to cause node splitting.
             'min_samples_leaf': [3,5,8]}       # minimum number of samples which can be stored in a tree leaf.
    
    # Initialize the RandomForestRegressor model
    rf = RandomForestRegressor()

    # Use GridSearchCV to perform a grid search over the parameter grid
    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=8, scoring='neg_root_mean_squared_error', n_jobs=-1)

    pre_train = True #Cambiar para usar modelo pre-entrenado

    if pre_train == False:
        # Fit the model to the training data
        grid_search.fit(X_train, y_train)        

        # Get the best parameters from the grid search
        rf_optimal_model = grid_search.best_estimator_
        # Imprimir los mejores parámetros encontrados
        print("Mejores parámetros encontrados:")
        print(grid_search.best_params_)
    else:
        rf_optimal_model = 0

    model_result = []
        
    predict(rf_optimal_model, 'Random Forest', model_result, X_train, y_train, X_test, y_test, pre_train, cuadrante)

   


   
def main():
    """
    Función principal que ejecuta el flujo de trabajo.
    
    Returns:
    None
    """
    # Cargar datos
    bike_data, climate_data = load_data()
        
    # Preprocesar y combinar datos
    combined_data = preprocess_and_merge_data(bike_data, climate_data)
    
    #Cuatro cuadrantes a analizar
    QO = ['NO', 'NE', 'SE', 'SO']
    
    for cuadrante in QO:
        print('Cuadrante: ', cuadrante)
        # Entrenar y evaluar el pipeline
        train_and_evaluate_model(combined_data, cuadrante)

if __name__ == "__main__":
    main()



