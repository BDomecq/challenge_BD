# src/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import load, dump

# libraries to evaluate performance
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

def some_util_function():
    """
    Ejemplo de una función auxiliar.
    """
    pass  # Implementa la función según sea necesario


def outliers_nan(data):
    # Reemplazar comas por puntos para interpretar los valores como números decimales
    data['duracion_recorrido'] = data['duracion_recorrido'].str.replace(',', '').astype(int)
    # Convertir la duración de segundos a minutos
    data['duracion_recorrido'] = data['duracion_recorrido'] / 60

    # Calcular los percentiles 1% y 99%
    p_1 = data['duracion_recorrido'].quantile(0.01)
    p_99 = data['duracion_recorrido'].quantile(0.99)
    # Aplicar Winsorization: reemplazar valores por los límites
    data['duracion_recorrido_w'] = np.clip(data['duracion_recorrido'], p_1, p_99)

    # Eliminar valores faltantes
    data.dropna(subset=['Género'], inplace=True) 

    return data

def features_date(data):
    # Convertir las columnas de fecha a tipo datetime
    data['fecha_origen_recorrido'] = pd.to_datetime(data['fecha_origen_recorrido'], errors='coerce')
    data['fecha_destino_recorrido'] = pd.to_datetime(data['fecha_destino_recorrido'], errors='coerce')

    # Separar en columnas de fecha y hora
    data['fecha_origen'] = data['fecha_origen_recorrido'].dt.date
    data['hora_origen'] = data['fecha_origen_recorrido'].dt.time

    # Asignar día de la semana (1=Lunes, 7=Domingo)
    data['dia_semana_origen'] = data['fecha_origen_recorrido'].dt.dayofweek + 1

    # Definir los días laborales (1=Lunes, ..., 5=Viernes)
    data['dia_laboral'] = data['dia_semana_origen'].apply(lambda x: 1 if x < 6 else 0)

    # Crear la columna Dia como integer
    data['dia'] = data['fecha_origen_recorrido'].dt.day
    # Crear la columna Mes como integer
    data['mes'] = data['fecha_origen_recorrido'].dt.month
    # Crear la columna Año como integer
    data['año'] = data['fecha_origen_recorrido'].dt.year

    # Lista de feriados en 2022
    feriados = pd.to_datetime([
        '2022-01-01', '2022-02-28', '2022-03-01', '2022-03-24', '2022-04-02',
        '2022-04-15', '2022-05-01', '2022-05-18', '2022-05-25', '2022-06-17',
        '2022-06-20', '2022-07-09', '2022-08-15', '2022-10-07', '2022-10-10', 
        '2022-11-20', '2022-11-21','2022-12-08', '2022-12-08', '2022-12-09',
        '2022-12-25'
    ])

    # Crear la columna Feriado
    data['feriado'] = data['fecha_origen_recorrido'].apply(lambda x: 1 if x.date() in feriados.date else 0)

    # Eliminar columnas que no se utilizarán
    data = data.drop(columns=['Unnamed: 0', 'X', 'fecha_destino_recorrido', 'fecha_origen_recorrido', 'nombre_estacion_origen', 'nombre_estacion_destino', 
                              'direccion_estacion_origen', 'direccion_estacion_destino', 'long_estacion_origen', 'lat_estacion_origen', 
                              'id_estacion_destino', 'long_estacion_destino', 'lat_estacion_destino', 'id_usuario', 'hora_origen'])

    # Convert 'Id_recorrido', 'id_estacion_origen' to integers
    for col in ['Id_recorrido', 'id_estacion_origen']:
        data[col] = data[col].str.replace('BAEcobici', '').astype(int)  

    return data

def demandaxcuadrante(data):
    # Calcular la demanda por cuadrante
    viajes_por_qo_dia = data.groupby(['QO', 'dia', 'mes', 'año'])['Id_recorrido'].count().reset_index()
    viajes_por_qo_dia.columns = ['QO', 'dia', 'mes', 'año', 'demanda']
    data = data.merge(viajes_por_qo_dia, on=['QO', 'dia', 'mes', 'año'], how='left')
    
    # Contar el número de estaciones por cuadrante
    estaciones_por_qo = data.groupby('QO')['id_estacion_origen'].nunique().reset_index()
    estaciones_por_qo.columns = ['QO', 'num_estaciones']
    proporciones_viajes = viajes_por_qo_dia.merge(estaciones_por_qo, on='QO')
    
    data = data.merge(proporciones_viajes[['QO', 'dia', 'mes', 'año']], on=['QO', 'dia', 'mes', 'año'], how='left')
    data = data.drop(columns=['Id_recorrido', 'id_estacion_origen', 'QD'])
    
    # One-Hot - Categorical Variables  
    data = pd.get_dummies(data, columns=['modelo_bicicleta', 'Género'])
    return data

def get_cuadrante(longitud, latitud):
    if -58.43 < longitud < -58.35 and -34.60 < latitud < -34.54:
        return 'NE'
    elif -58.55 < longitud < -58.43 and -34.60 < latitud < -34.54:
        return 'NO'
    elif -58.43 < longitud < -58.35 and -34.68 < latitud < -34.60:
        return 'SE'
    elif -58.55 < longitud < -58.43 and -34.68 < latitud < -34.60:
        return 'SO'
    else:
        return 'SO'

def cuadrantes(data):
    data['QO'] = data.apply(lambda row: get_cuadrante(row['long_estacion_origen'], row['lat_estacion_origen']), axis=1)
    data['QD'] = data.apply(lambda row: get_cuadrante(row['long_estacion_destino'], row['lat_estacion_destino']), axis=1)
    return data

def grouped_data(data):
    grouped = data.groupby(['QO', 'dia', 'mes', 'año']).agg({
        'duracion_recorrido_w': ['mean'],
        'demanda': 'first',
        'Género_FEMALE': 'sum',
        'Género_MALE': 'sum',
        'Género_OTHER': 'sum',
        'modelo_bicicleta_FIT': 'sum',
        'modelo_bicicleta_ICONIC': 'sum',
        'dia_laboral': 'first',
        'fecha_origen': 'first',
        'dia_semana_origen': 'first',
        'feriado': 'first'
    }).reset_index()
    grouped.columns = [
        'QO', 'dia', 'mes', 'año',
        'duracion_recorrido_w', 'demanda',
        'count_FEMALE', 'count_MALE', 'count_OTHER',
        'model_FIT', 'model_ICONIC',
        'dia_laboral', 'fecha_origen', 'dia_semana_origen', 'feriado'
    ]
    return grouped

def calculate_durations_by_gender(data, gender_column):
    return data[data[gender_column]].groupby(['QO', 'dia', 'mes', 'año'])['duracion_recorrido_w'].agg(['mean'])

def estadisticos(data):    
    duracion_media_FEMALE = calculate_durations_by_gender(data, 'Género_FEMALE').rename(columns={'mean': 'duracion_media_FEMALE'})
    duracion_media_MALE = calculate_durations_by_gender(data, 'Género_MALE').rename(columns={'mean': 'duracion_media_MALE'})
    duracion_media_OTHER = calculate_durations_by_gender(data, 'Género_OTHER').rename(columns={'mean': 'duracion_media_OTHER'})
    data_merg = grouped_data(data)
    data_merg = data_merg.merge(duracion_media_FEMALE, on=['QO', 'dia', 'mes', 'año'], how='left')
    data_merg = data_merg.merge(duracion_media_MALE, on=['QO', 'dia', 'mes', 'año'], how='left')
    data_merg = data_merg.merge(duracion_media_OTHER, on=['QO', 'dia', 'mes', 'año'], how='left')
    data_merg['prop_MALE_vs_FEMALE'] = data_merg.apply(lambda row: row['count_MALE'] / row['count_FEMALE'] if row['count_FEMALE'] > 0 else float('nan'), axis=1)
    data_merg['prop_MALE_vs_OTHER'] = data_merg.apply(lambda row: row['count_MALE'] / row['count_OTHER'] if row['count_OTHER'] > 0 else float('nan'), axis=1)
    data_merg['prop_FIT_vs_ICONIC'] = data_merg.apply(lambda row: row['model_FIT'] / row['model_ICONIC'] if row['model_ICONIC'] > 0 else float('nan'), axis=1)
    data_merg = data_merg.drop(columns=['count_FEMALE', 'count_MALE', 'count_OTHER', 'model_FIT', 'model_ICONIC'], axis=1)
    return data_merg


def normalization(data, technique):    
    if technique == 1:
        # here we use cube root
        data['demanda']=np.cbrt(data['demanda'])
    elif technique == 2:
        # here we use log10
        data['demanda']=np.log1p(data['demanda'])
    else:
        # here we use square root
        data['demanda']=np.sqrt(data['demanda'])
    
    return data

def scaled_data(X_train, X_test):
    # SCALADO DE DATOS
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def data_split(combined_data, dia = 25):
    #split data
    # Filtrar datos de entrenamiento y prueba por el día del mes
    train_mask = combined_data['dia'] <= dia
    test_mask = combined_data['dia'] > dia

    # Dividir en conjuntos de entrenamiento y prueba
    df_train = combined_data[train_mask]
    df_test = combined_data[test_mask]

    # Separar características (X) y variable objetivo (y) para entrenamiento y prueba
    X_train = df_train.drop(['demanda'], axis=1)
    y_train = df_train['demanda']
    X_test = df_test.drop(['demanda'], axis=1)
    y_test = df_test['demanda']

    print("Conjunto de entrenamiento:")
    print(X_train.shape)
    print("Conjunto de prueba:")
    print(X_test.shape)
    
    return X_train, y_train, X_test, y_test


def predict(ml_model,model_name, model_result, X_train, y_train, X_test, y_test, pre_train, cuadrante):
    if pre_train:
        #Cargo modelo pre-entrenado
        model = load('./models/best_model_{}.joblib'.format(cuadrante))
        
    else:      
        # model fitting
        model = ml_model.fit(X_train,y_train)

        # Guarda modelo 
        dump(model, './models/best_model_{}.joblib'.format(cuadrante))

    # predicting values
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Reverse the transformation on the predictions 
    y_train_pred_original = np.power(y_train_pred, 2)
    y_test_pred_original = np.power(y_test_pred, 2)

    # graph --> best fit line on test data
    sns.regplot(x=y_test_pred, y=y_test, line_kws={'color':'red'})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    '''Evaluation metrics on train data'''
    train_MSE  = round(mean_squared_error(y_train, y_train_pred),3)
    train_RMSE = round(np.sqrt(train_MSE),3)
    train_r2 = round(r2_score(y_train, y_train_pred),3)
    train_MAE = round(mean_absolute_error(y_train, y_train_pred),3)
    train_adj_r2 = round(1-(1-r2_score(y_train, y_train_pred))*((X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1)),3)
    print(f'train MSE : {train_MSE}')
    print(f'train RMSE : {train_RMSE}')
    print(f'train MAE : {train_MAE}')
    print(f'train R2 : {train_r2}')
    print(f'train Adj R2 : {train_adj_r2}')
    print('-'*150)

    '''Evaluation metrics on test data'''
    test_MSE  = round(mean_squared_error(y_test, y_test_pred),3)
    test_RMSE = round(np.sqrt(test_MSE),3)
    test_r2 = round(r2_score(y_test, y_test_pred),3)
    test_MAE = round(mean_absolute_error(y_test, y_test_pred),3)
    test_adj_r2 = round(1-(1-r2_score(y_test, y_test_pred))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)),3)
    print(f'test MSE : {test_MSE}')
    print(f'test RMSE : {test_RMSE}')
    print(f'test MAE : {test_MAE}')
    print(f'test R2 : {test_r2}')
    print(f'test Adj R2 : {test_adj_r2}')
    print('-'*150)

    # graph --> actual vs predicted on test data
    plt.figure(figsize=(6,5))
    plt.plot((y_test_pred_original))
    plt.plot(np.array((np.power(y_test, 2))))
    plt.legend(["Predicted","Actual"])
    plt.xlabel('Test Data {}'.format(cuadrante))
    plt.savefig('./results/plots/predictions_plot_{}.png'.format(cuadrante))
    plt.show()
    print('-'*150)

    '''actual vs predicted value on test data'''
    d = {'y_actual':np.power(y_test, 2), 'y_predict':y_test_pred_original, 'error':np.power(y_test, 2)-y_test_pred_original}
    print(pd.DataFrame(data=d).head().T)
    print('-'*150)
    # Guardar predicciones en CSV
    predictions = pd.DataFrame(y_test_pred_original)
    predictions.to_csv('./results/predictions_{}.csv'.format(cuadrante), index=False)

    # using the score from the performance metrics to create the final model_result.
    model_result.append({'model':model_name,
                        'train MSE':train_MSE,
                        'test MSE':test_MSE,
                        'train RMSE':train_RMSE,
                        'test RMSE':test_RMSE,
                        'train MAE':train_MAE,
                        'test MAE':test_MAE,
                        'train R2':train_r2,
                        'test R2':test_r2,
                        'train Adj R2':train_adj_r2,
                        'test Adj R2':test_adj_r2})
