# src/__init__.py

"""
Paquete para la gestión de datos y modelado.

Este paquete contiene clases y funciones para:
- Preprocesar datos de bicicletas.
- Combinar datos de bicicletas y clima.
- Crear y entrenar modelo de regresión RF.
- Devuelve la prediccion de la demanda.

Módulos disponibles:
- data_preprocessing: Contiene la clase BikeDataPreprocessor para limpiar y transformar datos de bicicletas de Bs As.
- data_merger: Contiene la clase DataMerger para fusionar datasets.
- utils: Contiene funciones auxiliares.
"""

from .data_preprocessing import BikeDataPreprocessor
from .data_merger import DataMerger
from .utils import * 

__all__ = [
    'BikeDataPreprocessor',
    'DataMerger',
    'create_pipeline',
]

# Version del paquete
__version__ = '1.0.0'