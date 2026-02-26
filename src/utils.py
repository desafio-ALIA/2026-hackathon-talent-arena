import json
import pandas as pd
from pathlib import Path

def load_data(file_path):
    """
    Carga datos estructurados desde un archivo JSON.
    
    Esencial para leer los datasets iniciales de entrada proporcionados en el hackathon.
    
    Args:
        file_path (str o Path): La ruta de sistema al archivo JSON.
        
    Returns:
        pd.DataFrame: Un DataFrame listo para procesarse mediante Pandas.
    """
    try:
        return pd.read_json(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_data(data, file_path):
    """
    Guarda los datos procesados durante el hackathon a un archivo JSON en disco.
    
    Es de gran utilidad para guardar predicciones, resultados generados y poder validarlos luego.
    
    Args:
        data (lista de diccionarios o pd.DataFrame): Dato/s a guardar (tus resultados o transformaciones).
        file_path (str o Path): La ruta junto al nombre del archivo de salida.
    """
    try:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
