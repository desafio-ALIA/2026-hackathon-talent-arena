import pandas as pd
import re
import json
from datasets import load_dataset
import os



def load_data(file_path, **args):
    """
    Carga los datos desde un archivo JSON.
    
    Esta función es el primer paso para procesar los datos de entrada del hackathon.
    Te permite importar los datos brutos a un DataFrame de Pandas para su fácil manipulación.
    
    Args:
        file_path (str o Path): Ruta al archivo JSON.
        
    Returns:
        pd.DataFrame: DataFrame que contiene los datos listos para el análisis.
    """
    try:
        return pd.read_json(file_path,**args)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def download_hf_dataset(dataset_name, save_path, split="train", **kwargs):
    """
    Descarga un dataset desde Hugging Face y lo guarda en local en formato JSON.
    
    Esta función facilita la obtención de datasets públicos o privados 
    desde el Hugging Face Hub, convirtiéndolos a DataFrame y guardándolos
    en disco con una estructura similar a los archivos JSON que ya utilizamos.
    
    Args:
        dataset_name (str): El nombre del dataset en el Hub (ej: "tatsu-lab/alpaca").
        save_path (str): Ruta donde guardar el archivo JSON (ej: "data/dataset.json").
        split (str, opcional): El split a descargar ("train", "test", etc.). Por defecto "train".
        **kwargs: Argumentos extra para `load_dataset` (ej. `token=True` para repositorios privados).
        
    Returns:
        pd.DataFrame: DataFrame con los datos, o None si hay algún error.
    """
    try:
        
        print(f"Descargando el dataset '{dataset_name}' (split: {split})...")
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        df = dataset.to_pandas()
        
        # Asegurarse de que el directorio donde queremos guardarlo existe
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Guardar como JSON con estructura de lista de registros (orient='records')
        df.to_json(save_path, orient='records', indent=4, force_ascii=False)
        print(f"Dataset guardado exitosamente en: {save_path}")
        
        return df
        
    except ImportError:
        print("Error: Necesitas la librería 'datasets'. Instálala ejecutando 'pip install datasets'.")
        return None
    except Exception as e:
        print(f"Error al descargar el dataset desde Hugging Face: {e}")
        return None



def prepare_dataset(df):
    """
    Prepara y estructura el dataset crudo para las pruebas del hackathon.
    
    Esta función se encarga de:
    1. Extraer el último turno válido de la conversación entre el usuario y el asistente.
    2. Rellenar las respuestas propuestas ('proposed_answer') en caso de que estén vacías con
       la respuesta final, lo cual es útil si el 'verdict' ha sido favorable (passed).
    3. Mapear el 'verdict' ('passed'/'failed') a un formato categórico/binario (1 o 0).
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos crudos originales.
        
    Returns:
        pd.DataFrame: Un DataFrame estructurado, listo para el EDA, evaluación o Fine-Tuning de Prometheus.
    """
    
        
    qa_last_messages = df["raw"].apply(lambda x: get_last_valid_turn(x["messages"])).apply(pd.Series)
    
    df = pd.concat([df[["verdict","challenge","proposed_answer"]], qa_last_messages],axis=1)
    # cuando el verdict es passed, la gente no introduce una proposed_answer, pero es adecuado tener la misma que la que ha hecho el modelo
    # bien para el llm como juez
    
    df['proposed_answer'] = df['proposed_answer'].fillna(df['answer'])
    
    # mapeamos a 0 o 1
    df['verdict'] = df['verdict'].str.lower().str.strip().map({'passed': 1, 'failed': 0}).fillna("").astype(str)
    
    return df

def save_data(data, file_path):
    """
    Guarda los datos procesados en un archivo JSON.
    
    Utiliza esta función para persistir tus DataFrames o listas de diccionarios después de procesarlos,
    generando los archivos de salida necesarios para las entregas del hackathon.
    
    Args:
        data (lista o pd.DataFrame): Los datos que deseas guardar.
        file_path (str o Path): La ruta de destino del archivo JSON a crear.
    """
    try:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


    

def get_last_valid_turn(messages):
    """
    Extrae el último intercambio válido entre el usuario y el asistente de una lista de mensajes.
    
    Busca de atrás hacia adelante el último par donde el rol sea 'assistant' precedido por 'user'.
    Valida que ambos mensajes tengan contenido real y extrae el historial previo para contexto.

    Args:
        messages (list): Lista de diccionarios con claves 'role' y 'content'.

    Returns:
        dict: Diccionario con 'question', 'answer' y 'history', o None si no se encuentra un par válido.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    for i in range(len(messages) - 1, 0, -1):
        assistant_msg = messages[i]
        user_msg = messages[i-1]
        
        # Validación de roles y contenido no vacío
        if (assistant_msg.get("role") == "assistant" and 
            user_msg.get("role") == "user" and
            assistant_msg.get("content", "").strip() and 
            user_msg.get("content", "").strip()):
            
            return {
                "question": user_msg["content"].strip(),
                "answer": assistant_msg["content"].strip(),
                "history": messages[:i-1]
            }
    return None

def format_instruction(sample, system_prompt, absolute_prompt, add_message_history=False, output_col="user_content", **kwargs):
    """
    Construye el prompt estructurado para el modelo Prometheus (LLM-as-a-Judge).
    
    Combina el historial de conversación, la respuesta propuesta, la respuesta de referencia 
    y la rúbrica de evaluación en una plantilla única. Ahora soporta variables dinámicas (kwargs).

    Args:
        sample (dict): Un ejemplo del dataset que contiene 'question', 'proposed_answer', 
                       'answer', 'verdict' y opcionalmente 'history'.
        system_prompt (str): El prompt del sistema general.
        absolute_prompt (str): La plantilla de prompt que puede usar llaves {var} customizadas.
        add_message_history (bool): Si añadir o no el historial.
        output_col (str): Clave de salida.
        **kwargs (dict): Variables extras a inyectar en el absolute_prompt.


    Returns:
        dict: Diccionario con la clave 'user_content' lista para ser procesada por el tokenizer.
    """
    category_name = sample.get('category_name') or ''
    challenge = sample.get('challenge') or ''
    question = sample.get('question') or ''
    proposed_answer = sample.get('proposed_answer') or ''
    answer = sample.get('answer') or ''
    history = sample.get('history') or []
    #verdict = sample.get('verdict') or None

    # Reconstrucción del historial para dar contexto al Juez (opcional)
    context = ""
    if add_message_history:
        context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
        context += f"\nUser: {question}"
    else:
        context = question

    # Construimos un diccionario base con las variables mapeadas obligatoriamente
    base_vars = {
        'category_name': category_name,
        'challenge': challenge,
        'question': context,
        'answer': answer,
        'proposed_answer': proposed_answer
    }
    
    # Le añadimos cualquier otra variable extra que se le haya pasado a format_instruction
    base_vars.update(kwargs)
    
    # Creamos un SafeDict para evitar KeyErrors si el usuario añadió una variable {rara} en el prompt
    class SafeDict(dict):
        def __missing__(self, key):
            return '{' + key + '}' # Si falla dejamos la variable tal cual y no rompemos la app
            
    # Inyección en la plantilla de evaluación absoluta
    user_content = system_prompt + "\n\n" + absolute_prompt.format_map(SafeDict(**base_vars))
    
    return {output_col: user_content}


def prepare_sft_binary_text(sample, tokenizer_eos_token='</s>', output_col_name="prompt_sft", input_col_name="user_content", reasoning_col_name="val_goal_reasoning"):
    """
    Prepara una muestra de datos para el Supervised Fine-Tuning (SFT) de Prometheus.
    """
    prompt = sample.get(input_col_name, "").strip()
    raw_verdict = sample.get("verdict")
    
    # Extraemos el razonamiento. Si por alguna razón está vacío, ponemos un texto genérico de respaldo
    # para no romper el formato de entrenamiento.
    reasoning = sample.get(reasoning_col_name)
    if not reasoning:
        reasoning = "The response is evaluated based on the provided rubric."
    else:
        reasoning = reasoning.strip()
    
    # 1. Normalizar el veredicto
    if isinstance(raw_verdict, str):
        raw_verdict = raw_verdict.strip().lower()

    # 2. Mapeo estricto a binario
    mapping = {
        1: "1", 0: "0", 
        "1": "1", "0": "0",
        "passed": "1", "failed": "0" 
    }
    
    label = mapping.get(raw_verdict)
    
    # 3. Manejo seguro de nulos ANTES de concatenar
    # Si label es None, devolveríamos "[RESULT] None", lo cual contaminaría el entrenamiento.
    if label is None:
        return {output_col_name: ""} 
    
    # 4. El formato PERFECTO para Prometheus
    full_text = f"{prompt}{reasoning} [RESULT] {label}{tokenizer_eos_token}"
    
    return {output_col_name: full_text}