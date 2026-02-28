import pandas as pd
import re
import json
from datasets import load_dataset
import os
import string


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
            data = data.to_dict(orient='records',
                                 indent=2,
                                 force_ascii=False)
        
        elif isinstance(data, Dataset):
            data = data.to_pandas().to_dict(orient='records',
                                 indent=2,
                                 force_ascii=False)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def message_to_conversation_str(history, question=""):
    """
    Convierte una lista de mensajes a un string de conversación previa a la respuesta del modelo.
    
    Args:
        history (list): Lista de diccionarios con claves 'role' y 'content'.
        question (str, opcional): Pregunta final del usuario.

    Returns:
        str: String de conversación previa a la respuesta del modelo.
    """
    content = "\n".join([f"{m.get('role', '').capitalize()}: {m.get('content', '')}" for m in history])
    if question:
        content += f"\nUser: {question}"
    return content
    

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
                "history": messages[:i-1],
                "conversation": message_to_conversation_str(messages[:i-1], user_msg["content"].strip())
            }
    return None




def extract_prompt_variables(sample, user_prompt):
    """
    Identifica las variables requeridas en una plantilla de prompt y las extrae del sample.
    Si alguna variable requerida no está en el sample, lanza un KeyError.

    Args:
        sample (dict o pd.Series): Un ejemplo del dataset que contiene las variables.
        user_prompt (str): La plantilla de prompt que usa llaves {var}.

    Returns:
        dict: Diccionario cerrado con únicamente las variables requeridas por el prompt.
    """
    # Identificar las variables que pide la plantilla de forma dinámica
    vars_in_prompt = [fname for _, fname, _, _ in string.Formatter().parse(user_prompt) if fname is not None]
    
    base_vars = {}
    
    # Extraer las variables del sample, validando que existan
    for var in vars_in_prompt:
        if var not in sample:
            raise KeyError(f"La variable '{var}' requerida en el prompt no está presente en el sample.")
        base_vars[var] = sample[var]
        
    return base_vars

def format_instruction(sample, system_prompt, user_prompt, output_col="user_content"):
    """
    Construye el prompt estructurado para el modelo Prometheus (LLM-as-a-Judge).
    
    Extrae las variables del prompt dinámicamente desde el sample. Si alguna variable
    requerida en la plantilla del prompt no está en el sample, lanzará un KeyError.

    Args:
        sample (dict o pd.Series): Un ejemplo del dataset que contiene las variables.
        system_prompt (str): El prompt del sistema general.
        user_prompt (str): La plantilla de prompt que usa llaves {var}.
        output_col (str): Clave de salida.

    Returns:
        dict: Diccionario con la clave 'user_content' lista para ser procesada por el tokenizer.
    """
    base_vars = extract_prompt_variables(sample, user_prompt)
            
    # Inyección en la plantilla de evaluación absoluta
    user_content = system_prompt + "\n\n" + user_prompt.format(**base_vars)
    
    return {output_col: user_content}


def prepare_sft_binary_text(sample, tokenizer_eos_token='</s>', output_col_name="prompt_sft", 
                            input_col_name="user_content", reasoning_col_name="val_goal_reasoning",
                            label_col_name="verdict"):
    """
    Prepara una muestra de datos para el Supervised Fine-Tuning (SFT) de Prometheus.
    """
    prompt = sample.get(input_col_name, "").strip()
    raw_verdict = sample.get(label_col_name)
    
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