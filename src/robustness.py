from datasets import Dataset


def create_robustness_dataset(df_input: pd.DataFrame = None, pred_col="user_content") -> Dataset:
    """
    Crea un dataset con distintas variaciones de ruido o corrupción 
    (typos, errores gramaticales, etc.) aplicadas a los prompts originales
    para evaluar la robustez del modelo.
    
    Args:
        df_input (pd.DataFrame): DataFrame de entrada con los prompts originales.
        pred_col (str): Nombre de la columna que contiene los prompts. Por defecto "user_content".

    Returns:
        Dataset: Dataset de HuggingFace con las variaciones generadas.
    """
    # 1. Definimos la configuración base como un diccionario
    config_notebook = {
        "n_typos": 1,
        "n_grammar_changes": 5,
        
        "typo_type_weights": {
            "qwerty": 0.55,
            "omission": 0.25,
            "abbr": 0.20,
            "space_remove": 0.10
        },
        
        "vowel_delete_bias": 0.9,
        "abbr_q_weight": 0.6,
        "abbr_pq_weight": 0.4,
        
        "grammar_rule_weights": {
            "habia_to_habian": 1.0,
            "hemos_to_habemos": 0.9,
            "homophones": 0.8,
            "porque": 0.9,
            "seseo_ceceo": 0.9,
            "preterite_s": 0.9,
            "drop_initial_h": 0.9,
            "swap_bv": 0.9
        },
        
        "remove_open_questions": True,
        "strip_accents": True,
        "remove_commas": True,
        "lowercase": True
    }


    prompts = df_input[pred_col].tolist()

    # Usamos process_prompts en lugar de process_csv ya que así evitamos problemas
    # de compatibilidad si el dataset de entrada es un JSON o ya viene cargado en pandas.
    custom_cfg = CustomConfig(**config_notebook)
    results = process_prompts(prompts, custom_cfg=custom_cfg)
    
    df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(df)

    return dataset

def model_preds(model, tokenizer, dataset: Dataset, input_col: str, suffix: str) -> Dataset:
        """
        Genera las predicciones para una columna en específico del dataset y formatea la salida
        (razonamiento vs puntuación).

        Args:
            model: Modelo a utilizar (Prometheus, etc.).
            tokenizer: Tokenizador del modelo.
            dataset (Dataset): Dataset con los inputs.
            input_col (str): Nombre de la columna de entrada.
            suffix (str): Sufijo para nombrar la columna de salida.

        Returns:
            Dataset: Dataset con las nuevas columnas calculadas.
        """
        dataset = dataset.map(
            model_predict_batched, 
            batched=True, 
            batch_size=8, 
            fn_kwargs={
                "model": model, 
                "tokenizer": tokenizer, 
                "input_col": input_col, 
                "output_suffix": suffix
            }
        )
        dataset = dataset.map(
            split_model_reason_result, 
            fn_kwargs={"output_suffix": suffix}
        )
        return dataset

def model_preds_robustness(model, tokenizer, dataset: Dataset) -> Dataset:
        """
        Aplica predicción de modelo sobre todas las columnas con variaciones 
        (original, typos, grammatical errors) dentro del dataset.

        Args:
            model: Modelo de inferencia.
            tokenizer: Tokenizador.
            dataset (Dataset): Dataset generado previamente que contiene las variaciones.

        Returns:
            Dataset: Dataset con los resultados e inferencias generadas.
        """
        cols = ["prompt_original", "prompt_typos", "prompt_grammatical_errors"]
        
        # Corrección: Extraemos la primera letra de las dos primeras palabras para formar el sufijo (po, pt, pg)
        create_suffix = lambda x: "".join([p[0] for p in x.split("_")[:2]]) + "_m"
        col_and_suffix = [(col, create_suffix(col)) for col in cols]

        for col, suffix in col_and_suffix:
            dataset = model_preds(model, tokenizer, dataset, col, suffix)
        
        return dataset