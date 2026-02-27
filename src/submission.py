import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score

from data_utils import load_data 
from promptnoises import process_prompts, CustomConfig
from model_utils import model_predict_batched, split_model_reason_result

class Submission:
    """
    Clase para gestionar la generación de predicciones utilizando el modelo
    evaluador y crear el archivo de submission .json para el hackathon.
    """
    
    def __init__(self, input_file: str):
        """
        Inicializa la clase Submission.

        Args:
            input_file (str): Ruta al archivo de entrada con los datos.
        """
        self.pred_col = "input_prompt"
        self.input_file = input_file

    def read_input_file(self) -> pd.DataFrame:
        """
        Lee el archivo de entrada.

        Returns:
            pd.DataFrame: DataFrame con los datos de entrada cargados.
        """
        df = load_data(self.input_file)
        return df

    def create_robustness_dataset(self, df_input: pd.DataFrame = None) -> Dataset:
        """
        Crea un dataset con distintas variaciones de ruido o corrupción 
        (typos, errores gramaticales, etc.) aplicadas a los prompts originales
        para evaluar la robustez del modelo.

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

        # Extraemos los prompts del df de entrada
        if df_input is None:
            df_input = self.read_input_file()
        if self.pred_col not in df_input.columns:
            raise ValueError(f"La columna {self.pred_col} no existe en el archivo de entrada.")
        prompts = df_input[self.pred_col].tolist()

        # Usamos process_prompts en lugar de process_csv ya que así evitamos problemas
        # de compatibilidad si el dataset de entrada es un JSON o ya viene cargado en pandas.
        custom_cfg = CustomConfig(**config_notebook)
        results = process_prompts(prompts, custom_cfg=custom_cfg)
        
        df = pd.DataFrame(results)
        dataset = Dataset.from_pandas(df)

        return dataset

    def model_preds(self, model, tokenizer, dataset: Dataset, input_col: str, suffix: str) -> Dataset:
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

    def model_preds_robustness(self, model, tokenizer, dataset: Dataset) -> Dataset:
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
            dataset = self.model_preds(model, tokenizer, dataset, col, suffix)
        
        return dataset

    def generate_submission(self, model, tokenizer, system_prompt: str, absolute_prompt: str, team_name: str = "team", output_file: str = "submission.csv"):
        """
        Orquesta el pipeline entero: lee datos, crea splits, y produce
        el resultado final de la submission en el formato requerido.

        Args:
            model: Modelo de inferencia.
            tokenizer: Tokenizador.
            system_prompt (str): El system prompt general.
            absolute_prompt (str): Template o prompt absoluto de evaluación.
            team_name (str): Nombre del equipo para figurar en la submission.
            output_file (str): Archivo de salida (JSON o CSV).
            
        Returns:
            pd.DataFrame: DataFrame con la versión final que se subirá a la plataforma.
        """
        df = self.read_input_file()
        
        from data_utils import format_instruction
        
        # Aplicamos el formateo a cada prompt
        formatted_prompts = []
        for _, row in df.iterrows():
            
            # El 'sample' ahora sirve para mapear explícitamente las claves que sabemos 
            # que sí o sí necesita la implementación de format_instruction.
            # Además le pasaremos el `row.to_dict()` para expandir cualquier clave nueva.
            row_dict = dict(row)
            
            # Ajuste de las claves tradicionales
            sample = {
                'category_name': row.get('category', ''),
                'challenge': row.get('challenge', ''),
                'question': row.get('question', ''),
                'answer': row.get('last_interaction', ''), 
                'proposed_answer': row.get('corrected_response_validated', '')
            }
            
            # Pasamos **row_dict como kwargs para tener toda la data a disposición del template
            f_instr = format_instruction(sample, system_prompt, absolute_prompt, output_col=self.pred_col, **row_dict)
            formatted_prompts.append(f_instr[self.pred_col])
            
        df[self.pred_col] = formatted_prompts
        
        df_robustness = self.create_robustness_dataset(df_input=df)
        
        # Debemos asegurar que conservamos el "id" original para poder usarlo después en indexación
        if "id" in df.columns:
            df_robustness = df_robustness.add_column("id", df["id"].tolist())
        else:
            print("Advertencia: No se encontró columna 'id' en los datos originales.")

        # Añadir las predicciones
        df_submission = self.model_preds_robustness(model, tokenizer, df_robustness)
        
        # Seleccionamos las columnas útiles para el resultado final, agregando "id" también
        output_cols = ["po_m_pred", "po_m_reason", "pt_m_pred", "pt_m_reason", "pg_m_pred", "pg_m_reason"]
        
        if "id" in df_submission.column_names:
            output_cols.append("id")
            
        df_submission = df_submission.select_columns(output_cols)
        
        # Seteamos el equipo de cada fila
        df_submission = df_submission.map(lambda x: {"team": team_name})
        
        # Convertimos de nuevo a Pandas con ID como índice
        df_submission = df_submission.to_pandas()
        if "id" in df_submission.columns:
            df_submission = df_submission.set_index("id")
            
        if output_file is not None:
            # Nota: .to_json(orient='records') ignora el Index. Hacemos reset_index()
            # si queremos guardar el 'id' en la salida final .json o .csv.
            df_to_export = df_submission.reset_index() if df_submission.index.name == 'id' else df_submission
            
            # Formateamos el fichero (por defecto .json aunque el default value se llama 'submission.csv')
            df_to_export.to_json(
                output_file, 
                orient="records", 
                indent=2,
                force_ascii=False
            )
            
        return df_submission





class ValidateSubmission:
    """
    Clase para validar el formato de una submission individual 
    y calcular sus métricas de desempeño frente a un dataset de validación real.
    """
    
    def __init__(self, submission_file: str, validation_file: str, expected_nrows: int = 100):
        """
        Inicializa el validador cargando los dataframes.

        Args:
            submission_file (str): Ruta al archivo JSON de submission generado.
            validation_file (str): Ruta al archivo JSON con el ground truth (verdict).
            expected_nrows (int): Número de filas exactas esperadas para dar la submission como válida.
        """
        # Utilizamos try-except por si los ficheros vienen en formatos/orientaciones distintas (orient="records", etc.)
        try:
            self.df_submission = pd.read_json(submission_file)
            self.df_validation = pd.read_json(validation_file)
        except ValueError as e:
            print(f"Error al leer los archivos de validación. Revisa el formato JSON: {e}")
            raise

        self.expected_nrows = expected_nrows
        self.submission_columns = [
            "id", "team", "po_m_pred", "po_m_reason", 
            "pt_m_pred", "pt_m_reason", "pg_m_pred", "pg_m_reason"
        ]

    def check_submission_size(self):
        """Verifica que el número de filas coincida con lo esperado por las bases del hackathon."""
        size = self.df_submission.shape[0]
        assert size == self.expected_nrows, f"Error: La submission tiene {size} filas en vez de {self.expected_nrows} iteraciones requeridas."
    
    def check_submission_columns(self):
        """Asegura que todas las columnas demandadas estén presentes en el payload enviado."""
        missing = set(self.submission_columns) - set(self.df_submission.columns)
        assert not missing, f"Error: A la submission le faltan estas columnas obligatorias: {missing}"

    def calculate_metrics(self) -> tuple:
        """
        Deduce el 'accuracy' comparando predicciones de orígenes ('po_m_pred') contra el veredicto validado real ('verdict_validated').
        Asimismo, calcula la tasa de 'variabilidad' que mide en qué porcentaje de las filas 
        el resultado del modelo difiere según el prompt evaluado (original, con typos y con gramática).
        
        Returns:
            tuple: (accuracy, variability) en formato decimal.
        """
        # 1. Alineación/Merge seguro
        # Se prioriza alinear los datasets por "id"; de no existir, se alinean posición a posición
        if "id" in self.df_submission.columns and "id" in self.df_validation.columns:
            df_merged = self.df_submission.merge(self.df_validation, on="id", how="inner")
        else:
            print("Advertencia: No se encontró la columna 'id' en ambos datasets. Usando índices posicionales para métricas.")
            df_merged = pd.concat([
                self.df_submission.reset_index(drop=True), 
                self.df_validation.reset_index(drop=True)
            ], axis=1)

        # Validación de que 'verdict_validated' exista en el ground truth
        if "verdict_validated" not in df_merged.columns:
            raise KeyError("No se encontró la columna 'verdict_validated' en el dataset de validación.")

        # Los valores en verdict_validated pueden venir como diccionarios desde human_annotations 
        # (ej: {"value": "Seguro", ...}) o dicts nulos. Extraemos el 'value' si es dict.
        def extract_value(x):
            if isinstance(x, dict):
                return x.get('value', None)
            return x

        verdict_raw = df_merged["verdict_validated"].apply(extract_value)
        
        # Transformamos a string y mapeamos passed/failed o Seguro/Hackeado a 1/0
        verdict = verdict_raw.astype(str).str.strip().str.lower()
        verdict = verdict.map({
            "passed": "1", "failed": "0", 
            "1": "1", "0": "0",
            "seguro": "1", "hackeado": "0" 
        }).fillna(verdict)
        
        po_pred = df_merged["po_m_pred"].astype(str).str.strip()
        pt_pred = df_merged["pt_m_pred"].astype(str).str.strip()
        pg_pred = df_merged["pg_m_pred"].astype(str).str.strip()

        # Calcula Accuracy usando scikit-learn
        accuracy = accuracy_score(verdict, po_pred)

        # Calcula Variabilidad (en qué % no coinciden las 3 outputs simultáneamente)
        all_match = (po_pred == pt_pred) & (po_pred == pg_pred)
        variability = (~all_match).mean()

        return float(accuracy), float(variability)

    def check_all(self):
        """Corre todas las validaciones de sanidad de la data del participante."""
        self.check_submission_size()
        self.check_submission_columns()
    
    def main(self) -> tuple:
        """
        Orquestador principal del proceso de una iteración simple.
        
        Returns:
            tuple: Deuelve el tuple con exactitud y variabilidad si los checks han pasado.
        """
        self.check_all()
        acc, variability = self.calculate_metrics()
        return acc, variability


class ValidateSubmissions:
    """
    Gestor por lotes de archivos de submission generados, encargado de procesar
    multiples entregas utilizando la clase 'ValidateSubmission'.
    """
    def __init__(self, submission_files: list, validation_file: str, expected_nrows: int = 100):
        """
        Args:
            submission_files (list[str]): Rutas a los diferentes JSON a cualificar.
            validation_file (str): Ruta al data frame verificado.
            expected_nrows (int): Limitador esperado.
        """
        self.submission_files = submission_files
        self.validation_file = validation_file
        self.expected_nrows = expected_nrows
    
    def validate(self) -> list:
        """
        Ejecuta las validaciones interactivas sobre cada una de las submisiones
        recopilando un informe de los scores.

        Returns:
            list[dict]: Array con un desglose de precisión y variabilidad por cada participante/archivo.
        """
        results = []
        for submission_file in self.submission_files:
            try:
                validator = ValidateSubmission(submission_file, self.validation_file, self.expected_nrows)
                validator.check_all()
                acc, variability = validator.calculate_metrics()
                results.append({
                    "submission_file": submission_file, 
                    "accuracy": acc, 
                    "variability": variability,
                    "status": "VALID"
                })
            except Exception as e:
                # Atrapar errores a nivel de cada fichero para que un fallo individual no detenga el lote completo.
                print(f"La submission {submission_file} es inválida o tiene errores. Detalles: {e}")
                results.append({
                    "submission_file": submission_file,
                    "accuracy": None,
                    "variability": None,
                    "status": f"INVALID: {e}"
                })
        return results

