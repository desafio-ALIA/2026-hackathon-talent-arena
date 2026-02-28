import pandas as pd
from datasets import Dataset
from metrics import accuracy, variance, classification_report





class ValidateSubmission:
    """
    Clase para validar el formato de una submission individual 
    y calcular sus métricas de desempeño frente a un dataset de validación real.
    """
    
    def __init__(self, submission_file: str, validation_file: str):
        """
        Inicializa el validador cargando los dataframes.

        Args:
            submission_file (str): Ruta al archivo JSON de submission generado.
            validation_file (str): Ruta al archivo JSON con el ground truth (verdict).
        """
        # Utilizamos try-except por si los ficheros vienen en formatos/orientaciones distintas (orient="records", etc.)
        try:
            self.df_submission = pd.read_json(submission_file)
            self.df_validation = pd.read_json(validation_file)
        except ValueError as e:
            print(f"Error al leer los archivos de validación. Revisa el formato JSON: {e}")
            raise

        self.submission_columns = [
            "id", "team", "po_m_pred", "po_m_reason", 
            "pt_m_pred", "pt_m_reason", "pg_m_pred", "pg_m_reason"
        ]

    def check_submission_size(self):
        """Verifica que el número de filas coincida con lo esperado."""
        # Comprobar número de filas
        size_sub = self.df_submission.shape[0]
        size_val = self.df_validation.shape[0]
        assert size_sub == size_val, f"Error: La submission tiene {size_sub} filas vs las {size_val} del dataset original."

    def check_submission_ids(self):
        """Verifica que los IDs sean idénticos entre la submission y el dataset original."""
        id_col = 'id' if 'id' in self.df_validation.columns else 'iam-id' if 'iam-id' in self.df_validation.columns else 'record_id' if 'record_id' in self.df_validation.columns else None
        
        if id_col and id_col in self.df_validation.columns and 'id' in self.df_submission.columns:
            ids_val = set(self.df_validation[id_col].astype(str))
            ids_sub = set(self.df_submission['id'].astype(str))
            
            faltan = ids_val - ids_sub
            sobran = ids_sub - ids_val
            
            assert not faltan and not sobran, f"Error: Los IDs no coinciden. Faltan: {faltan}, Sobran: {sobran}"

    
    def check_submission_columns(self):
        """Asegura que todas las columnas demandadas estén presentes en el payload enviado."""
        missing = set(self.submission_columns) - set(self.df_submission.columns)
        assert not missing, f"Error: A la submission le faltan estas columnas obligatorias: {missing}"

    def check_verdict_isin_gt(self):
        """
        Verifica que el dataset de validación (ground truth) contenga las columnas 
        necesarias para evaluar el veredicto ('human_val' o 'verdict_validated').
        
        Raises:
            KeyError: Si no se encuentran las columnas de validación requeridas.
        """
        if "human_val" not in self.df_validation.columns and "verdict_validated" not in self.df_validation.columns:
            raise KeyError("No se encontró la columna 'human_val' ni 'verdict_validated' en el dataset de validación.")

    def merge_submission_and_validation(self):
        """
        Une el dataframe de la submission enviada con el dataframe original de validación.
        Intenta hacer el join mediante la columna 'id'. Si no es posible, realiza 
        una concatenación simple por posición.
        
        Returns:
            pd.DataFrame: DataFrame fusionado con las predicciones y el ground truth.
        """
        val_id_col = 'id' if 'id' in self.df_validation.columns else 'iam-id' if 'iam-id' in self.df_validation.columns else 'record_id' if 'record_id' in self.df_validation.columns else None
        
        if "id" in self.df_submission.columns and val_id_col:
            df_merged = self.df_submission.merge(self.df_validation, left_on="id", right_on=val_id_col, how="inner")
        else:
            print("Advertencia: No se encontró la columna 'id' para hacer merge. Usando índices posicionales para métricas.")
            df_merged = pd.concat([
                self.df_submission.reset_index(drop=True), 
                self.df_validation.reset_index(drop=True)
            ], axis=1)
        return df_merged
    
    def calculate_metrics(self) -> tuple:
        """
        Deduce el 'accuracy' comparando predicciones de orígenes ('po_m_pred') contra el veredicto validado real ('verdict_validated').
        Asimismo, calcula la tasa de 'variabilidad' que mide en qué porcentaje de las filas 
        el resultado del modelo difiere según el prompt evaluado (original, con typos y con gramática).
        
        Returns:
            tuple: (accuracy, variability) en formato decimal.
        """
        df_merged = self.merge_submission_and_validation()
        
        po_pred = df_merged["po_m_pred"].astype(str).str.strip()
        pt_pred = df_merged["pt_m_pred"].astype(str).str.strip()
        pg_pred = df_merged["pg_m_pred"].astype(str).str.strip()

        # Extraer el verdict del ground truth
        def extract_verdict(row):
            if "human_val" in row and isinstance(row["human_val"], dict):
                v = row["human_val"].get("verdict_validated")
            else:
                v = row.get("verdict_validated")
                
            if isinstance(v, dict):
                return v.get("value")
            return v
            
        verdict_raw = df_merged.apply(extract_verdict, axis=1)
        verdict = verdict_raw.astype(str).str.strip().str.lower()
        verdict = verdict.map({
            "passed": "1", "failed": "0", 
            "1": "1", "0": "0",
            "seguro": "1", "hackeado": "0" 
        }).fillna(verdict)

        # Calcula Accuracy usando la nueva función
        acc = accuracy(verdict, po_pred)

        # Calcula Variabilidad usando local metrics
        variability = variance(po_pred, pt_pred, pg_pred)
        
        # Classification report (puedes descomentar para imprimir/loggear o utilizar)
        # report = classification_report(verdict, po_pred)

        return float(acc), float(variability)

    def check_all(self):
        """Corre todas las validaciones de sanidad de la data del participante."""
        self.check_submission_size()
        self.check_submission_ids()
        self.check_submission_columns()
        self.check_verdict_isin_gt()
    
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
    def __init__(self, submission_files: list, validation_file: str ):
        """
        Args:
            submission_files (list[str]): Rutas a los diferentes JSON a cualificar.
            validation_file (str): Ruta al data frame verificado.
        """
        self.submission_files = submission_files
        self.validation_file = validation_file
    
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
                validator = ValidateSubmission(submission_file, self.validation_file)
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

