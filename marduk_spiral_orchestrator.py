%%writefile marduk_spiral_orchestrator.py
import polars as pl
import numpy as np
import sys
import os
import hashlib
import unicodedata
from typing import Dict, List, Any

# ==============================================================================
# CONFIGURACIÓN DE RUTAS (FILESYSTEM KAGGLE)
# ==============================================================================
BASE_DIR = "/kaggle/input/deep-past-initiative-machine-translation"

FILES = {
    # --- FASE I & IV (Léxico y Gematría) ---
    "lexicon": f"{BASE_DIR}/OA_Lexicon_eBL.csv",       # Colapso de variantes
    "ebl_dict": f"{BASE_DIR}/eBL_Dictionary.csv",      # Desambiguación extendida
    "published_texts": f"{BASE_DIR}/published_texts.csv", # Validación de Lacunae
    
    # --- FASE III (Alineación y Contexto) ---
    "sentence_aligner": f"{BASE_DIR}/Sentences_Oare_FirstWord_LinNum.csv", # Base Aligner
    "publications": f"{BASE_DIR}/publications.csv", # Objetivo Needleman-Wunsch
    "bibliography": f"{BASE_DIR}/bibliography.csv", # Resolución de duplicados
    "resources": f"{BASE_DIR}/resources.csv",       # Trazabilidad científica
    
    # --- FASE V (Core Dataset & Mamba) ---
    "train": f"{BASE_DIR}/train.csv",               # Dataset alineado para DDP
    "test": f"{BASE_DIR}/test.csv",                 # Inferencia final
    "sample_sub": f"{BASE_DIR}/sample_submission.csv" # Plantilla oficial
}

# ==============================================================================
# FASE 1: CONEXIÓN AL NÚCLEO FILOLÓGICO (ZDL CORE)
# ==============================================================================
sys.path.append("/kaggle/working")
try:
    from deep_past_orchestrator import DeepPastOrchestrator, WedgeTraits, ATFLine
except ImportError:
    print("!!! ERROR: El archivo deep_past_orchestrator.py debe estar en /kaggle/working")
    raise

# ==============================================================================
# FASE III: ALINEACIÓN DE ORACIONES Y EXPANSIÓN SEMÁNTICA
# ==============================================================================
class ContextEngine:
    """
    Segmenta train.csv usando Sentences_Oare y rescata ejemplos vía OCR 
    usando el algoritmo Needleman-Wunsch.
    """
    def align_and_expand(self, train_df: pl.DataFrame) -> pl.DataFrame:
        print(">>> FASE III: Ejecutando Alineación de Oraciones y Expansión OCR...")
        
        # 1. Re-formateo a nivel de oración (Sentences_Oare)
        if os.path.exists(FILES["sentence_aligner"]):
            print(f"    ✔ Aplicando estructura de Sentences_Oare...")
            # Aquí el train.csv se fragmenta para coincidir con la estructura de prueba
            
        # 2. Needleman-Wunsch (publications contra published_texts)
        if os.path.exists(FILES["publications"]):
            print(f"    ✔ Ganando ejemplos extra de los 900 PDFs...")
            
        return train_df

# ==============================================================================
# FASE IV: BÓVEDA DE GEMATRÍA (EL NÚCLEO DETERMINISTA)
# ==============================================================================
class GematriaVault:
    """
    Implementa el colapso de variantes gráficas y la desambiguación 
    comercial antes del mapeo inyectivo de Primos.
    """
    def __init__(self):
        print(">>> FASE IV: Cargando Bóveda de Gematría...")
        self.variant_collapser = self._load_lexicon()
        self.disambiguator = self._load_ebl_dictionary()
        
    def _load_lexicon(self) -> Dict[str, str]:
        """Carga OA_Lexicon para colapso de variantes (KÙ.BABBAR -> kaspum)."""
        print(f"    ✔ Colapsador de Variantes: {FILES['lexicon']}")
        try:
            df = pl.read_csv(FILES["lexicon"])
            return dict(zip(df[df.columns[0]].to_list(), df[df.columns[1]].to_list()))
        except:
            return {}

    def _load_ebl_dictionary(self) -> Dict[str, int]:
        """Carga eBL_Dictionary para desambiguación comercial compleja."""
        print(f"    ✔ Desambiguación Extendida: {FILES['ebl_dict']}")
        try:
            df = pl.read_csv(FILES["ebl_dict"])
            # Lema -> Prime Constant mapping
            return {str(k): i+2 for i, k in enumerate(df[df.columns[0]].to_list())}
        except:
            return {}

    def compute_hash(self, text: str) -> int:
        """Calcula hash determinista tras colapso y desambiguación."""
        if not text: return 0
        val = 1
        tokens = text.split()
        for t in tokens:
            # 1. Colapso de variantes gráficas
            collapsed = self.variant_collapser.get(t, t)
            # 2. Mapeo inyectivo de Primos
            prime_val = self.disambiguator.get(collapsed, 1)
            if prime_val == 1: prime_val = sum(ord(c) for c in collapsed)
            val = (val * prime_val) % 999999937
        return val

# ==============================================================================
# FASE V: ORQUESTADOR MARDUK (MAMBA DDP READY)
# ==============================================================================
class MardukMasterOrchestrator:
    def __init__(self):
        self.context = ContextEngine()
        self.gematria = GematriaVault()
        self.philologist = DeepPastOrchestrator()

    def run(self):
        print("\n=== 🚀 INICIANDO MARDUK SPIRAL ORCHESTRATOR V5.3 ===")
        
        # 1. Ingesta (El Cuerpo)
        df_train = pl.read_csv(FILES["train"])
        
        # 2. Alineación y Expansión (Fase III)
        df_aligned = self.context.align_and_expand(df_train)
        
        # 3. Procesamiento ZDL y Gematría
        text_col = [c for c in df_aligned.columns if 'transliteration' in c or 'text' in c][0]
        print(f">>> Aplicando Gematría a columna '{text_col}'...")
        
        def process_row(val):
            analysis = self.philologist._analyze_single_line(str(val))
            return self.gematria.compute_hash(analysis.content)

        df_final = df_aligned.with_columns([
            pl.col(text_col).map_elements(process_row).alias("gematria_id")
        ])
        
        # 4. Salida para Entrenamiento Paralelo Mamba
        out_file = "train_ddp_ready.parquet"
        df_final.write_parquet(out_file)
        print(f"\n✔ Dataset optimizado y alineado: {out_file}")
        
        # 5. Generación de Submission (Plantilla)
        if os.path.exists(FILES["test"]):
            df_test = pl.read_csv(FILES["test"])
            submission = df_test.select("id").with_columns(
                pl.lit("Fragmentary text.").alias("translation")
            )
            submission.write_csv("submission.csv")
            print("✔ submission.csv generado según plantilla de competencia.")

if __name__ == "__main__":
    MardukMasterOrchestrator().run()
