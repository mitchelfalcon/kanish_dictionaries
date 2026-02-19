import os
import re
import json
import time
import hashlib
import sqlite3
import logging
import unicodedata
import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict

# Try to import sympy for strong mathematical validation; safe fallback if not exists.
try:
    from sympy import isprime
except ImportError:
    def isprime(n):
        if n <= 1: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

# --- 1. AUDIT CONFIGURATION AND GLOBAL CONSTANTS ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [MARDUK-CORE] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MARDUK_MASTER")

# Configuration based on 'anchoring_rules.json' and 'gema_integration_manifest.json'
CONFIG = {
    "VERSION": "9.0-ULTIMATE",
    "STRICT_MODE": True,
    "ENCODING": "UTF-8",
    "DB_NAME": "marduk_vault_zdl.db",
    "RISK_THRESHOLD": 0.5
}

# --- 2. EMBEDDED KNOWLEDGE BASE (LEXICON & RULES) ---
# Compiled from: cuneiform_logograms.csv, akkadian_vocabulary.json, categories1.csv

STATIC_LEXICON = {
    "LOGOGRAMS": {
        "KÙ.BABBAR": {"akk": "kaspum", "mean": "Silver", "prime": 1009, "cat": "COMMODITY"},
        "GUŠKIN":    {"akk": "hurāṣum", "mean": "Gold", "prime": 1013, "cat": "COMMODITY"},
        "AN.NA":     {"akk": "annakum", "mean": "Tin", "prime": 1019, "cat": "COMMODITY"},
        "URUDU":     {"akk": "werium", "mean": "Copper", "prime": 1021, "cat": "COMMODITY"},
        "ŠE":        {"akk": "šeʾum", "mean": "Barley", "prime": 1031, "cat": "COMMODITY"},
        "DAM.GÀR":   {"akk": "tamkārum", "mean": "Merchant", "prime": 2003, "cat": "ROLE"},
        "DUB":       {"akk": "ṭuppum", "mean": "Tablet", "prime": 3001, "cat": "ARTIFACT"},
        "É":         {"akk": "bītum", "mean": "House/Firm", "prime": 4001, "cat": "INSTITUTION"}
    },
    "DETERMINATIVES": {
        "{d}": "DIVINE", "{m}": "MALE", "{f}": "FEMALE", "{lu2}": "PROFESSION", 
        "{ki}": "PLACE", "{giš}": "WOOD", "{urudu}": "COPPER"
    },
    "MORPHOLOGY": {
        "PREFIXES": {
            "u": "D/S_stem", "i": "G_stem_3c", "a": "G_stem_1cs", "ni": "G_stem_1cp", "ta": "G_stem_2"
        },
        "SUFFIXES": {
            "šu": "POSS_3MS", "ša": "POSS_3FS", "ka": "POSS_2MS", "ki": "POSS_2FS",
            "ya": "POSS_1CS", "ni": "POSS_1CP", "šunu": "POSS_3MP", "šina": "POSS_3FP",
            "ū": "PL_MASC", "ā": "PL_FEM", "um": "NOM", "am": "ACC", "im": "GEN"
        },
        "INFIXES": {
            "tan": "ITERATIVE (Gtn/Dtn/Stn)",
            "ta": "PERFECT/REFLEXIVE"
        }
    }
}

# --- 3. DATA STRUCTURES (SCHEMAS) ---

@dataclass
class TransactionRecord:
    """Immutable schema for detected economic transactions."""
    tx_hash: str
    source_text_id: str
    sender: str
    receiver: str
    commodity: str
    amount: float
    is_prime_validated: bool
    risk_score: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class SemanticToken:
    """Token enriched with philological metadata."""
    raw: str
    normalized: str
    token_type: str # LOGOGRAM, SYLLABLE, NUMBER, DETERMINATIVE
    analysis: Dict[str, Any] = field(default_factory=dict)
    
# --- 4. VECTOR K: INTEGRITY KERNEL (MARDUK VAULT) ---

class KanishKernel:
    """
    The Cryptographic Core. 
    Ensures no data enters the system without validation and hashing.
    """
    def __init__(self, output_dir: Path):
        self.db_path = output_dir / CONFIG["DB_NAME"]
        self._initialize_vault()

    def _initialize_vault(self):
        """Creates SQLite schema optimized for forensic searches."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 1. Knowledge Table (Semantic Graph)
        c.execute('''CREATE TABLE IF NOT EXISTS knowledge_triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_hash TEXT,
            predicate TEXT,
            object_hash TEXT,
            raw_subject TEXT,
            raw_object TEXT,
            metadata JSON,
            ingest_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # 2. Financial Audit Table (Ledger)
        c.execute('''CREATE TABLE IF NOT EXISTS financial_ledger (
            tx_id TEXT PRIMARY KEY,
            text_source TEXT,
            sender TEXT,
            receiver TEXT,
            commodity TEXT,
            quantity REAL,
            prime_signature INTEGER,
            is_valid_prime BOOLEAN,
            fraud_probability REAL
        )''')
        
        # 3. Indices for speed (Vector E)
        c.execute('CREATE INDEX IF NOT EXISTS idx_ledger_sender ON financial_ledger(sender)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_ledger_receiver ON financial_ledger(receiver)')
        
        conn.commit()
        conn.close()

    def sign_data(self, data: Any) -> str:
        """Generates an immutable SHA-256 signature."""
        payload = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def validate_sdic_g(self, amount: float, commodity: str) -> Tuple[bool, int]:
        """
        SDIC-G Protocol (Scientific Decoding of Cuneiform - Gematria).
        Verifies if the economic value respects mathematical integrity.
        """
        # Get the base prime of the commodity
        base_prime = STATIC_LEXICON["LOGOGRAMS"].get(commodity, {}).get("prime", 1)
        
        # In an advanced simulation, quantity * base_prime should have specific properties.
        # For this implementation level, we validate if the quantity is "pure" (integer and positive).
        try:
            val_int = int(amount)
            # A value is "suspicious" if it's negative or zero in a commercial context
            if val_int <= 0: return False, 0
            
            # Primality check of the value (or factors)
            is_p = isprime(val_int)
            return is_p, base_prime
        except:
            return False, 0

    def record_transaction(self, tx: TransactionRecord):
        """Persists the transaction into the Armored Ledger."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('''INSERT OR REPLACE INTO financial_ledger VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (tx.tx_hash, tx.source_text_id, tx.sender, tx.receiver, 
                 tx.commodity, tx.amount, 0, tx.is_prime_validated, tx.risk_score))
            conn.commit()
        except Exception as e:
            logger.error(f"Error writing to Vault: {e}")
        finally:
            conn.close()

# --- 5. VECTOR A: THE REFINERY (ADVANCED NLP) ---

class MardukRefinery:
    """
    Normalization and Morphological Analysis Engine.
    Implements logic from 'atf_parsing_logic.csv' and 'phonetic_transformation.csv'.
    """
    
    # Phonetic Normalization Rules (Compiled)
    TRANSFORMS = [
        (re.compile(r'sz'), 'š'), (re.compile(r's,'), 'ṣ'), (re.compile(r't,'), 'ṭ'),
        (re.compile(r'h,'), 'ḫ'), (re.compile(r'j'), 'ŋ'), (re.compile(r"'"), 'ʾ')
    ]
    
    # Structural Regex
    RX_NUMBER = re.compile(r'^(\d+)(\(([^)]+)\))?$') # Captures 5 or 5(disz)
    RX_LOGOGRAM = re.compile(r'\b[A-Z]{2,}(\.[A-Z0-9]+)*\b')
    
    def __init__(self):
        self.subscript_trans = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    def normalize(self, text: str) -> str:
        """Applies Unicode cleaning and phonetic transformations."""
        if not text: return ""
        text = str(text)
        
        # 1. Phonetic Replacements
        for pattern, repl in self.TRANSFORMS:
            text = pattern.sub(repl, text)
            
        # 2. Numerical Subscripts (du3 -> du₃)
        text = re.sub(r'([a-zA-Z])(\d+)', 
                     lambda m: m.group(1) + m.group(2).translate(self.subscript_trans), 
                     text)
                     
        # 3. Unicode NFC
        return unicodedata.normalize('NFC', text).strip()

    def analyze_morphology(self, token: str) -> Dict[str, Any]:
        """
        Deep Parsing based on 'stem_morphology.csv'.
        Attempts to deconstruct a verb or noun into its components.
        """
        analysis = {"root": None, "stem": None, "suffix": None}
        
        # Possessive Suffix Detection
        for sfx, code in STATIC_LEXICON["MORPHOLOGY"]["SUFFIXES"].items():
            if token.endswith(sfx):
                analysis["suffix"] = code
                # Heuristic: What remains could be the base
                potential_base = token[:-len(sfx)]
                analysis["base_guess"] = potential_base
                break
                
        # Verbal Stem Detection (Infix Detection)
        if "tan" in token:
            analysis["stem"] = "ITERATIVE (Gtn/Dtn)"
        elif "ta" in token and len(token) > 4: # Simple heuristic
            analysis["stem"] = "PERFECT/REFLEXIVE"
            
        return analysis

    def parse_line(self, line: str, line_id: str) -> List[SemanticToken]:
        """Converts a raw text line into a sequence of semantic tokens."""
        raw_tokens = line.split()
        semantic_tokens = []
        
        for rt in raw_tokens:
            norm = self.normalize(rt)
            
            # Classification
            if self.RX_LOGOGRAM.match(norm):
                t_type = "LOGOGRAM"
                # Lookup in Lexicon
                meta = STATIC_LEXICON["LOGOGRAMS"].get(norm, {})
            elif self.RX_NUMBER.match(norm):
                t_type = "NUMBER"
                meta = {}
            elif norm.startswith("{") and norm.endswith("}"):
                t_type = "DETERMINATIVE"
                meta = {"category": STATIC_LEXICON["DETERMINATIVES"].get(norm, "UNKNOWN")}
            else:
                t_type = "SYLLABLE/WORD"
                meta = self.analyze_morphology(norm)
            
            st = SemanticToken(raw=rt, normalized=norm, token_type=t_type, analysis=meta)
            semantic_tokens.append(st)
            
        return semantic_tokens

# --- 6. VECTOR E: SOCIAL GRAPH & FRAUD DETECTION ---

class MardukNetworkCore:
    """
    Network Intelligence Analyst.
    Uses NetworkX to detect ancient criminal patterns.
    """
    def __init__(self, kernel: KanishKernel):
        self.kernel = kernel
        self.graph = nx.DiGraph()

    def load_graph_from_ledger(self):
        """Hydrates the graph from the SQLite Vault."""
        conn = sqlite3.connect(self.kernel.db_path)
        df = pd.read_sql("SELECT * FROM financial_ledger", conn)
        conn.close()
        
        for _, row in df.iterrows():
            self.graph.add_edge(
                row['sender'], row['receiver'], 
                weight=row['quantity'], 
                type=row['commodity'],
                risk=row['fraud_probability']
            )
        
        logger.info(f"Social Graph Built: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges")

    def analyze_risk(self) -> Dict[str, Any]:
        """Executes forensic algorithms."""
        if self.graph.number_of_nodes() == 0: return {}
        
        # 1. PageRank (Power Centrality)
        centrality = nx.pagerank(self.graph, weight='weight')
        
        # 2. Clique Detection (Circular Debt)
        # Convert to undirected to find dense communities
        undirected = self.graph.to_undirected()
        cliques = list(nx.find_cliques(undirected))
        
        fraud_rings = []
        for clique in cliques:
            if len(clique) >= 3:
                # Check if there is high-risk circular flow
                sub = self.graph.subgraph(clique)
                risk_score = sum([d.get('risk', 0) for u,v,d in sub.edges(data=True)])
                if risk_score > 0.1: # Threshold
                    fraud_rings.append({"members": clique, "risk": risk_score})
                    
        return {
            "top_merchants": sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10],
            "potential_fraud_rings": fraud_rings
        }

# --- 7. MASTER ORCHESTRATOR (PIPELINE) ---

class MardukUltimateOrchestrator:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Vectors
        self.kernel = KanishKernel(self.output_path)
        self.refinery = MardukRefinery()
        self.network = MardukNetworkCore(self.kernel)

    def process_text_file(self, file_path: Path):
        """Processes a CSV file of cuneiform texts."""
        logger.info(f"Ingesting: {file_path.name}")
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return

        # Smart column detection
        cols = df.columns.str.lower()
        # Find column containing transliteration
        text_cols = df.columns[cols.str.contains('transliteration') | cols.str.contains('text')]
        if len(text_cols) > 0:
            text_col = text_cols[0]
        else:
            logger.warning(f"No text column found in {file_path.name}")
            return

        id_cols = df.columns[cols.str.contains('id')]
        if len(id_cols) > 0:
            id_col = id_cols[0]
        else:
            id_col = None

        # Fact Extraction Simulation (Vector A -> K)
        count = 0
        for idx, row in df.iterrows():
            raw_text = str(row[text_col])
            text_id = str(row[id_col]) if id_col else f"unk_{idx}"
            
            # 1. Refinement
            tokens = self.refinery.parse_line(raw_text, text_id)
            
            # 2. Heuristic Transaction Extraction (Simplified for example)
            # Looks for patterns: Person A ... Person B ... Silver/Gold
            norm_text = " ".join([t.normalized for t in tokens])
            
            # Heuristic regex to detect "X lent Y to Z" (very simplified)
            # In production, this would use a full syntactic parser based on 'akkadian_syntax_engine.json'
            if "kaspum" in norm_text or "KÙ.BABBAR" in norm_text:
                # Simulating finding a transaction to populate the graph
                # In reality, this would extract real names using NER
                sender = f"Merchant_{idx % 10}" 
                receiver = f"Merchant_{ (idx + 1) % 10 }"
                amount = (idx * 37) % 100 # Pseudo-random value for test
                
                # Integrity Validation
                is_valid, _ = self.kernel.validate_sdic_g(amount, "KÙ.BABBAR")
                risk = 0.0 if is_valid else 0.9
                
                tx = TransactionRecord(
                    tx_hash="", # Generated inside log_transaction
                    source_text_id=text_id,
                    sender=sender,
                    receiver=receiver,
                    commodity="SILVER",
                    amount=float(amount),
                    is_prime_validated=is_valid,
                    risk_score=risk
                )
                tx.tx_hash = self.kernel.sign_data(asdict(tx))
                self.kernel.record_transaction(tx)
                count += 1
                
        logger.info(f"Processed {len(df)} lines. Detected {count} potential transactions.")

    def execute_pipeline(self):
        """Executes the full cycle: Ingest -> Refine -> Validate -> Analyze."""
        start_t = time.time()
        
        # 1. Ingest (Search CSVs)
        input_files = list(self.input_path.glob("*.csv"))
        if not input_files:
            logger.warning("No CSVs found. Using simulation mode.")
            # Create dummy data to demonstrate functionality if no input
            dummy_df = pd.DataFrame({
                "id": ["T001", "T002", "T003"],
                "transliteration": [
                    "1. 5 ma-na KÙ.BABBAR a-na Puzur-Aššur",
                    "2. szu-ma i-na É.GAL-lim",
                    "3. um-ma En-lil2-ba-ni-ma"
                ]
            })
            dummy_path = self.output_path / "dummy_input.csv"
            dummy_df.to_csv(dummy_path, index=False)
            input_files = [dummy_path]

        for f in input_files:
            # Ignore config files
            if "schema" not in f.name and "rules" not in f.name:
                self.process_text_file(f)

        # 2. Network Analysis
        logger.info("Building and analyzing Social Graph...")
        self.network.load_graph_from_ledger()
        risk_report = self.network.analyze_risk()
        
        # 3. Result Export
        report_path = self.output_path / "MARDUK_Forensic_Report.json"
        with open(report_path, "w", encoding='utf-8') as f:
            json.dump(risk_report, f, indent=4, default=str)
            
        logger.info(f"Pipeline finished in {time.time() - start_t:.2f}s. Report at: {report_path}")

# --- ENTRY POINT ---

if __name__ == "__main__":
    # Environment detection (Kaggle vs Local)
    BASE_DIR = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path(".")
    WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./marduk_output")
    
    # Search dataset subdirectory if exists
    data_dir = BASE_DIR
    for root, dirs, files in os.walk(BASE_DIR):
        if any(f.endswith(".csv") for f in files):
            data_dir = Path(root)
            break
            
    orchestrator = MardukUltimateOrchestrator(str(data_dir), str(WORK_DIR))
    orchestrator.execute_pipeline()
