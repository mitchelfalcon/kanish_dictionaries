import pickle
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from pathlib import Path

# --- EXTENDED SCHEMAS ---

@dataclass
class ForensicReport:
    """
    Summarizes the audit findings for a single tablet.
    """
    tablet_id: str
    swer_value: float
    fraud_risk: str
    identified_actors: List[str]
    chronology: str

# --- EXTENDED CORE ENGINE ---

class SumerianForensicExtension:
    """
    Extends the base pipeline to perform deep forensic and prosopographical analysis.
    
    Attributes:
        data_path: Path to persistent storage.
        swer_threshold: Maximum allowed variance in silver weights.
    """
    def __init__(self, data_path: Path, swer_threshold: float = 0.05):
        self.data_path = data_path
        self.swer_threshold = swer_threshold
        self.social_graph: Dict[str, List[str]] = {}
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Creates the persistence layer if it does not exist."""
        self.data_path.mkdir(parents=True, exist_ok=True)

    def process_ner_output(self, tablet_id: str, entities: List[Tuple[str, str]]) -> List[str]:
        """
        Maps NER entities (Person, City, Item) to the social graph.
        
        Args:
            tablet_id: The Gematria ID of the source document.
            entities: List of (text, tag) tuples from the NER model.
            
        Returns:
            List of unique Person IDs identified.
        """
        extracted_actors = []
        for text, tag in entities:
            if tag == "PERSON":
                actor_id = f"PN_{text.upper().replace(' ', '_')}"
                extracted_actors.append(actor_id)
                self._update_graph(actor_id, tablet_id)
        return extracted_actors

    def _update_graph(self, actor_id: str, tablet_id: str) -> None:
        """Maintains the O(1) adjacency list for the prosopography."""
        if actor_id not in self.social_graph:
            self.social_graph[actor_id] = []
        self.social_graph[actor_id].append(tablet_id)

    def calculate_transaction_integrity(self, expected: float, actual: float) -> Tuple[float, bool]:
        """
        Calculates SWER and determines if the transaction violates economic rules.
        """
        if expected == 0:
            return 0.0, False
            
        swer = abs(expected - actual) / expected
        is_anomaly = swer > self.swer_threshold
        return swer, is_anomaly

    def save_state(self) -> None:
        """Serializes the social graph and knowledge base to disk."""
        with open(self.data_path / "social_graph.pkl", "wb") as f:
            pickle.dump(self.social_graph, f)

    def load_state(self) -> None:
        """Deserializes the knowledge base for cross-session persistence."""
        graph_file = self.data_path / "social_graph.pkl"
        if graph_file.exists():
            with open(graph_file, "rb") as f:
                self.social_graph = pickle.load(f)

# --- INTEGRATION EXAMPLE ---

def execute_extended_audit():
    """
    Demonstrates the end-to-end integration of NLP tags and Forensic analysis.
    """
    storage = Path("/kaggle/working/forensic_data")
    ext = SumerianForensicExtension(storage)
    
    # Simulating output from the Bi-LSTM NER Model
    ner_results = [("Ashur-nada", "PERSON"), ("Kanish", "CITY"), ("10 shekels", "QUANTITY")]
    
    # 1. Prosopography
    actors = ext.process_ner_output("0xABC123", ner_results)
    
    # 2. Forensic SWER calculation
    swer, alert = ext.calculate_transaction_integrity(expected=10.0, actual=9.2)
    
    # 3. Persistence
    ext.save_state()
    
    print(f"Audit Result for {actors[0]}:")
    print(f"SWER: {swer:.4f} | Alert Triggered: {alert}")

if __name__ == "__main__":
    execute_extended_audit()
