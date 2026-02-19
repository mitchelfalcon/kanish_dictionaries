import pandas as pd
import numpy as np
import unicodedata
import re
import json
import os

class KanishVortexEngine:
    def __init__(self, data_dir="."):
        """
        Initializes the engine using the specific CSVs uploaded by the user.
        """
        self.data_dir = data_dir
        print("Initializing Kanish Vortex Engine with User Files...")
        
        # 1. LOAD CHARACTER MAPS & PARSING LOGIC
        # ---------------------------------------------------------
        # Load ATF Conventions (sz -> š)
        try:
            self.atf_conv = pd.read_csv(os.path.join(data_dir, "atf_conventions..csv"))
            self.char_map = dict(zip(self.atf_conv['ASCII_ATF'], self.atf_conv['Character']))
        except FileNotFoundError:
            self.char_map = {}
            print("Warning: atf_conventions..csv not found. Skipping normalization.")

        # Load Parsing Regex Logic
        try:
            self.parsing_logic = pd.read_csv(os.path.join(data_dir, "atf_parsing_logic.csv"))
        except FileNotFoundError:
            self.parsing_logic = pd.DataFrame()

        # 2. BUILD UNIFIED ROOT DICTIONARY (The "Heart")
        # ---------------------------------------------------------
        self.root_lookup = {}
        
        # Source A: Verbs
        try:
            verbs = pd.read_csv(os.path.join(data_dir, "akkadian_verbs.csv"))
            for _, row in verbs.iterrows():
                self.root_lookup[row['Word']] = {'meaning': row['Definition'], 'pos': 'V', 'source': 'verbs'}
        except: pass

        # Source B: Function Words
        try:
            funcs = pd.read_csv(os.path.join(data_dir, "akkadian_function_words.csv"))
            for _, row in funcs.iterrows():
                self.root_lookup[row['Lemma']] = {'meaning': row['Meaning'], 'pos': row['Type'], 'source': 'function'}
        except: pass

        # Source C: Categories/Commodities (Context & Kingdoms)
        try:
            cats = pd.read_csv(os.path.join(data_dir, "categories1.csv"))
            for _, row in cats.iterrows():
                # Store kingdom/context info
                self.root_lookup[row['Akkadian_Term']] = {
                    'meaning': row['English_Translation'], 
                    'pos': 'N', 
                    'kingdom': row['CATEGORY'],
                    'source': 'categories'
                }
        except: pass

        # 3. LOAD GRAMMAR ENGINE (Suffix Rules)
        # ---------------------------------------------------------
        try:
            self.syntax_df = pd.read_csv(os.path.join(data_dir, "akkadian_grammar_engine.csv"))
            # Sort suffixes by length (longest first) to avoid partial matches (e.g., -atum before -um)
            self.syntax_df['len'] = self.syntax_df['Suffix_Pattern'].str.len()
            self.syntax_df = self.syntax_df.sort_values(by='len', ascending=False)
        except:
            self.syntax_df = pd.DataFrame()

        # 4. LOAD SIGN LIST (Borger)
        # ---------------------------------------------------------
        try:
            # Handling potential bad lines in borger.csv
            self.borger = pd.read_csv(os.path.join(data_dir, "borger.csv"), on_bad_lines='skip', encoding='utf-8')
            # Mapping: Valores de Borger -> Firmar (Glyph)
            self.sign_lookup = dict(zip(self.borger['Valores de Borger'], self.borger['Firmar']))
        except:
            self.sign_lookup = {}

        print(f"Engine Ready. Loaded {len(self.root_lookup)} roots and {len(self.syntax_df)} grammar rules.")

    def normalize_transliteration(self, text):
        """
        Applies ATF conventions (sz -> š) and Unicode NFC normalization.
        """
        if not isinstance(text, str): return ""
        
        # 1. Apply ATF Conventions from CSV
        for ascii_code, unicode_char in self.char_map.items():
            if ascii_code in text:
                text = text.replace(ascii_code, unicode_char)
        
        # 2. NFC Normalization
        return unicodedata.normalize('NFC', text)

    def peel_morphology(self, token):
        """
        Hierarchical peeling using the User's uploaded Grammar Engine.
        """
        # Step 0: Check if token is a known root (Exact Match)
        if token in self.root_lookup:
            data = self.root_lookup[token]
            return f"{token}[{data['meaning']}]{data['pos']}"

        # Step 1: Logogram Check (Uppercase)
        if token.isupper():
            # If in dictionary (e.g., KÙ.BABBAR from categories1.csv?)
            # If not, treat as generic logogram
            clean_token = token.replace(".", "")
            if clean_token in self.root_lookup:
                 data = self.root_lookup[clean_token]
                 return f"{token.lower()}[{data['meaning']}]N"
            return f"{token}[LOG]N"

        # Step 2: Suffix Peeling (The "Onion" Algorithm)
        best_lemma = None
        
        # Iterate through grammar rules (longest suffixes first)
        for _, rule in self.syntax_df.iterrows():
            suffix = rule['Suffix_Pattern']
            pos_tag = rule['Category'].split('/')[0][0].upper() # Extract 'N' from 'Noun/Adjective'
            
            if token.endswith(suffix):
                # Peel!
                potential_root = token[:-len(suffix)]
                
                # Check if the "peeled" root exists in our dictionary
                if potential_root in self.root_lookup:
                    data = self.root_lookup[potential_root]
                    best_lemma = f"{potential_root}[{data['meaning']}]{data['pos']}"
                    return best_lemma # Found it!
        
        # Fallback for unknown morphology
        return f"{token}[UNK]X"

    def process_test_file(self, input_file, output_file):
        """
        Main Execution Pipeline for test.csv
        """
        print(f"Processing {input_file}...")
        df = pd.read_csv(input_file)
        results = []

        for idx, row in df.iterrows():
            raw_text = row.get('transliteration', '')
            # 1. Normalize
            clean_text = self.normalize_transliteration(raw_text)
            
            # 2. Tokenize (Simple split for now, could use Regex from parsing_logic)
            tokens = clean_text.split()
            lemmas = []
            
            for token in tokens:
                # 3. Trinity Imputation / Logic
                if "[...]" in token or "x" in token or "..." in token:
                    # Imputation logic would go here. For now, flag it.
                    lemmas.append(f"{token}[imputed]X")
                else:
                    # 4. Morphology Peeling
                    lemma = self.peel_morphology(token)
                    lemmas.append(lemma)
            
            results.append({
                "id": row['id'],
                "transliteration": clean_text,
                "lemma": " ".join(lemmas),
                "translation": "" # Placeholder for SVO reordering logic
            })

        # Export
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(output_file, index=False)
        print(f"Done! Saved certified submission to {output_file}")
        return submission_df

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Initialize Engine in current directory
    engine = KanishVortexEngine(data_dir=".")
    
    # Run Pipeline on the uploaded test.csv
    if os.path.exists("test.csv"):
        final_df = engine.process_test_file("test.csv", "submission_certified.csv")
        print("\n--- SAMPLE OUTPUT (Top 5 Rows) ---")
        print(final_df.head())
    else:
        print("Please ensure test.csv is in the directory.")
