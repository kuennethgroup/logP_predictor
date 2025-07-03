import os  # Importiert das os-Modul, das Funktionen zum Arbeiten mit dem Betriebssystem bereitstellt, z.B. für Umgebungsvariablen.

# Dieser Abschnitt sorgt dafür, dass die Konsolenausgabe aufgeräumt bleibt und keine störenden Warnungen oder unnötigen Log-Meldungen zu sehen sind.
# Das ist besonders nützlich bei paralleler Verarbeitung, beim Arbeiten mit großen Frameworks (wie TensorFlow) oder wenn viele Bibliotheken genutzt werden, die noch nicht alle auf die neueste Python-Version angepasst sind.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from psmiles import PolymerSmiles as PS
import pandas as pd
import itertools
from typing import Dict, Optional, List
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Crippen
from tqdm import tqdm
import logging

# Filename: Alle Log-Meldungen werden in die Datei polymer_errors.log geschrieben.
# level=logging.ERROR: Es werden nur Fehler (und schlimmere Ereignisse) gespeichert, keine Infos oder Warnungen.
# Zweck: Fehler im Code werden protokolliert und sind später nachvollziehbar.
logging.basicConfig(filename='polymer_errors.log', level=logging.ERROR)

# Monomer 1
monomer1 = PS("CC(C)NC(=O)C(-*)C-*").canonicalize

# Monomer 2 Liste
dict_monomer2 = []
# ! Zahlen ändern
max_sum = 100
max_x = 10
for x in range(1, max_x+1):
    chem_struct = "[*]CC([*])C(=O)NCCN" + "(CN" * (x-1) + "(CCN1CCCCC1)" + "C(C)O)" * (x-1) + "C(C)O"
    dict_monomer2.append({"x": x, "monomer2_psmiles": PS(chem_struct).canonicalize})

# Kombinationen generieren
def generate_combinations():
    for n, m, x in itertools.product(range(1, max_sum+1), range(1, max_sum+1), range(1, max_x+1)):
        if n + m <= max_sum:
            yield n, m, x

# Copolymere generieren
def generate_copolymer(n: int, m: int, x: int) -> Dict:
    # Die Funktion nimmt drei ganzzahlige Parameter und gibt ein Dictionary zurück (-> Dict).
    result = {
        # Legt ein Dictionary mit den wichtigsten Feldern an.
        # monomer1_psmiles: String-Repräsentation von Monomer 1. Die anderen Felder werden im Verlauf ergänzt.
        "n": n,
        "m": m,
        "x": x,
        "monomer1_psmiles": str(monomer1),
        "monomer2_psmiles": None,
        "copolymers_psmiles": None,
        "error": None
    }
    try:
        # Aus der Liste dict_monomer2 wird das passende Monomer 2 gewählt (x-1, weil Python-Listen bei 0 beginnen).
        monomer2 = dict_monomer2[x-1]["monomer2_psmiles"]
        result["monomer2_psmiles"] = str(monomer2)
        pattern = [0]*n + [1]*m
        copolymer = monomer1.linear_copolymer(monomer2, pattern=pattern)
        result["copolymers_psmiles"] = str(copolymer)
    except Exception as e:
        result["error"] = str(e)
        logging.error(f"n={n}, m={m}, x={x}: {e}")
    return result

# Hilfsfunktion: Endgruppen terminieren für logP
import re
def terminate_polymer_smiles(psmiles: str, endgroup: str = "[H]") -> str:
    # Ersetzt [*] oder * durch [H] (oder eine andere Endgruppe)
    return re.sub(r'\[\*\]|\*', endgroup, psmiles)

# logP der Copolymere ermitteln mit terminierter Endgruppe: Die Funktion nimmt einen SMILES-String, wandelt ihn in ein Molekül um und berechnet dessen logP-Wert.
def calculate_logp(psmiles: str) -> Optional[float]:
    # Die Funktion erwartet einen SMILES-String als Eingabe (smiles: str).
    # Sie gibt entweder einen Float (den logP-Wert) oder None zurück (Optional[float]).
    # Das ist eine Typangabe (Type Hint) aus dem Modul typing in Python.
    # Sie sagt aus, dass die Funktion entweder einen Wert vom Typ float oder den Wert None zurückgeben kann.
    try:
        # Die Funktion MolFromSmiles aus RDKit wandelt den SMILES-String in ein Molekül-Objekt um.
        # Falls der SMILES ungültig ist, gibt die Funktion None zurück.
        smiles_terminated = terminate_polymer_smiles(psmiles)
        mol = Chem.MolFromSmiles(smiles_terminated)
        if mol is None:
            return None
        # Berechnet den logP-Wert (Maß für die Fettlöslichkeit eines Moleküls) mit der RDKit-Funktion.
        # Rundet das Ergebnis auf drei Nachkommastellen.
        return round(Crippen.MolLogP(mol), 3)
    except Exception as e:
        # Falls irgendwo im try-Block ein Fehler auftritt (z.B. bei der Berechnung), wird dieser im Logfile festgehalten.
        logging.error(f"LogP error for {psmiles}: {e}")
        return None

# Fingerprints der Copolymere generieren:
# Sie nimmt einen Polymer-SMILES-String (psmiles, also eine spezielle chemische Notation für Polymere) als Eingabe.
# Sie versucht, daraus mit der Klasse PolymerSmiles (abgekürzt PS) ein Polymer-Objekt zu erzeugen.
# Dann berechnet sie den Fingerprint (eine Art numerische Repräsentation der Molekülstruktur) mit der Methode "polyBERT".
# Sie gibt diesen Fingerprint als Python-Liste von Floats zurück.
def safe_fingerprint(psmiles: str) -> Optional[List[float]]:
    # Die Funktion gibt entweder eine Liste von Floats (den Fingerprint) oder None zurück.
    try:
        polymer = PS(psmiles)
        fp = polymer.fingerprint("polyBERT")
        return fp.tolist()
    except Exception as e:
        logging.error(f"Fingerprint error for {psmiles}: {e}")
        return None

# Fingerprints der Homopolymere berechnen und speichern
fp_monomer1 = safe_fingerprint(str(monomer1))
fp_monomer2_list = [safe_fingerprint(str(d["monomer2_psmiles"])) for d in dict_monomer2]

# Funktion zur Copolymergenerierung, logP- und Fingerprintermittlung  
def process_combination(args):
    # Die Funktion erwartet ein Tupel oder eine Liste mit drei Werten (n, m, x).
    n, m, x = args
    # Es wird eine Hilfsfunktion aufgerufen, die aus diesen Parametern ein Copolymer erzeugt und die wichtigsten Informationen (z.B. die SMILES-Notation) in einem Dictionary ablegt.
    result = generate_copolymer(n, m, x)
    # Nur wenn ein valider Copolymer-SMILES-String erzeugt wurde (d.h. kein Fehler auftrat), werden weitere Analysen durchgeführt.
    if result["copolymers_psmiles"]:
        result["smiles_terminated"] = terminate_polymer_smiles(result["copolymers_psmiles"]) 
        # Die Funktion berechnet den logP-Wert (eine chemische Eigenschaft, die die Lipophilie beschreibt) für das erzeugte Copolymer und speichert ihn im Dictionary.
        result["copolymers_logP"] = calculate_logp(result["copolymers_psmiles"])
        # Fingerprintberechnung
        fp1 = fp_monomer1
        fp2 = fp_monomer2_list[x-1]
        result["fp1"] = fp1
        result["fp2"] = fp2
        if fp1 is not None and fp2 is not None:
            n_total = n + m
            # Gewichteter Mittelwert er Fingerprints
            fp_copolymer = [(n * v1 + m * v2) / n_total for v1, v2 in zip(fp1, fp2)]
            result["copolymers_fingerprints"] = fp_copolymer
        else:
            result["copolymers_fingerprints"] = None
    return result

def run_pipeline(max_workers=4, max_combinations=None):
    # Hier wird eine Liste aller gewünschten Kombinationen (z.B. von Parametern wie n, m, x) erzeugt.
    # generate_combinations() ist eine Generatorfunktion, die alle möglichen Kombinationen liefert.
    # Mit itertools.islice(..., max_combinations) wird die Anzahl der Kombinationen ggf. begrenzt (z.B. auf die ersten 1000).
    # Ergebnis: combinations ist eine Liste von Tupeln, z.B. [(1,1,1), (1,1,2), ...].
    combinations = list(itertools.islice(generate_combinations(), max_combinations))
    # Mit pool.imap_unordered(process_combination, combinations, chunksize=50) werden die Aufgaben (je eine Kombination) parallel an die Worker verteilt:
    # Jede Kombination wird an die Funktion process_combination übergeben.
    # Die Verarbeitung geschieht parallel auf mehreren CPU-Kernen.
    # Die Ergebnisse werden sofort zurückgegeben, sobald sie fertig sind – also in beliebiger Reihenfolge (nicht in der Reihenfolge der Eingabe).
    # Das ist effizient, weil schnelle Aufgaben nicht auf langsamere warten müssen.
    # Mit chunksize=50 werden jeweils 50 Aufgaben auf einmal an einen Worker geschickt, was die Effizienz bei großen Datenmengen steigert.
    # tqdm(...) sorgt für einen Fortschrittsbalken.
    with multiprocessing.Pool(max_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_combination, combinations, chunksize=50), total=len(combinations), desc="Processing"))
    # Aus allen Ergebnissen ohne Fehler (r["error"] is None) wird ein pandas DataFrame gebaut.
    df = pd.DataFrame([r for r in results if r["error"] is None])
    # Dann wird weiter gefiltert: Es bleiben nur Zeilen übrig, bei denen der Eintrag copolymers_fingerprints eine nicht-leere Liste oder ein nicht-leeres Tupel ist.
    df = df[df["copolymers_fingerprints"].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)].copy()
    # Die Funktion gibt das gefilterte DataFrame zurück.
    return df 

if __name__ == "__main__":
    # Mit if __name__ == "__main__": stellst du sicher, dass dieser Ablauf nur beim direkten Ausführen des Skripts passiert – und nicht, wenn das Skript als Modul in ein anderes Python-Programm importiert wird
    df = run_pipeline(max_workers=4, max_combinations=None)
    print(df[["x","n","m", "copolymers_psmiles", "smiles_terminated", "fp1", "fp2", "copolymers_fingerprints", "copolymers_logP"]].head())
    print("Sample combinations:")
    print(df.sample(10))
    print(f"Length of the dataframe: {len(df)}")

    df.to_parquet("./data_0604_100_10.parquet")

    print("Data export completed!")



