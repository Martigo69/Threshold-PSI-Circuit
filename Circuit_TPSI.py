"""
Consolidated Threshold PSI Circuit - Unified Implementation
Merge of bloom.py and Circuit_menu.py with menu enable/disable option
"""

import itertools
import random
import os
import json
import array
import time
import re
import gc
import sys
from datetime import datetime

import numpy as np
from probables import BloomFilter
from concrete import fhe
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - To enable and disable the interactive menu
# ============================================================================
USE_MENU = True 


# ============================================================================
# Realtime logging helpers (append mode, line-buffered)
# ============================================================================

_LOG_STREAM = None


def setup_realtime_logging(log_file_path: str = "party_sets/party_sets_runtime.log") -> str:
    """Route all stdout/stderr to a persistent log file in realtime (no console output)."""
    global _LOG_STREAM

    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    _LOG_STREAM = open(log_file_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = _LOG_STREAM
    sys.stderr = _LOG_STREAM
    print("\n" + "=" * 80)
    print(f"Session started at {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    print(f"Realtime log file: {log_file_path}")
    print("=" * 80)
    return log_file_path


def _format_seconds(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}s"


def _render_timing_table(title: str, rows: list) -> None:
    """Render an ASCII timing table for one or many runs."""
    if not rows:
        print(f"\n{title}: no rows")
        return

    headers = [
        "Run",
        "Dataset Source",
        "Preprocessing",
        "Circuit Creation + Optimization",
        "Plaintext Computation",
        "Encrypted Computation + Decryption",
    ]

    prepared_rows = []
    for row in rows:
        prepared_rows.append([
            str(row.get("run", "-")),
            str(row.get("dataset_source", "-")),
            _format_seconds(row.get("preprocessing_total_time_s")),
            _format_seconds(row.get("circuit_creation_time_s")),
            _format_seconds(row.get("plaintext_computation_time_s")),
            _format_seconds(row.get("encrypted_computation_and_decryption_time_s")),
        ])

    widths = [len(h) for h in headers]
    for r in prepared_rows:
        widths = [max(widths[i], len(r[i])) for i in range(len(headers))]

    def _line(char="-"):
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    print(f"\n{title}")
    print(_line("="))
    print("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(_line("-"))
    for r in prepared_rows:
        print("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(_line("="))


# ============================================================================
# ThresholdCircuit - Core PSI Implementation
# ============================================================================

class ThresholdCircuit:
    """
    Encapsulates the full threshold-circuit / Bloom-filter workflow for N parties
    with a threshold T (at least T parties must share an element for it to appear
    in the intersection Bloom filter).

    Parameters
    ----------
    num_parties : int
        Total number of parties (N).
    threshold : int
        Minimum number of parties that must share an IP for it to be detected (T).
    false_positive_rate : float
        Ignored at runtime; effective value is always 1/party_set_size.
    num_common_ips : int
        Number of IP addresses that will be seeded into exactly T parties.
    party_set_size : int
        Number of IPs in each party's full set.
    """

    PARTY_SETS_DIR = "party_sets"
    PARTY_SETS_META_FILE = "party_sets_meta.json"

    def __init__(
        self,
        num_parties: int,
        threshold: int,
        false_positive_rate: float = 0.0005,
        num_common_ips: int = 2,
        party_set_size: int = 10**4,
    ):
        self.num_parties = num_parties
        self.threshold = threshold
        self.num_common_ips = num_common_ips
        self.party_set_size = party_set_size

        self.requested_false_positive_rate = 1.0 / self.party_set_size
        computed_fpr = self.requested_false_positive_rate
       
        _template = None
        for _ in range(400):
            try:
                _template = BloomFilter(
                    est_elements=party_set_size,
                    false_positive_rate=computed_fpr,
                )
                break
            except Exception as exc:
                print(f"  [WARN] BloomFilter init failed with FPR={computed_fpr:.6f}: {exc}")
                print(f"  [Update] Doubling FPR: {computed_fpr:.6f} -> {computed_fpr * 2.0:.6f}")
                computed_fpr *= 2.0
        

        self.false_positive_rate = computed_fpr
        self.num_bloom_bits = _template.number_bits   
        self.num_hash_funcs = _template.number_hashes 
        self._bits_per_chunk = int(_template._bits_per_elm)
        self._bloom_typecode = _template.bloom.typecode
        self._ip_space = 254 ** 4
        self._ip_cursor = random.randrange(self._ip_space)

    def _effective_num_common_ips(self) -> int:
        """Return effective C represented by data shape for current N/T/S."""
        if self.threshold == 1:
            return self.num_parties * self.party_set_size
        return self.num_common_ips

    def _dataset_tag(self) -> str:
        return (
            f"n{self.num_parties}_t{self.threshold}_"
            f"c{self._effective_num_common_ips()}_s{self.party_set_size}"
        )

    def _dataset_dir(self) -> str:
        return os.path.join(self.PARTY_SETS_DIR, self._dataset_tag())

    # ------------------------------------------------------------------
    # Data layer  party set persistence
    # ------------------------------------------------------------------

    def _counter_to_ip(self, counter: int) -> str:
        """Map a counter to IPv4 octets in [1, 254] to avoid 0/255 edge octets."""
        base = 254
        o4 = (counter % base) + 1
        counter //= base
        o3 = (counter % base) + 1
        counter //= base
        o2 = (counter % base) + 1
        counter //= base
        o1 = (counter % base) + 1
        return f"{o1}.{o2}.{o3}.{o4}"

    def _random_unique_ip(self, used_ips: set) -> str:
        """Return a unique IPv4 address that does not appear in used_ips."""
        for _ in range(self._ip_space):
            ip = self._counter_to_ip(self._ip_cursor)
            self._ip_cursor = (self._ip_cursor + 1) % self._ip_space
            if ip not in used_ips:
                used_ips.add(ip)
                return ip
        raise RuntimeError("Exhausted IPv4 candidate space while requesting unique IPs.")

    def _write_party_sets(self, party_sets: list) -> None:
        """Persist party_sets to individual text files (one IP per line)."""
        dataset_dir = self._dataset_dir()
        os.makedirs(dataset_dir, exist_ok=True)

        for idx, party_set in enumerate(party_sets):
            file_path = os.path.join(dataset_dir, f"party_{idx + 1}.txt")
            with open(file_path, "w") as fh:
                fh.write("\n".join(party_set) + "\n")

    def _party_set_metadata_path(self) -> str:
        return os.path.join(self._dataset_dir(), self.PARTY_SETS_META_FILE)

    def _current_party_set_metadata(self) -> dict:
        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "false_positive_rate": self.false_positive_rate,
            "num_common_ips": self._effective_num_common_ips(),
            "party_set_size": self.party_set_size,
        }

    def _write_party_set_metadata(self) -> None:
        os.makedirs(self._dataset_dir(), exist_ok=True)
        with open(self._party_set_metadata_path(), "w") as fh:
            json.dump(self._current_party_set_metadata(), fh, indent=2)

    def _read_party_set_metadata(self):
        path = self._party_set_metadata_path()
        if not os.path.exists(path):
            return None
        with open(path, "r") as fh:
            return json.load(fh)

    def _party_set_file_paths(self) -> list:
        dataset_dir = self._dataset_dir()
        if not os.path.exists(dataset_dir):
            return []
        files = []
        for name in os.listdir(dataset_dir):
            if name.startswith("party_") and name.endswith(".txt"):
                files.append(os.path.join(dataset_dir, name))
        return sorted(files)

    def _count_non_empty_lines(self, file_path: str) -> int:
        with open(file_path, "r") as fh:
            return sum(1 for line in fh if line.strip())

    def _item_counts(self, party_sets: list) -> dict:
        counts = {}
        for party in party_sets:
            for item in set(party):
                counts[item] = counts.get(item, 0) + 1
        return counts

    def _is_party_set_cache_valid(self) -> bool:
        metadata = self._read_party_set_metadata()
        if metadata != self._current_party_set_metadata():
            return False

        expected_files = {
            os.path.join(self._dataset_dir(), f"party_{i + 1}.txt")
            for i in range(self.num_parties)
        }
        actual_files = set(self._party_set_file_paths())
        if actual_files != expected_files:
            return False

        ordered_files = sorted(actual_files)
        line_counts = [self._count_non_empty_lines(file_path) for file_path in ordered_files]
        if any(line_count != self.party_set_size for line_count in line_counts):
            return False

        return True

    def _clear_party_set_cache(self) -> None:
        dataset_dir = self._dataset_dir()
        if not os.path.exists(dataset_dir):
            return
        for name in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    def _read_party_sets(self):
        """
        Load party sets from disk if all N files exist.
        Returns a list of IP-address lists, or None if any file is missing.
        """
        expected_files = [
            os.path.join(self._dataset_dir(), f"party_{i + 1}.txt")
            for i in range(self.num_parties)
        ]
        if not all(os.path.exists(p) for p in expected_files):
            return None

        party_sets = []
        for file_path in expected_files:
            with open(file_path, "r") as fh:
                party_sets.append([line.strip() for line in fh if line.strip()])
        return party_sets

    def load_or_create_party_sets(self) -> list:
        """
        [OPERATION]: Load existing party IP sets from disk, or generate and persist new ones.
        
        [PURPOSE]: Provides a single entry point for obtaining party sets with caching. 
        Avoids regenerating datasets if they already exist with matching parameters.
        Enables reproducibility and efficient reuse of generated data across runs.
        
        Parameters
        ----------
        None. Uses instance attributes (num_parties, threshold, party_set_size, etc.).
        
        Returns
        -------
        list of list of str
            Party sets where each inner list contains IP addresses for one party.
            Outer list has length num_parties; each inner list has length party_set_size.
        
        Notes
        -----
        - Dataset existence is determined by checking the party_sets/<dataset_tag>/ directory.
        - If cache is valid (metadata matches current params and all N files exist), loads from disk.
        - If cache is invalid or missing, clears old files and generates new party sets.
        - Sets self._last_dataset_source to 'loaded' or 'created' for tracking.
        """
        # Quick cache check: if files already exist with correct metadata, load them
        is_available = self._is_party_set_cache_valid()
        self._last_dataset_source = "loaded" if is_available else "created"

        print(
            f"[INPUT] Dataset {self._dataset_tag()} : "
            f"{'AVAILABLE' if is_available else 'NOT AVAILABLE'}"
        )

        if is_available:
            print(f"[INPUT] Loading existing input files for {self._dataset_tag()}")
            existing = self._read_party_sets()
            if existing is not None:
                return existing

        # Cache doesn't exist or is invalid: generate fresh based on parameters
        print(f"[INPUT] Creating new input files for {self._dataset_tag()}")
        self._clear_party_set_cache()
        return self.generate_party_sets()


    def build_bloom_filters_from_party_sets(self, party_sets: list) -> list:
        """
        [OPERATION]: Build Bloom filters from in-memory party IP sets (batch operation).
        
        [PURPOSE]: Converts all N party sets into their Bloom filter bit representations 
        in a single batch. Convenience wrapper for repeated calls to build_bloom_filter().
        
        Parameters
        ----------
        party_sets : list of list of str
            Party IP sets where outer list has length num_parties and each inner list 
            contains IP addresses for one party.
        
        Returns
        -------
        list of list of int
            Bloom filter bit arrays (one per party). Each inner list has length num_bloom_bits 
            with elements 0 or 1.
        
        Notes
        -----
        - Calls build_bloom_filter() for each party sequentially.
        - No caching; rebuilds filters even if they already exist.
        - Prints progress indicating the dataset ID being processed.
        ""

    def generate_party_sets(self) -> list:
        """
        [OPERATION]: Generate N party IP sets with controlled intersection and write to disk.
        
        [PURPOSE]: Creates synthetic datasets for threshold-PSI testing where a known 
        subset of IPs appears in exactly T parties, and remaining slots are filled with 
        unique IPs. This enables controlled measurement of intersection accuracy.
        
        Parameters
        ----------
        None. Uses instance attributes (num_parties, threshold, party_set_size, num_common_ips).
        
        Returns
        -------
        list of list of str
            Generated party IP sets: outer list has length num_parties; 
            each inner list contains party_set_size unique IPv4 addresses.
        
        Notes
        -----
        - Common IPs: Generates num_common_ips addresses and seeds them into 
          threshold randomly-selected parties.
        - Unique IPs: Fills remaining slots in each party with unique IPs 
          from the IPv4 range [1.1.1.1, 254.254.254.254] (avoiding 0/255).
        - Persistence: Writes each party set to party_sets/<dataset_tag>/party_N.txt 
          and metadata to party_sets_meta.json.
        ""
        party_sets = [set() for _ in range(self.num_parties)]
        used_ips: set = set()

        common_ips = [
            self._random_unique_ip(used_ips) for _ in range(self.num_common_ips)
        ]
        chosen_parties = random.sample(range(self.num_parties), self.threshold)
        for ip in common_ips:
            for party_idx in chosen_parties:
                party_sets[party_idx].add(ip)

        # Fill remaining slots with unique IPs sequentially.

        for party_idx, party_set in enumerate(party_sets, start=1):
            while len(party_set) < self.party_set_size:
                party_set.add(self._random_unique_ip(used_ips))

        party_lists = [list(s) for s in party_sets]
        self._write_party_sets(party_lists)
        self._write_party_set_metadata()
        return party_lists

    def run_detailed_pipeline(self) -> dict:
        """
        [OPERATION]: Execute the complete end-to-end threshold-PSI workflow: data load, 
        Bloom filter construction, circuit build, plaintext evaluation, encryption, 
        encrypted evaluation, and decryption.
        
        [PURPOSE]: Provides a single orchestration point that runs all pipeline stages, 
        measures execution time at each stage, and returns comprehensive results and metrics. 
        Used for benchmarking and validating correctness.
        
        Parameters
        ----------
        None. Uses instance configuration (num_parties, threshold, etc.).
        
        Returns
        -------
        dict
            Comprehensive pipeline result containing:
            - 'party_sets': list of loaded/generated IP sets.
            - 'bloom_filters': list of per-party Bloom filter bit arrays.
            - 'canonical_circuit': full circuit before minimization (e.g., AB!C + A!BC + ABC).
            - 'optimized_circuit': minimized SOP form (e.g., AB + AC + BC).
            - 'plaintext_result': bit array from plaintext circuit evaluation.
            - 'decrypted_result': bit array from encrypted evaluation (should match plaintext).
            - 'encrypted_matches_plaintext': bool indicating correctness check result.
            - 'timings': dict with keys load_dataset_time_s, build_bloom_filters_time_s, 
              input_encryption_time_s, encrypted_computation_time_s, decryption_time_s, 
              plaintext_computation_time_s, encrypted_computation_and_decryption_time_s, etc.
            - 'dataset_source': 'loaded' or 'created' (from _last_dataset_source).
        
        Notes
        -----
        - All timing measurements use time.perf_counter() for precision.
        - Plaintext result is the ground truth for correctness validation.
        - Encrypted computation time includes per-bit encryption, evaluation, and aggregate overhead.
        ""
        timings = {
            "load_dataset_time_s": None,
            "build_bloom_filters_time_s": None,
            "input_encryption_time_s": None,
            "preprocessing_total_time_s": None,
            "circuit_creation_time_s": None,
            "plaintext_computation_time_s": None,
            "encrypted_computation_and_decryption_time_s": None,
            "encrypted_computation_time_s": None,
            "decryption_time_s": None,
        }

        load_start = time.perf_counter()
        party_sets = self.load_or_create_party_sets()
        timings["load_dataset_time_s"] = time.perf_counter() - load_start

        bloom_start = time.perf_counter()
        bloom_filters = self.build_bloom_filters_from_party_sets(party_sets)
        timings["build_bloom_filters_time_s"] = time.perf_counter() - bloom_start

        circuit_start = time.perf_counter()
        canonical_circuit = self.build_canonical_circuit()
        optimized_circuit = self.optimize_circuit(canonical_circuit)
        timings["circuit_creation_time_s"] = time.perf_counter() - circuit_start

        plain_start = time.perf_counter()
        plaintext_result = self.evaluate_plaintext_circuit(optimized_circuit, bloom_filters)
        timings["plaintext_computation_time_s"] = time.perf_counter() - plain_start

        decrypted_result = None
        encrypt_inputs_elapsed = 0.0
        encrypted_eval_elapsed = None
        decryption_elapsed = None
        encrypted_computation_plus_decryption_elapsed = None

        enc_prep_start = time.perf_counter()
        encrypted_bloom_filters = [
            self.encrypt_bloom_filter(party_bloom_filter)
            for party_bloom_filter in bloom_filters
        ]
        encrypt_inputs_elapsed = time.perf_counter() - enc_prep_start

        enc_eval_start = time.perf_counter()
        encrypted_result = self.evaluate_encrypted_circuit(
            optimized_circuit,
            encrypted_bloom_filters,
        )
        encrypted_eval_elapsed = time.perf_counter() - enc_eval_start

        decrypt_start = time.perf_counter()
        decrypted_result = self.decrypt_bloom_filter(encrypted_result)
        decryption_elapsed = time.perf_counter() - decrypt_start
        encrypted_computation_plus_decryption_elapsed = encrypted_eval_elapsed + decryption_elapsed
        encrypted_matches_plaintext = decrypted_result == plaintext_result
        print(
            "[CHECK] Encrypted result matches plaintext: "
            f"{'PASS' if encrypted_matches_plaintext else 'FAIL'}"
        )

        timings["input_encryption_time_s"] = encrypt_inputs_elapsed
        timings["preprocessing_total_time_s"] = (
            timings["load_dataset_time_s"]
            + timings["build_bloom_filters_time_s"]
            + timings["input_encryption_time_s"]
        )
        timings["encrypted_computation_time_s"] = encrypted_eval_elapsed
        timings["decryption_time_s"] = decryption_elapsed
        timings["encrypted_computation_and_decryption_time_s"] = encrypted_computation_plus_decryption_elapsed

        return {
            "dataset_source": getattr(self, "_last_dataset_source", "unknown"),
            "party_sets": party_sets,
            "bloom_filters": bloom_filters,
            "canonical_circuit": canonical_circuit,
            "optimized_circuit": optimized_circuit,
            "plaintext_result": plaintext_result,
            "decrypted_result": decrypted_result,
            "encrypted_matches_plaintext": encrypted_matches_plaintext,
            "timings": timings,
        }

    # ------------------------------------------------------------------
    # Bloom filter layer
    # ------------------------------------------------------------------

    def build_bloom_filter(self, party_set: list) -> list:
        """
        [OPERATION]: Construct a Bloom filter from a party's IP set and return 
        its unpacked bit array representation.
        
        [PURPOSE]: Converts a raw party IP set into a flat bit vector suitable for 
        both plaintext and FHE encrypted circuit evaluation. Unpacking allows 
        per-bit operations on encrypted data.
        
        Parameters
        ----------
        party_set : list of str
            IPv4 addresses belonging to one party (e.g., ['1.2.3.4', '5.6.7.8', ...]).
        
        Returns
        -------
        list of int
            Flattened Bloom filter bit array of length num_bloom_bits, 
            where each element is 0 or 1.
        
        Notes
        -----
        - Internally uses pyprobables.BloomFilter with false_positive_rate 
          and party_set_size parameters from instance.
        - pyprobables stores bits as packed 32-bit integers; this function unpacks 
          them into individual bits for FHE compatibility.
        - Returned list is truncated to num_bloom_bits (first M bits only).
        ""
        bloom_filter = BloomFilter(
            est_elements=self.party_set_size,
            false_positive_rate=self.false_positive_rate,
        )
        for ip in party_set:
            bloom_filter.add(ip)
        
        result_bits = []
        for chunk in bloom_filter.bloom:
            for bit_pos in range(self._bits_per_chunk):
                result_bits.append((chunk >> bit_pos) & 1)
        return result_bits[: self.num_bloom_bits]

    def _reconstruct_bloom_filter_from_bits(self, bloom_filter_bits: list) -> BloomFilter:
        """Rebuild a pyprobables BloomFilter object from a flat bit list."""
        helper = BloomFilter(
            est_elements=self.party_set_size,
            false_positive_rate=self.false_positive_rate,
        )

        chunks = []
        for chunk_idx in range(helper.bloom_length):
            chunk_val = 0
            base = chunk_idx * self._bits_per_chunk
            for bit_pos in range(self._bits_per_chunk):
                bit_idx = base + bit_pos
                if bit_idx < len(bloom_filter_bits) and bloom_filter_bits[bit_idx] == 1:
                    chunk_val |= (1 << bit_pos)
            chunks.append(chunk_val)

        helper._bloom = array.array(self._bloom_typecode, chunks)
        return helper

    # ------------------------------------------------------------------
    # Circuit layer  build, optimise, evaluate plaintext
    # ------------------------------------------------------------------

    def build_canonical_circuit(self) -> str:
        """
        [OPERATION]: Generate the full canonical SOP circuit including all minterms 
        where at least T of N parties are active (positive literals).
        
        [PURPOSE]: Constructs the unoptimized boolean logic that encodes the threshold 
        condition. Serves as input to optimize_circuit() for minimization.
        
        Parameters
        ----------
        None. Uses instance attributes (num_parties, threshold).
        
        Returns
        -------
        str
            Canonical SOP string with all minterms, e.g., 'AB!C + A!BC + !ABC + ABC' 
            for N=3, T=2. Variables are always in alphabetical order (A, B, C, ...).
        
        Notes
        -----
        - Generates C(N, k) minterms for each k >= T (number of active parties).
        - Each minterm is a conjunction (AND) of all N variables, with negation 
          for inactive parties (e.g., A AND NOT C).
        - Result is verbose;  optimize_circuit() reduces it to minimal form.
        - Example: N=3, T=2 has C(3,2)=3 + C(3,3)=1 = 4 minterms.
        ""

        party_labels = [chr(ord("A") + i) for i in range(self.num_parties)]
        terms = []

        for num_active in range(self.threshold, self.num_parties + 1):
            for active_subset in itertools.combinations(party_labels, num_active):
                active_set = set(active_subset)
                literals = [
                    label if label in active_set else f"!{label}"
                    for label in party_labels
                ]
                terms.append("".join(literals))

        return " + ".join(terms)

    def optimize_circuit(self, circuit: str) -> str:
        """
        [OPERATION]: Minimize a canonical threshold circuit to minimal SOP form.
        
        [PURPOSE]: Reduces circuit complexity by exploiting the symmetric structure of 
        threshold functions. Removes redundant terms, enabling faster FHE evaluation.
        
        Parameters
        ----------
        circuit : str
            Canonical SOP circuit from build_canonical_circuit() 
            (e.g., 'AB!C + A!BC + !ABC + ABC').
        
        Returns
        -------
        str
            Optimized SOP with only positive literals (e.g., 'AB + AC + BC'). 
            For T-of-N threshold, result contains C(N, T) terms (all combinations of 
            exactly T positive variables).
        
        Notes
        -----
        - Threshold functions are symmetric; minimal SOP is always C(N, T) positive-literal terms.
        - Example: N=3, T=2 canonical 'AB!C + A!BC + !ABC + ABC' 
          reduces to 'AB + AC + BC' (C(3,2) = 3 terms).
        - All negations are removed in the optimized form.
        - Parsing is robust: splits on '+' and handles whitespace.
        ""
        terms = [t.strip() for t in circuit.split("+") if t.strip()]
        if not terms:
            return ""

        variables = sorted({circuit_var for term in terms for circuit_var in term if circuit_var.isalpha()})

        threshold = min(
            sum(
                1
                for pos, circuit_var in enumerate(term)
                if circuit_var.isalpha() and (pos == 0 or term[pos - 1] != "!")
            )
            for term in terms
        )

        optimized_terms = [
            "".join(combo) for combo in itertools.combinations(variables, threshold)
        ]
        return " + ".join(optimized_terms)

    def evaluate_plaintext_circuit(self, circuit: str, bloom_filters: list) -> list:
        """
        [OPERATION]: Evaluate optimized SOP circuit logic over plaintext 
        (unencrypted) Bloom filter bit arrays.
        
        [PURPOSE]: Computes the ground-truth intersection result. Used for correctness 
        validation by comparing against encrypted evaluation results and for rapid 
        prototyping without FHE overhead.
        
        Parameters
        ----------
        circuit : str
            Optimized SOP string with only positive literals (e.g., 'AB + AC + BC').
        bloom_filters : list of list of int
            Per-party Bloom filter bit arrays from build_bloom_filter(). 
            Index 0 = Party A, index 1 = Party B, etc. (alphabetical order).
        
        Returns
        -------
        list of int
            Intersection result bit array of length num_bloom_bits. 
            Each element is 0 or 1 indicating whether that bit satisfies the threshold.
        
        Notes
        -----
        - Parses circuit string by splitting on '+' to extract AND terms.
        - For each term (e.g., 'ABC'), ANDs the bits from indexed parties; 
          results are ORed together per bit position.
        - Circuit variables are case-insensitive; 'A' = Party 0, 'B' = Party 1, etc.
        - This is mathematically identical to encrypted evaluation but runs in plaintext.
        ""
        terms = [t.strip() for t in circuit.split("+") if t.strip()]
        if not terms:
            return [0] * self.num_bloom_bits

        bit_count = len(bloom_filters[0]) if bloom_filters else self.num_bloom_bits
        results = []
        for bit_idx in range(bit_count):
            output_bit = 0
            for term in terms:
                term_bit = 1
                for circuit_var in term:
                    if circuit_var.isalpha():
                        party_idx = ord(circuit_var.upper()) - ord("A")
                        term_bit &= bloom_filters[party_idx][bit_idx]
                output_bit |= term_bit
            results.append(output_bit)

        return results

    def benchmark_computation_times_from_party_sets(self) -> dict:
        """
        [OPERATION]: Run full pipeline and return timing breakdown across all stages.
        
        [PURPOSE]: Measures end-to-end and per-stage computation costs using real party sets. 
        Essential for scalability analysis, performance comparison, and FHE overhead quantification.
        
        Parameters
        ----------
        None. Uses instance configuration and run_detailed_pipeline().
        
        Returns
        -------
        dict
            Comprehensive metrics including:
            - Timing keys: 'load_dataset_time_s', 'build_bloom_filters_time_s', 
              'input_encryption_time_s', 'encrypted_computation_time_s', 
              'decryption_time_s', 'plaintext_computation_time_s', 
              'encrypted_computation_and_decryption_time_s', 'circuit_creation_time_s', 
              'preprocessing_total_time_s'.
            - Parameters: 'num_parties', 'threshold', 'num_common_ips', 'party_set_size', 
              'num_bloom_bits'.
            - Validation: 'encrypted_matches_plaintext' (bool).
            - Context: 'dataset_source' ('loaded' or 'created').
        
        Notes
        -----
        - All timings are in seconds (float), measured with time.perf_counter().
        - 'preprocessing_total_time_s' = sum(load + build_bloom_filters + input_encryption).
        - 'encrypted_computation_and_decryption_time_s' = encrypted_computation + decryption.
        - encrypted_matches_plaintext should be True for correct implementation.
        ""
        details = self.run_detailed_pipeline()
        timings = details["timings"]

        input_encryption_elapsed = timings["input_encryption_time_s"]
        encrypted_computation_elapsed = timings["encrypted_computation_time_s"]
        decryption_elapsed = timings["decryption_time_s"]
        encrypted_compute_plus_decrypt_elapsed = timings["encrypted_computation_and_decryption_time_s"]
        prep_elapsed = timings["preprocessing_total_time_s"]
        plain_elapsed = timings["plaintext_computation_time_s"]
        encrypted_match = details["encrypted_matches_plaintext"]

        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "num_common_ips": self.num_common_ips,
            "party_set_size": self.party_set_size,
            "num_bloom_bits": self.num_bloom_bits,
            "preprocessing_total_time_s": prep_elapsed,
            "plaintext_computation_time_s": plain_elapsed,
            "encrypted_computation_and_decryption_time_s": encrypted_compute_plus_decrypt_elapsed,
            "input_encryption_time_s": input_encryption_elapsed,
            "encrypted_computation_time_s": encrypted_computation_elapsed,
            "decryption_time_s": decryption_elapsed,
            "encrypted_matches_plaintext": encrypted_match,
            "dataset_source": details["dataset_source"],
            "circuit_creation_time_s": timings["circuit_creation_time_s"],
            "load_dataset_time_s": timings["load_dataset_time_s"],
            "build_bloom_filters_time_s": timings["build_bloom_filters_time_s"],
        }

    # ------------------------------------------------------------------
    # Concrete-python FHE layer  compile, encrypt, decrypt, evaluate
    # ------------------------------------------------------------------

    def _parse_positive_sop_terms(self, circuit: str) -> list:
        """
        [OPERATION]: Parse an optimized SOP circuit string into lists of party indices.
        
        [PURPOSE]: Converts human-readable circuit string (e.g., 'AB + AC + BC') 
        into machine-friendly format for FHE compilation. Enables concrete-python 
        to build the threshold logic function.
        
        Parameters
        ----------
        circuit : str
            Optimized SOP string with only positive literals (e.g., 'AB + AC + BC').
        
        Returns
        -------
        list of list of int
            Parsed terms where each inner list contains party indices (0-based).
            Example: 'AB + AC + BC' -> [[0, 1], [0, 2], [1, 2]] (for parties A=0, B=1, C=2).
        
        Notes
        -----
        - Splits circuit on '+' to extract individual AND terms.
        - For each term, maps characters (A, B, C, ...) to 0-based indices.
        - Output format is ready for _build_single_bit_threshold_compiler().
        - Assumes circuit contains only uppercase letters (A-Z) as party labels.
        ""

    def _build_single_bit_threshold_compiler(self, parsed_terms: list):
        """
        [OPERATION]: Create a concrete-python compiler function that evaluates 
        threshold logic for a single Bloom filter bit position.
        
        [PURPOSE]: Builds the FHE circuit template that will be compiled and keygen'd 
        once, then reused for per-bit encrypted evaluation. The returned compiler 
        is decorated with @fhe.compiler to indicate encrypted inputs/outputs.
        
        Parameters
        ----------
        parsed_terms : list of list of int
            Party index lists from _parse_positive_sop_terms(). 
            Example: [[0, 1], [0, 2], [1, 2]] means (party0 AND party1) OR (party0 AND party2) OR ...
        
        Returns
        -------
        A concrete-python FHE compiler object (decorated function)
            The compiler can be called with compiler.compile(inputset) 
            to generate a compiled circuit, then compiled.keygen() to generate keys.
        
        Notes
        -----
        - The returned compiler is a function called bit_threshold(bits) that takes 
          an encrypted array of party bits and returns one encrypted output bit.
        - Logic: for each term, ANDs the bits from specified parties; ORs all results.
        - Encryption is indicated by @fhe.compiler decorator and parameter {'bits': 'encrypted'}.
        - The function uses arithmetic operations (*, +, -) to implement AND/OR 
          on encrypted data (no branching allowed in FHE).
        ""

        @fhe.compiler({"bits": "encrypted"})
        def bit_threshold(bits):
            output = 0
            for term in parsed_terms:
                and_val = 1
                for idx in term:
                    and_val = and_val * bits[idx]
                output = output + and_val - (output * and_val)
            return output

        return bit_threshold

    def _init_fhe_context(self, circuit: str) -> None:
        """
        [OPERATION]: Lazily initialize and cache an FHE context (compiled circuit and keys) 
        for a specific optimized SOP circuit.
        
        [PURPOSE]: Performs expensive one-time compilation and keygen operations per unique 
        circuit. Caches results to avoid redundant compilation. Called automatically by 
        evaluate_encrypted_circuit() before evaluation.
        
        Parameters
        ----------
        circuit : str
            Optimized SOP circuit string (e.g., 'AB + AC + BC'). Used as cache key.
        
        Returns
        -------
        None. Side effects: Populates self._compiled_circuits[circuit] with compiled FHE context.
        
        Notes
        -----
        - On first call per circuit: parses circuit, builds compiler, compiles with binary inputset, 
          and generates FHE keys.
        - On subsequent calls: returns immediately (cache hit).
        - Compilation time is significant (can be seconds). Caching prevents redundant work.
        - Inputset: [array([0]*num_parties), array([1]*num_parties)] covers all binary boundary cases.
        - Raises RuntimeError if concrete-python is not installed or keygen fails.
        - Sets self._active_compiled_circuit for use by evaluate_encrypted_circuit() and decrypt_bloom_filter().
        ""
        if not hasattr(self, "_compiled_circuits"):
            self._compiled_circuits = {}

        if circuit in self._compiled_circuits:
            return

        try:
            parsed_terms = self._parse_positive_sop_terms(circuit)
            compiler = self._build_single_bit_threshold_compiler(parsed_terms)
            print(f"[FHE] Parsed {len(parsed_terms)} positive SOP terms for encrypted evaluation.")

            # Two boundary vectors are sufficient: inputs are strictly binary [0,1].
            inputset = [
                np.array([0] * self.num_parties, dtype=np.int64),
                np.array([1] * self.num_parties, dtype=np.int64),
            ]
            print("[FHE] Compiling Concrete circuit...")
            compiled = compiler.compile(inputset)
            print("[FHE] Generating FHE keys...")
            compiled.keygen()
            self._compiled_circuits[circuit] = compiled
            print("[FHE] Concrete circuit context is ready.")
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialise concrete-python FHE circuit. "
                "Ensure concrete-python is installed and configured.\n"
                f"Underlying error: {exc}"
            ) from exc

    def encrypt_bloom_filter(self, bloom_filter_bits: list) -> np.ndarray:
        """
        [OPERATION]: Convert Bloom filter bit array to numpy array for FHE encryption.
        
        [PURPOSE]: Prepares plaintext bits in the dense integer format required by 
        concrete-python's encrypt() method. A lightweight conversion that enables 
        downstream encrypted evaluation.
        
        Parameters
        ----------
        bloom_filter_bits : list of int
            Plaintext Bloom filter bit array from build_bloom_filter(). 
            Elements must be 0 or 1.
        
        Returns
        -------
        np.ndarray (dtype=int64)
            Dense numpy array representation of bit array, ready for 
            concrete-python encrypt() calls.
        
        Notes
        -----
        - This is a lightweight wrapper; no cryptographic operation occurs here.
        - int64 dtype is required by concrete-python FHE API.
        - Array is 1-D with length num_bloom_bits.
        ""
        return np.array(bloom_filter_bits, dtype=np.int64)

    def decrypt_bloom_filter(self, ciphertext: list) -> list:
        """
        [OPERATION]: Decrypt a list of per-bit FHE ciphertexts to recover plaintext bits.
        
        [PURPOSE]: Recovers the final intersection result from encrypted computation. 
        Enables validation against plaintext evaluation and extraction of candidate IPs.
        
        Parameters
        ----------
        ciphertext : list of encrypted values
            Per-bit ciphertexts from evaluate_encrypted_circuit(). 
            Length = num_bloom_bits. Each element is a concrete-python ciphertext.
        
        Returns
        -------
        list of int
            Decrypted bit array of length num_bloom_bits, where each element is 0 or 1.
        
        Notes
        -----
        - Requires _active_compiled_circuit to be set (set by evaluate_encrypted_circuit()).
        - Calls compiled.decrypt(ciphertext[i]) for each bit sequentially.
        - Decryption is sequential (no parallelization) to avoid overhead.
        - Result should match plaintext_result from evaluate_plaintext_circuit() if 
          encrypted evaluation was correct.
        ""
        if not hasattr(self, "_active_compiled_circuit"):
            raise RuntimeError("No active concrete-python circuit found for decryption.")

        compiled = self._active_compiled_circuit
        decrypted = [0] * len(ciphertext)

        print(f"[FHE] Decrypting result bits ({len(ciphertext)} bits)...")

        for idx, enc_bit in enumerate(ciphertext):
            decrypted[idx] = int(compiled.decrypt(enc_bit))

        print("[FHE] Decryption complete")
        return decrypted

    def evaluate_encrypted_circuit(self, circuit: str, encrypted_bloom_filters: list) -> list:
        """
        [OPERATION]: Evaluate optimized SOP circuit over encrypted Bloom filter bits 
        using concrete-python FHE, performing one bit position at a time.
        
        [PURPOSE]: Computes intersection result while keeping all data encrypted end-to-end. 
        Enables secure multi-party intersection without revealing intermediate values.
        
        Parameters
        ----------
        circuit : str
            Optimized SOP circuit with only positive literals (e.g., 'AB + AC + BC').
        encrypted_bloom_filters : list of np.ndarray
            Per-party encrypted Bloom filter bit arrays from encrypt_bloom_filter(). 
            Index 0 = Party A, 1 = Party B, etc.
        
        Returns
        -------
        list of encrypted values
            Ciphertext per-bit results (one ciphertext per bit position). 
            Length = num_bloom_bits. Each element is a concrete-python ciphertext object.
        
        Notes
        -----
        - Lazily initializes FHE context (compilation and keygen) on first call per circuit via _init_fhe_context().
        - Inner loop: for each bit position, assembles per-bit inputs as encrypted np.array, 
          calls compiled.encrypt() and compiled.run() to evaluate.
        - Uses the per-bit threshold compiler built by _build_single_bit_threshold_compiler().
        - Result ciphertexts are later decrypted by decrypt_bloom_filter().
        ""
        self._init_fhe_context(circuit)
        compiled = self._compiled_circuits[circuit]
        self._active_compiled_circuit = compiled

        if not encrypted_bloom_filters:
            return []

        bit_count = len(encrypted_bloom_filters[0])
        if bit_count == 0:
            return []

        results = [None] * bit_count

        print(f"[FHE] Encrypting and evaluating Bloom bits ({bit_count} bits)...")

        for bit_idx in range(bit_count):
            bit_inputs = np.array(
                [int(encrypted_bloom_filters[p][bit_idx]) for p in range(self.num_parties)],
                dtype=np.int64,
            )
            results[bit_idx] = compiled.run(compiled.encrypt(bit_inputs))

        print("[FHE] Encrypted evaluation complete")
        return results

    def extract_candidates_from_intersection_bloom(self, intersection_bits: list, party_sets: list) -> list:
        """
        [OPERATION]: Recover candidate elements from decrypted intersection Bloom filter 
        by checking all elements in the union against the final bit array.
        
        [PURPOSE]: Converts the encrypted intersection Bloom filter bits (binary result) 
        back into candidate IP addresses. Used for validation and result extraction.
        
        Parameters
        ----------
        intersection_bits : list of int
            Final decrypted Bloom filter bit array from decrypt_bloom_filter(). 
            Length = num_bloom_bits; elements are 0 or 1.
        party_sets : list of list of str
            Original party IP sets (needed to recover the universe of candidates).
        
        Returns
        -------
        list of str
            Candidate IP addresses that hash to 'set' bits in intersection_bits. 
            May include false positives because Bloom filters are probabilistic.
        
        Notes
        -----
        - Universe is the sorted union of all IPs from all party sets.
        - For each IP, reconstructs the Bloom filter object and checks if IP matches.
        - Result can include false positives (Bloom filter property) but should not 
          omit true positives.
        - Extracted candidates should be close to exact_threshold_intersection() 
          but may have extras due to false positives.
        ""
        helper = self._reconstruct_bloom_filter_from_bits(intersection_bits)

        universe = sorted({ip for party in party_sets for ip in party})
        candidates = []
        for item in universe:
            if helper.check(item):
                candidates.append(item)
        return candidates

    def exact_threshold_intersection(self, party_sets: list) -> list:
        """
        [OPERATION]: Compute exact threshold intersection from plaintext party sets 
        without using Bloom filters (exact set operation).
        
        [PURPOSE]: Provides ground-truth result (no false positives like Bloom filters). 
        Used for validating Bloom-based intersection results and measuring false positive rate.
        
        Parameters
        ----------
        party_sets : list of list of str
            Party IP sets where each inner list contains IPs for one party.
        
        Returns
        -------
        list of str
            Sorted list of IPs that appear in at least T parties (exact result, 
            no false positives).
        
        Notes
        -----
        - Counts IP occurrences across all parties (count = number of parties containing IP).
        - Filters IPs where count >= threshold.
        - Result is the mathematical set intersection at threshold level.
        - Expensive operation (O(total_ips * num_parties)) but guarantees exactness.
        - Should match extract_candidates_from_intersection_bloom() 
          minus false positives from Bloom filter.
        ""
        counts = {}
        for party in party_sets:
            for item in set(party):
                counts[item] = counts.get(item, 0) + 1
        return sorted([item for item, count in counts.items() if count >= self.threshold])


# ============================================================================
# Graphing Functions
# ============================================================================

def _plot_scaling_graph(
    x_values: list,
    plaintext_times: list,
    encrypted_times: list,
    title: str,
    x_label: str,
    output_path: str,
):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, plaintext_times, marker="o", label="Plaintext")

    has_any_encrypted = any(v is not None for v in encrypted_times)
    if has_any_encrypted:
        enc_plot = [np.nan if v is None else v for v in encrypted_times]
        plt.plot(x_values, enc_plot, marker="s", label="Encrypted (Concrete FHE)")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Measured computation time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_encrypted_breakdown_graph(
    x_values: list,
    preprocessing_plus_encryption_times: list,
    encrypted_compute_times: list,
    decryption_times: list,
    title: str,
    x_label: str,
    output_path: str,
):
    plt.figure(figsize=(10, 6))

    preenc_plot = [np.nan if v is None else v for v in preprocessing_plus_encryption_times]
    compute_plot = [np.nan if v is None else v for v in encrypted_compute_times]
    decrypt_plot = [np.nan if v is None else v for v in decryption_times]

    plt.plot(x_values, preenc_plot, marker="o", label="Preprocessing + Encryption")
    plt.plot(x_values, compute_plot, marker="s", label="Encrypted-space Computation")
    plt.plot(x_values, decrypt_plot, marker="^", label="Decryption")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Measured time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ============================================================================
# Scaling Experiments Support Functions
# ============================================================================

def _expand_range_spec(range_spec):
    if isinstance(range_spec, str):
        token = range_spec.strip().strip("[]")
        parts = [part.strip() for part in token.split("..") if part.strip()]
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            step = 1 if start <= end else -1
            return list(range(start, end + step, step))
        if len(parts) == 3:
            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
            if step == 0:
                raise ValueError("Range step cannot be zero.")
            return list(range(start, end + (1 if step > 0 else -1), step))

    if isinstance(range_spec, (list, tuple)):
        if len(range_spec) == 2:
            start, end = int(range_spec[0]), int(range_spec[1])
            step = 1 if start <= end else -1
            return list(range(start, end + step, step))
        if len(range_spec) == 3:
            start, end, step = int(range_spec[0]), int(range_spec[1]), int(range_spec[2])
            if step == 0:
                raise ValueError("Range step cannot be zero.")
            return list(range(start, end + (1 if step > 0 else -1), step))

    raise ValueError(f"Unsupported range specification: {range_spec}")


def _expand_sequence_spec(spec, default_values):
    if spec is None:
        return list(default_values)
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        if "values" in spec:
            return spec["values"]
        if "range" in spec:
            return _expand_range_spec(spec["range"])
        if "start" in spec and "end" in spec:
            return _expand_range_spec([spec["start"], spec["end"], spec.get("step", 1)])
    if isinstance(spec, str) and ".." in spec:
        return _expand_range_spec(spec)
    return list(default_values)


# ============================================================================
# Scaling Experiments - Main Function
# ============================================================================

def run_scaling_experiments(
    false_positive_rate: float = 0.0005,
    max_parties: int = 10,
    output_dir: str = "benchmark_outputs",
    scaling_sweeps: dict = None,
    only_sweep: str = "all",
):
    """
    Run scaling sweeps and save plots:
    1) parties sweep (2..max_parties)
    2) threshold sweep (1..N)
    3) party set size sweep (10..10^7)
    4) common IPs sweep (fixed N, T, set size)
    
    Parameters:
    -----------
    only_sweep : str, one of {"all", "parties", "threshold", "set_size", "common_ips"}
        Which sweep(s) to run. Default "all" runs all four.
    """
    default_sweeps = {
        "parties": {
            "n_values": {"range": [2, max_parties]},
            "threshold_mode": "half",
            "party_set_size": 10**3,
            "num_common_ips": 10,
        },
        "threshold": {
            "n_fixed": max_parties,
            "t_values": {"range": [1, max_parties]},
            "party_set_size": 10**3,
            "num_common_ips": 10,
        },
        "set_size": {
            "set_sizes": [10, 100, 1000],
            "num_parties": 10,
            "threshold": 5,
            "num_common_ips": 1,
        },
        "common_ips": {
            "common_values": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
            "num_parties": 10,
            "threshold": 5,
            "party_set_size": 10**3,
        },
    }
    if isinstance(scaling_sweeps, dict):
        for section, section_values in scaling_sweeps.items():
            if section in default_sweeps and isinstance(section_values, dict):
                default_sweeps[section].update(section_values)

    os.makedirs(output_dir, exist_ok=True)
    _all_start = time.perf_counter()

    only_sweep = (only_sweep or "all").strip().lower()
    valid_sweeps = {"all", "parties", "threshold", "set_size", "common_ips"}
    if only_sweep not in valid_sweeps:
        raise ValueError(f"Invalid only_sweep value: {only_sweep}")

    def _run_parties_sweep():
        _t0 = time.perf_counter()
        parties_x, parties_plain, parties_enc = [], [], []
        parties_preenc, parties_eval, parties_dec = [], [], []
        timing_rows = []

        parties_cfg = default_sweeps["parties"]
        parties_n_values = [
            int(v)
            for v in _expand_sequence_spec(parties_cfg.get("n_values"), list(range(2, max_parties + 1)))
        ]
        fixed_set_size_for_parties = int(parties_cfg.get("party_set_size", 10**3))
        parties_common_ips = int(parties_cfg.get("num_common_ips", 10))
        threshold_mode = str(parties_cfg.get("threshold_mode", "half"))

        print(f"[1/4] Parties sweep  N values: {parties_n_values}  (real IP files -- disk I/O per step)")
        for idx, n in enumerate(parties_n_values, 1):
            if threshold_mode == "half":
                t = max(1, n // 2)
            else:
                t = int(parties_cfg.get("threshold", max(1, n // 2)))
                t = max(1, min(t, n))

            tc = ThresholdCircuit(
                num_parties=n,
                threshold=t,
                false_positive_rate=false_positive_rate,
                num_common_ips=parties_common_ips,
                party_set_size=fixed_set_size_for_parties,
            )
            print(f"\n      [{idx}/{len(parties_n_values)}] Experiment: N={n}, T={t}")
            print(
                f"        Parameters: party_set_size={fixed_set_size_for_parties}, "
                f"num_common_ips={parties_common_ips}, FPR={tc.false_positive_rate:.3e}"
            )
            print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
            print("        Stage order: input load/create -> bloom build -> encrypt -> circuit -> plaintext -> encrypted computation \+ decryption")

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets()
            _exp_time = time.perf_counter() - _exp_start

            parties_x.append(n)
            parties_plain.append(metrics["plaintext_computation_time_s"])
            parties_enc.append(metrics["encrypted_computation_and_decryption_time_s"])
            parties_preenc.append(metrics["preprocessing_total_time_s"])
            parties_eval.append(metrics["encrypted_computation_time_s"])
            parties_dec.append(metrics["decryption_time_s"])
            timing_rows.append(
                {
                    "run": f"N={n},T={t}",
                    "dataset_source": metrics.get("dataset_source", "unknown"),
                    "preprocessing_total_time_s": metrics.get("preprocessing_total_time_s"),
                    "circuit_creation_time_s": metrics.get("circuit_creation_time_s"),
                    "plaintext_computation_time_s": metrics.get("plaintext_computation_time_s"),
                    "encrypted_computation_and_decryption_time_s": metrics.get("encrypted_computation_and_decryption_time_s"),
                }
            )
            print(
                f"        Completed in {_exp_time:.2f}s | source={metrics.get('dataset_source', 'unknown')} | "
                f"plaintext: {metrics['plaintext_computation_time_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_computation_and_decryption_time_s']:.3f}s"
            )
            print(
                f"        Breakdown: load={_format_seconds(metrics.get('load_dataset_time_s'))}, "
                f"bloom={_format_seconds(metrics.get('build_bloom_filters_time_s'))}, "
                f"encrypt={_format_seconds(metrics.get('input_encryption_time_s'))}, "
                f"enc_eval={_format_seconds(metrics.get('encrypted_computation_time_s'))}, "
                f"decrypt={_format_seconds(metrics.get('decryption_time_s'))}"
            )

        print(f"\n[1/4] Parties sweep  done  ({time.perf_counter()-_t0:.1f}s)")
        _plot_scaling_graph(
            parties_x,
            parties_plain,
            parties_enc,
            title="Computation Time vs Number of Parties",
            x_label="Number of parties (N)",
            output_path=os.path.join(output_dir, "scaling_num_parties.png"),
        )
        _plot_encrypted_breakdown_graph(
            parties_x,
            parties_preenc,
            parties_eval,
            parties_dec,
            title="Encrypted Timing Breakdown vs Number of Parties",
            x_label="Number of parties (N)",
            output_path=os.path.join(output_dir, "scaling_num_parties_encrypted_breakdown.png"),
        )
        _render_timing_table("Timing Table - Parties Sweep", timing_rows)

    def _run_threshold_sweep():
        _t0 = time.perf_counter()
        threshold_x, threshold_plain, threshold_enc = [], [], []
        threshold_preenc, threshold_eval, threshold_dec = [], [], []
        timing_rows = []

        threshold_cfg = default_sweeps["threshold"]
        fixed_set_size_for_threshold = int(threshold_cfg.get("party_set_size", 10**3))
        n_fixed = int(threshold_cfg.get("n_fixed", max_parties))
        threshold_common_ips = int(threshold_cfg.get("num_common_ips", 10))
        t_values = [int(v) for v in _expand_sequence_spec(threshold_cfg.get("t_values"), list(range(1, n_fixed + 1)))]
        t_values = [t for t in t_values if 1 <= t <= n_fixed]

        print(f"[2/4] Threshold sweep  T values: {t_values}  N={n_fixed}  (real IP files -- disk I/O per step)")
        for idx, t in enumerate(t_values, 1):
            tc = ThresholdCircuit(
                num_parties=n_fixed,
                threshold=t,
                false_positive_rate=false_positive_rate,
                num_common_ips=threshold_common_ips,
                party_set_size=fixed_set_size_for_threshold,
            )
            print(f"\n      [{idx}/{len(t_values)}] Experiment: N={n_fixed}, T={t}")
            print(
                f"        Parameters: party_set_size={fixed_set_size_for_threshold}, "
                f"num_common_ips={threshold_common_ips}, FPR={tc.false_positive_rate:.3e}"
            )
            print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
            print("        Stage order: input load/create -> bloom build -> encrypt -> circuit -> plaintext -> encrypted computation \+ decryption")

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets()
            _exp_time = time.perf_counter() - _exp_start

            threshold_x.append(t)
            threshold_plain.append(metrics["plaintext_computation_time_s"])
            threshold_enc.append(metrics["encrypted_computation_and_decryption_time_s"])
            threshold_preenc.append(metrics["preprocessing_total_time_s"])
            threshold_eval.append(metrics["encrypted_computation_time_s"])
            threshold_dec.append(metrics["decryption_time_s"])
            timing_rows.append(
                {
                    "run": f"N={n_fixed},T={t}",
                    "dataset_source": metrics.get("dataset_source", "unknown"),
                    "preprocessing_total_time_s": metrics.get("preprocessing_total_time_s"),
                    "circuit_creation_time_s": metrics.get("circuit_creation_time_s"),
                    "plaintext_computation_time_s": metrics.get("plaintext_computation_time_s"),
                    "encrypted_computation_and_decryption_time_s": metrics.get("encrypted_computation_and_decryption_time_s"),
                }
            )
            print(
                f"        Completed in {_exp_time:.2f}s | source={metrics.get('dataset_source', 'unknown')} | "
                f"plaintext: {metrics['plaintext_computation_time_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_computation_and_decryption_time_s']:.3f}s"
            )
            print(
                f"        Breakdown: load={_format_seconds(metrics.get('load_dataset_time_s'))}, "
                f"bloom={_format_seconds(metrics.get('build_bloom_filters_time_s'))}, "
                f"encrypt={_format_seconds(metrics.get('input_encryption_time_s'))}, "
                f"enc_eval={_format_seconds(metrics.get('encrypted_computation_time_s'))}, "
                f"decrypt={_format_seconds(metrics.get('decryption_time_s'))}"
            )

        print(f"\n[2/4] Threshold sweep  done  ({time.perf_counter()-_t0:.1f}s)")
        _plot_scaling_graph(
            threshold_x,
            threshold_plain,
            threshold_enc,
            title=f"Computation Time vs Threshold (N={n_fixed})",
            x_label="Threshold (T)",
            output_path=os.path.join(output_dir, "scaling_threshold.png"),
        )
        _plot_encrypted_breakdown_graph(
            threshold_x,
            threshold_preenc,
            threshold_eval,
            threshold_dec,
            title=f"Encrypted Timing Breakdown vs Threshold (N={n_fixed})",
            x_label="Threshold (T)",
            output_path=os.path.join(output_dir, "scaling_threshold_encrypted_breakdown.png"),
        )
        _render_timing_table("Timing Table - Threshold Sweep", timing_rows)

    def _run_set_size_sweep():
        _t0 = time.perf_counter()
        size_cfg = default_sweeps["set_size"]
        size_x = [int(v) for v in _expand_sequence_spec(size_cfg.get("set_sizes"), [10, 100, 1000])]
        size_plain, size_enc = [], []
        size_preenc, size_eval, size_dec = [], [], []
        timing_rows = []

        n_for_size = int(size_cfg.get("num_parties", 10))
        t_for_size = int(size_cfg.get("threshold", 5))
        c_for_size = int(size_cfg.get("num_common_ips", 1))

        print(f"[3/4] Set-size sweep  values: {size_x}  N={n_for_size}  T={t_for_size}  (real IP files -- disk I/O per step)")
        for i, set_size in enumerate(size_x, 1):
            tc = ThresholdCircuit(
                num_parties=n_for_size,
                threshold=t_for_size,
                false_positive_rate=false_positive_rate,
                num_common_ips=c_for_size,
                party_set_size=set_size,
            )
            print(f"\n      [{i}/{len(size_x)}] Experiment: N={n_for_size}, T={t_for_size}, party_set_size={set_size:,}")
            print(f"        Parameters: num_common_ips={c_for_size}, FPR={tc.false_positive_rate:.3e}")
            print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
            print("        Stage order: input load/create -> bloom build -> encrypt -> circuit -> plaintext -> encrypted computation \+ decryption")

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets()
            _exp_time = time.perf_counter() - _exp_start

            size_plain.append(metrics["plaintext_computation_time_s"])
            size_enc.append(metrics["encrypted_computation_and_decryption_time_s"])
            size_preenc.append(metrics["preprocessing_total_time_s"])
            size_eval.append(metrics["encrypted_computation_time_s"])
            size_dec.append(metrics["decryption_time_s"])
            timing_rows.append(
                {
                    "run": f"N={n_for_size},T={t_for_size},S={set_size}",
                    "dataset_source": metrics.get("dataset_source", "unknown"),
                    "preprocessing_total_time_s": metrics.get("preprocessing_total_time_s"),
                    "circuit_creation_time_s": metrics.get("circuit_creation_time_s"),
                    "plaintext_computation_time_s": metrics.get("plaintext_computation_time_s"),
                    "encrypted_computation_and_decryption_time_s": metrics.get("encrypted_computation_and_decryption_time_s"),
                }
            )
            print(
                f"        Completed in {_exp_time:.2f}s | source={metrics.get('dataset_source', 'unknown')} | "
                f"plaintext: {metrics['plaintext_computation_time_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_computation_and_decryption_time_s']:.3f}s"
            )
            print(
                f"        Breakdown: load={_format_seconds(metrics.get('load_dataset_time_s'))}, "
                f"bloom={_format_seconds(metrics.get('build_bloom_filters_time_s'))}, "
                f"encrypt={_format_seconds(metrics.get('input_encryption_time_s'))}, "
                f"enc_eval={_format_seconds(metrics.get('encrypted_computation_time_s'))}, "
                f"decrypt={_format_seconds(metrics.get('decryption_time_s'))}"
            )

        print(f"\n[3/4] Set-size sweep  done  ({time.perf_counter()-_t0:.1f}s)")
        _plot_scaling_graph(
            size_x,
            size_plain,
            size_enc,
            title=f"Computation Time vs Party Set Size (N={n_for_size}, T={t_for_size})",
            x_label="Party set size",
            output_path=os.path.join(output_dir, "scaling_party_set_size.png"),
        )
        _plot_encrypted_breakdown_graph(
            size_x,
            size_preenc,
            size_eval,
            size_dec,
            title=f"Encrypted Timing Breakdown vs Party Set Size (N={n_for_size}, T={t_for_size})",
            x_label="Party set size",
            output_path=os.path.join(output_dir, "scaling_party_set_size_encrypted_breakdown.png"),
        )
        _render_timing_table("Timing Table - Set-Size Sweep", timing_rows)

    def _run_common_ips_sweep():
        _t0 = time.perf_counter()
        common_cfg = default_sweeps["common_ips"]
        common_x = [int(v) for v in _expand_sequence_spec(common_cfg.get("common_values"), [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])]
        common_plain, common_enc = [], []
        common_preenc, common_eval, common_dec = [], [], []
        timing_rows = []

        n_for_common = int(common_cfg.get("num_parties", 10))
        t_for_common = int(common_cfg.get("threshold", 5))
        set_size_for_common = int(common_cfg.get("party_set_size", 10**3))

        print(f"[4/4] Common-IPs sweep  N={n_for_common}  T={t_for_common}  set_size={set_size_for_common:,}")
        print("      (party files are written to disk and regenerated whenever params change)")
        for i, common_ips in enumerate(common_x, 1):
            tc = ThresholdCircuit(
                num_parties=n_for_common,
                threshold=t_for_common,
                false_positive_rate=false_positive_rate,
                num_common_ips=common_ips,
                party_set_size=set_size_for_common,
            )
            print(f"\n      [{i}/{len(common_x)}] Experiment: N={n_for_common}, T={t_for_common}, num_common_ips={common_ips}")
            print(f"        Parameters: party_set_size={set_size_for_common:,}, FPR={tc.false_positive_rate:.3e}")
            print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
            print("        Stage order: input load/create -> bloom build -> encrypt -> circuit -> plaintext -> encrypted computation \+ decryption")

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets()
            _exp_time = time.perf_counter() - _exp_start

            common_plain.append(metrics["plaintext_computation_time_s"])
            common_enc.append(metrics["encrypted_computation_and_decryption_time_s"])
            common_preenc.append(metrics["preprocessing_total_time_s"])
            common_eval.append(metrics["encrypted_computation_time_s"])
            common_dec.append(metrics["decryption_time_s"])
            timing_rows.append(
                {
                    "run": f"N={n_for_common},T={t_for_common},C={common_ips}",
                    "dataset_source": metrics.get("dataset_source", "unknown"),
                    "preprocessing_total_time_s": metrics.get("preprocessing_total_time_s"),
                    "circuit_creation_time_s": metrics.get("circuit_creation_time_s"),
                    "plaintext_computation_time_s": metrics.get("plaintext_computation_time_s"),
                    "encrypted_computation_and_decryption_time_s": metrics.get("encrypted_computation_and_decryption_time_s"),
                }
            )
            print(
                f"        Completed in {_exp_time:.2f}s | source={metrics.get('dataset_source', 'unknown')} | "
                f"plaintext: {metrics['plaintext_computation_time_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_computation_and_decryption_time_s']:.3f}s"
            )
            print(
                f"        Breakdown: load={_format_seconds(metrics.get('load_dataset_time_s'))}, "
                f"bloom={_format_seconds(metrics.get('build_bloom_filters_time_s'))}, "
                f"encrypt={_format_seconds(metrics.get('input_encryption_time_s'))}, "
                f"enc_eval={_format_seconds(metrics.get('encrypted_computation_time_s'))}, "
                f"decrypt={_format_seconds(metrics.get('decryption_time_s'))}"
            )

        print(f"\n[4/4] Common-IPs sweep  done  ({time.perf_counter()-_t0:.1f}s)")
        _plot_scaling_graph(
            common_x,
            common_plain,
            common_enc,
            title=(
                "Computation Time vs Number of Common IPs "
                f"(N={n_for_common}, T={t_for_common}, SetSize={set_size_for_common})"
            ),
            x_label="Number of common IPs",
            output_path=os.path.join(output_dir, "scaling_num_common_ips.png"),
        )
        _plot_encrypted_breakdown_graph(
            common_x,
            common_preenc,
            common_eval,
            common_dec,
            title=(
                "Encrypted Timing Breakdown vs Number of Common IPs "
                f"(N={n_for_common}, T={t_for_common}, SetSize={set_size_for_common})"
            ),
            x_label="Number of common IPs",
            output_path=os.path.join(output_dir, "scaling_num_common_ips_encrypted_breakdown.png"),
        )
        _render_timing_table("Timing Table - Common-IPs Sweep", timing_rows)

    # Execute selected sweeps
    if only_sweep == "all":
        _run_parties_sweep()
        _run_threshold_sweep()
        _run_set_size_sweep()
        _run_common_ips_sweep()
    elif only_sweep == "parties":
        _run_parties_sweep()
    elif only_sweep == "threshold":
        _run_threshold_sweep()
    elif only_sweep == "set_size":
        _run_set_size_sweep()
    elif only_sweep == "common_ips":
        _run_common_ips_sweep()

    print(f"\nCompleted selected scaling sweep(s) in {time.perf_counter()-_all_start:.1f}s  --  plots saved in '{output_dir}'")


# ============================================================================
# Menu Functions (for interactive mode)
# ============================================================================

def _prompt_numbered_choice(title: str, options: list, default_index: int = 0):
    print(f"\n{title}")
    for idx, option in enumerate(options, start=1):
        print(f"{idx}) {option}")

    while True:
        choice = input(f"Choose [default {default_index + 1}]: ").strip()
        if not choice:
            return options[default_index]
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Try again.")


def prompt_sample_parameters(defaults: dict) -> dict:
    num_parties = _prompt_numbered_choice(
        "Choose number of parties",
        [2, 3, 4, 5],
        default_index=1,
    )
    threshold = _prompt_numbered_choice(
        "Choose threshold",
        list(range(1, num_parties + 1)),
        default_index=min(defaults["threshold"], num_parties) - 1,
    )
    num_common_ips = _prompt_numbered_choice(
        "Choose number of common IPs",
        [1, 2, 3, 5, 10],
        default_index=2,
    )
    party_set_size = _prompt_numbered_choice(
        "Choose party set size",
        [10, 25, 50, 100, 250],
        default_index=3,
    )
    return {
        "num_parties": num_parties,
        "threshold": threshold,
        "num_common_ips": num_common_ips,
        "party_set_size": party_set_size,
    }


def run_sample_demo(config: dict) -> None:
    tc = ThresholdCircuit(**config)
    print("\n[RUN] Starting custom/sample execution")
    print(f"[RUN] Configuration: N={tc.num_parties}, T={tc.threshold}, C={tc.num_common_ips}, S={tc.party_set_size}")
    print(f"[RUN] Bloom filter size M={tc.num_bloom_bits:,} bits, hash functions K={tc.num_hash_funcs}")
    print(
        "[RUN] False positive rate "
        f"target={tc.requested_false_positive_rate:.3e} (1/party_set_size), "
        f"effective={tc.false_positive_rate:.3e}"
    )

    details = tc.run_detailed_pipeline()

    party_sets = details["party_sets"]
    bloom_filters = details["bloom_filters"]
    canonical_circuit = details["canonical_circuit"]
    optimized_circuit = details["optimized_circuit"]
    plaintext_result = details["plaintext_result"]
    decrypted_result = details["decrypted_result"]
    timings = details["timings"]

    print(f"[RUN] Dataset source: {details['dataset_source']}")
    print(f"[RUN] Party element counts: {[len(party) for party in party_sets]}")
    print(
        f"[RUN] Built {len(bloom_filters)} Bloom filters "
        f"({tc.num_bloom_bits} bits each, K={tc.num_hash_funcs}, FPR={tc.false_positive_rate:.3e})"
    )
    for idx, bloom_filter in enumerate(bloom_filters, start=1):
        print(f"[RUN] Bloom filter Party-{idx}: {bloom_filter}")

    print(f"[RUN] Canonical circuit: {canonical_circuit}")
    print(f"[RUN] Optimized circuit: {optimized_circuit}")

    ones_count = sum(plaintext_result)
    print(f"\nIntersection Bloom filter: {len(plaintext_result)} bits,  "
          f"{ones_count} set ({100*ones_count/len(plaintext_result):.1f}% density)")
    print(f"[RUN] Plaintext intersection bits: {plaintext_result}")

    recovered_from_plain = tc.extract_candidates_from_intersection_bloom(
        plaintext_result,
        party_sets,
    )
    exact_elements = tc.exact_threshold_intersection(party_sets)
    print(f"Recovered intersection candidates from plaintext bloom result: {recovered_from_plain}")
    print(f"Exact threshold intersection from plaintext sets: {exact_elements}")

    print("\n--- Concrete-Python Encrypted Evaluation ---")
    if decrypted_result is None:
        print("[RUN] Encrypted evaluation not available on this machine")
    else:
        print(f"[RUN] Decrypted intersection bits: {decrypted_result}")
        matches = decrypted_result == plaintext_result
        print(f"[RUN] Matches plaintext result: {matches}")
        recovered_candidates = tc.extract_candidates_from_intersection_bloom(
            decrypted_result,
            party_sets,
        )
        print(f"[RUN] Recovered candidates from decrypted bloom filter: {recovered_candidates}")

    _render_timing_table(
        "Timing Table - Custom Run",
        [
            {
                "run": tc._dataset_tag(),
                "dataset_source": details["dataset_source"],
                "preprocessing_total_time_s": timings["preprocessing_total_time_s"],
                "circuit_creation_time_s": timings["circuit_creation_time_s"],
                "plaintext_computation_time_s": timings["plaintext_computation_time_s"],
                "encrypted_computation_and_decryption_time_s": timings["encrypted_computation_and_decryption_time_s"],
            }
        ],
    )


def _print_prep_preflight_stats(ordered_configs: list, filtered_configs: list, skipped_count: int) -> None:
    """Print preflight summary before starting dataset preparation."""
    print("\n" + "="*60)
    print("PREFLIGHT: Dataset Preparation Summary")
    print("="*60)
    print(f"Total configs requested: {len(ordered_configs)}")
    print(f"Safe configs to generate: {len(filtered_configs)}")
    if skipped_count > 0:
        print(f"[WARN] Oversized configs (skipped): {skipped_count}")
    
    if os.path.exists(ThresholdCircuit.PARTY_SETS_DIR):
        cached_dirs = [d for d in os.listdir(ThresholdCircuit.PARTY_SETS_DIR)
                      if os.path.isdir(os.path.join(ThresholdCircuit.PARTY_SETS_DIR, d))]
        print(f"Cached datasets available: {len(cached_dirs)}")
        if cached_dirs:
            print("  Cached configs:")
            for cached_dir in sorted(cached_dirs)[:10]:
                print(f"    - {cached_dir}")
            if len(cached_dirs) > 10:
                print(f"     and {len(cached_dirs) - 10} more")
    print("="*60 + "\n")


def prepare_input_files(sample_config: dict, scaling_config: dict) -> None:
    print("\nPreparing input IP files for sample run and scaling experiments")

    sweeps_cfg = scaling_config.get("scaling_sweeps", {}) if isinstance(scaling_config, dict) else {}
    max_parties = int(scaling_config.get("max_parties", 10)) if isinstance(scaling_config, dict) else 10
    configs = {
        (
            int(sample_config["num_parties"]),
            int(sample_config["threshold"]),
            int(sample_config["num_common_ips"]),
            int(sample_config["party_set_size"]),
        )
    }

    parties_cfg = sweeps_cfg.get("parties", {}) if isinstance(sweeps_cfg, dict) else {}
    parties_n_values = [int(v) for v in _expand_sequence_spec(parties_cfg.get("n_values"), list(range(2, max_parties + 1)))]
    parties_set_size = int(parties_cfg.get("party_set_size", 10**3))
    parties_common_ips = int(parties_cfg.get("num_common_ips", 10))
    parties_mode = str(parties_cfg.get("threshold_mode", "half"))
    parties_fixed_t = int(parties_cfg.get("threshold", 1))
    for n in parties_n_values:
        t = max(1, n // 2) if parties_mode == "half" else max(1, min(parties_fixed_t, n))
        configs.add((n, t, parties_common_ips, parties_set_size))

    threshold_cfg = sweeps_cfg.get("threshold", {}) if isinstance(sweeps_cfg, dict) else {}
    n_fixed = int(threshold_cfg.get("n_fixed", max_parties))
    t_values = [int(v) for v in _expand_sequence_spec(threshold_cfg.get("t_values"), list(range(1, n_fixed + 1)))]
    threshold_set_size = int(threshold_cfg.get("party_set_size", 10**3))
    threshold_common_ips = int(threshold_cfg.get("num_common_ips", 10))
    for t in t_values:
        if 1 <= t <= n_fixed:
            configs.add((n_fixed, t, threshold_common_ips, threshold_set_size))

    set_size_cfg = sweeps_cfg.get("set_size", {}) if isinstance(sweeps_cfg, dict) else {}
    set_sizes = [int(v) for v in _expand_sequence_spec(set_size_cfg.get("set_sizes"), [10, 100, 1000])]
    n_for_size = int(set_size_cfg.get("num_parties", 10))
    t_for_size = int(set_size_cfg.get("threshold", 5))
    c_for_size = int(set_size_cfg.get("num_common_ips", 1))
    for set_size in set_sizes:
        configs.add((n_for_size, t_for_size, c_for_size, set_size))

    common_cfg = sweeps_cfg.get("common_ips", {}) if isinstance(sweeps_cfg, dict) else {}
    common_values = [int(v) for v in _expand_sequence_spec(common_cfg.get("common_values"), [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])]
    n_for_common = int(common_cfg.get("num_parties", 10))
    t_for_common = int(common_cfg.get("threshold", 5))
    set_size_for_common = int(common_cfg.get("party_set_size", 10**3))
    for common_ips in common_values:
        configs.add((n_for_common, t_for_common, common_ips, set_size_for_common))

    print(
        "Active prep config: "
        f"max_parties={max_parties}, "
        f"parties_set_size={parties_set_size}, "
        f"threshold_n_fixed={n_fixed}, threshold_set_size={threshold_set_size}, "
        f"set_size_values={set_sizes}, "
        f"common_n={n_for_common}, common_t={t_for_common}, common_set_size={set_size_for_common}, "
        f"common_values={common_values}"
    )

    ordered_configs = sorted(
        configs,
        key=lambda cfg: (cfg[0], cfg[1], cfg[2], cfg[3]),
        reverse=True,
    )
    requested_configs = list(ordered_configs)

    requested_parallel = int(scaling_config.get("parallel_dataset_workers", 0)) if isinstance(scaling_config, dict) else 0
    allow_unsafe_parallel = bool(scaling_config.get("allow_unsafe_parallel_prep", False)) if isinstance(scaling_config, dict) else False
    skip_oversized_prep = bool(scaling_config.get("skip_oversized_prep", False)) if isinstance(scaling_config, dict) else False
    max_prepare_workload = int(scaling_config.get("max_prepare_workload", 0)) if isinstance(scaling_config, dict) else 0
    cooldown_seconds = float(scaling_config.get("cooldown_seconds_between_datasets", 0.0)) if isinstance(scaling_config, dict) else 0.0

    skipped_configs = []
    if skip_oversized_prep and not allow_unsafe_parallel and max_prepare_workload > 0:
        filtered_configs = []
        for cfg in ordered_configs:
            n, _t, _c, s = cfg
            workload = n * s
            if workload > max_prepare_workload:
                skipped_configs.append((cfg, workload))
            else:
                filtered_configs.append(cfg)

        if skipped_configs:
            print(
                f"Skipping {len(skipped_configs)} oversized dataset prep jobs "
                f"(max_prepare_workload={max_prepare_workload:,}, set allow_unsafe_parallel_prep=true to override)."
            )

        ordered_configs = filtered_configs
        if not ordered_configs:
            print("No safe dataset prep jobs remain after workload filtering.")
            return
    elif not allow_unsafe_parallel and max_prepare_workload > 0:
        print(
            "Oversized-prep skipping is disabled (skip_oversized_prep=false). "
            "All requested datasets will be generated/adapted."
        )

    _print_prep_preflight_stats(requested_configs, ordered_configs, len(skipped_configs))

    # Heuristic workload proxy: roughly proportional to in-memory set footprint.
    workload_sizes = [n * s for (n, _t, _c, s) in ordered_configs]
    max_workload = max(workload_sizes) if workload_sizes else 0

    # Input dataset preparation runs sequentially by request.
    max_parallel_datasets = 1

    # Large datasets can consume substantial memory; throttle by default to avoid OOM kills.
    if not allow_unsafe_parallel:
        if max_workload >= 2_000_000:
            max_parallel_datasets = min(max_parallel_datasets, 1)
        elif max_workload >= 800_000:
            max_parallel_datasets = min(max_parallel_datasets, 2)

    print(
        f"Preparing {len(ordered_configs)} dataset configurations "
        f"with max_parallel_datasets={max_parallel_datasets} "
        f"(largest workload proxy N*set_size={max_workload:,})"
    )

    def _prepare_one(tc: ThresholdCircuit):
        started = time.perf_counter()
        tc.load_or_create_party_sets()
        return tc._dataset_tag(), (time.perf_counter() - started)

    total = len(ordered_configs)
    for idx, config in enumerate(ordered_configs, start=1):
        n, t, c, s = config
        tc = ThresholdCircuit(
            num_parties=n,
            threshold=t,
            num_common_ips=c,
            party_set_size=s,
        )
        dataset_tag = tc._dataset_tag()
        print(f"[{idx}/{total}] Starting {dataset_tag}")
        prepared_tag, elapsed = _prepare_one(tc)
        print(f"[{idx}/{total}] Prepared {prepared_tag} in {elapsed:.1f}s")
        gc.collect()
        if cooldown_seconds > 0:
            print(f"[{idx}/{total}] Cooldown for {cooldown_seconds:.1f}s")
            time.sleep(cooldown_seconds)

    print("All requested input IP files are ready.")


def verify_single_dataset(dataset_name: str) -> None:
    """
    Verify a single cached dataset by name (e.g., 'n20_t10_c10_s1000000').
    Checks if directory name parameters match actual data in files.
    """
    dataset_path = os.path.join(ThresholdCircuit.PARTY_SETS_DIR, dataset_name)
    
    if not os.path.isdir(dataset_path):
        print(f"\n[FAIL] Dataset '{dataset_name}' not found in '{ThresholdCircuit.PARTY_SETS_DIR}'.")
        return
    
    result = {
        "name": dataset_name,
        "path": dataset_path,
        "status": "PASS",
        "errors": [],
    }
    
    print(f"\nVerifying dataset: {dataset_name}\n")
    
    # Parse directory name
    match = re.match(r"n(\d+)_t(\d+)_c(\d+)_s(\d+)", dataset_name)
    if not match:
        print(f"[FAIL] Invalid directory name format: {dataset_name}")
        return
    
    expected_n, expected_t, expected_c, expected_s = map(int, match.groups())
    effective_expected_c = (expected_n * expected_s) if expected_t == 1 else expected_c
    if expected_t == 1:
        print(
            f"Expected parameters: N={expected_n}, T={expected_t}, "
            f"C={effective_expected_c} (effective; dir has c={expected_c}), S={expected_s}"
        )
    else:
        print(f"Expected parameters: N={expected_n}, T={expected_t}, C={expected_c}, S={expected_s}")
    
    # Read metadata
    meta_path = os.path.join(dataset_path, ThresholdCircuit.PARTY_SETS_META_FILE)
    if not os.path.exists(meta_path):
        print(f"[FAIL] Missing metadata file: {ThresholdCircuit.PARTY_SETS_META_FILE}\n")
        return
    
    try:
        with open(meta_path, "r") as fh:
            metadata = json.load(fh)
    except Exception as e:
        print(f"[FAIL] Failed to read metadata: {e}\n")
        return
    
    # Validate metadata
    meta_n = int(metadata.get("num_parties", 0))
    meta_t = int(metadata.get("threshold", 0))
    meta_c = int(metadata.get("num_common_ips", 0))
    meta_s = int(metadata.get("party_set_size", 0))
    
    print(f"Metadata values:  N={meta_n}, T={meta_t}, C={meta_c}, S={meta_s}")
    
    errors = []
    if meta_n != expected_n:
        errors.append(f"num_parties: dir={expected_n}, meta={meta_n}")
    if meta_t != expected_t:
        errors.append(f"threshold: dir={expected_t}, meta={meta_t}")
    if expected_t != 1 and meta_c != expected_c:
        errors.append(f"num_common_ips: dir={expected_c}, meta={meta_c}")
    if meta_s != expected_s:
        errors.append(f"party_set_size: dir={expected_s}, meta={meta_s}")
    
    if errors:
        print("\n[FAIL] Metadata mismatches:")
        for err in errors:
            print(f"    |- {err}")
    else:
        print("[OK] Metadata matches directory name")
    if expected_t == 1:
        print("[OK] Metadata C check skipped for T=1 (effective C is N*S)")
    
    # Check party files
    party_files = []
    for i in range(expected_n):
        party_file = os.path.join(dataset_path, f"party_{i + 1}.txt")
        if os.path.exists(party_file):
            party_files.append(party_file)
    
    if len(party_files) != expected_n:
        print(f"\n[FAIL] Party files: Expected {expected_n}, found {len(party_files)}")
        return
    
    print(f"[OK] All {expected_n} party files present")
    
    # Read and validate party sets
    try:
        party_sets = []
        size_errors = []
        for i, party_file in enumerate(party_files):
            with open(party_file, "r") as fh:
                ips = [line.strip() for line in fh if line.strip()]
            party_sets.append(ips)
            if len(ips) != expected_s:
                size_errors.append(f"party_{i + 1}.txt: {len(ips)} IPs (expected {expected_s})")
        
        if size_errors:
            print("\n[FAIL] Party set size mismatches:")
            for err in size_errors:
                print(f"    |- {err}")
        else:
            print(f"[OK] All party sets have {expected_s} IPs")
        
        # Validate IP structure
        ip_counts = {}
        for party in party_sets:
            for ip in set(party):
                ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        common_ips = [ip for ip, cnt in ip_counts.items() if cnt == expected_t]
        unique_ips = [ip for ip, cnt in ip_counts.items() if cnt == 1]
        if expected_t == 1:
            invalid_ips = [ip for ip, cnt in ip_counts.items() if cnt != 1]
        else:
            invalid_ips = [ip for ip, cnt in ip_counts.items() if cnt not in (1, expected_t)]
        
        print(f"\nIP Structure:")
        print(
            f"  |- Common IPs (appearing {expected_t} times): "
            f"{len(common_ips)} (expected {effective_expected_c})"
        )
        print(f"  |- Unique IPs (appearing 1 time): {len(unique_ips)}")
        
        if len(common_ips) != effective_expected_c:
            print(f"\n[FAIL] Common IP count mismatch: found {len(common_ips)}, expected {effective_expected_c}")
        else:
            print(f"[OK] Common IP count matches")

        if invalid_ips:
            print(f"\n[FAIL] Invalid IP occurrences: {len(invalid_ips)} IPs with invalid counts")
            if expected_t == 1:
                print(f"    (Should only have counts of 1)")
            else:
                print(f"    (Should only have counts of 1 or {expected_t})")
            for ip in invalid_ips[:5]:
                cnt = ip_counts[ip]
                print(f"    |- {ip}: appears {cnt} times")
            if len(invalid_ips) > 5:
                print(f"    |-  and {len(invalid_ips) - 5} more")
        else:
            print(f"[OK] All IPs have valid occurrence counts")
        
        total_ips = sum(len(party) for party in party_sets)
        expected_total = expected_n * expected_s
        if total_ips != expected_total:
            print(f"\n[FAIL] Total IPs mismatch: {total_ips} (expected {expected_total})")
        else:
            print(f"[OK] Total IP count correct ({total_ips})")
        
        # Final status
        common_count_failed = (len(common_ips) != effective_expected_c)
        if errors or size_errors or common_count_failed or invalid_ips or (total_ips != expected_total):
            print(f"\n[FAIL] Dataset FAILED validation")
        else:
            print(f"\n[OK] Dataset PASSED all validations")
        
    except Exception as e:
        print(f"\n[FAIL] Error reading/parsing party files: {e}")


def verify_all_cached_datasets() -> None:
    """
    Scan party_sets/ directory and validate all cached datasets.
    Checks if directory name parameters match actual data in party_*.txt files.
    """
    if not os.path.exists(ThresholdCircuit.PARTY_SETS_DIR):
        print(f"\nNo datasets found. Directory '{ThresholdCircuit.PARTY_SETS_DIR}' does not exist.")
        return

    dataset_dirs = []
    for name in sorted(os.listdir(ThresholdCircuit.PARTY_SETS_DIR)):
        dataset_path = os.path.join(ThresholdCircuit.PARTY_SETS_DIR, name)
        if os.path.isdir(dataset_path):
            dataset_dirs.append((name, dataset_path))

    if not dataset_dirs:
        print(f"\nNo datasets found in '{ThresholdCircuit.PARTY_SETS_DIR}'.")
        return

    print(f"\n{'='*70}")
    print(f"Verifying {len(dataset_dirs)} cached dataset(s)")
    print(f"{'='*70}\n")

    passed = 0
    failed = 0
    results = []

    for idx, (dir_name, dir_path) in enumerate(dataset_dirs, start=1):
        print(f"[{idx}/{len(dataset_dirs)}] Checking {dir_name}", end=" ", flush=True)
        
        result = {
            "name": dir_name,
            "path": dir_path,
            "status": "PASS",
            "errors": [],
        }

        # Parse directory name (format: n{N}_t{T}_c{C}_s{S})
        match = re.match(r"n(\d+)_t(\d+)_c(\d+)_s(\d+)", dir_name)
        if not match:
            result["status"] = "FAIL"
            result["errors"].append(f"Invalid directory name format: {dir_name}")
            results.append(result)
            failed += 1
            print("FAIL (invalid name format)")
            continue

        expected_n, expected_t, expected_c, expected_s = map(int, match.groups())
        effective_expected_c = (expected_n * expected_s) if expected_t == 1 else expected_c

        # Read metadata file
        meta_path = os.path.join(dir_path, ThresholdCircuit.PARTY_SETS_META_FILE)
        if not os.path.exists(meta_path):
            result["status"] = "FAIL"
            result["errors"].append(f"Missing metadata file: {ThresholdCircuit.PARTY_SETS_META_FILE}")
            results.append(result)
            failed += 1
            print("FAIL (missing metadata)")
            continue

        try:
            with open(meta_path, "r") as fh:
                metadata = json.load(fh)
        except Exception as e:
            result["status"] = "FAIL"
            result["errors"].append(f"Failed to read metadata: {e}")
            results.append(result)
            failed += 1
            print(f"FAIL (metadata read error: {e})")
            continue

        # Validate metadata against directory name
        meta_n = int(metadata.get("num_parties", 0))
        meta_t = int(metadata.get("threshold", 0))
        meta_c = int(metadata.get("num_common_ips", 0))
        meta_s = int(metadata.get("party_set_size", 0))

        if meta_n != expected_n:
            result["errors"].append(f"num_parties mismatch: dir={expected_n}, meta={meta_n}")
        if meta_t != expected_t:
            result["errors"].append(f"threshold mismatch: dir={expected_t}, meta={meta_t}")
        if expected_t != 1 and meta_c != expected_c:
            result["errors"].append(f"num_common_ips mismatch: dir={expected_c}, meta={meta_c}")
        if meta_s != expected_s:
            result["errors"].append(f"party_set_size mismatch: dir={expected_s}, meta={meta_s}")

        # Check party files exist
        party_files = []
        for i in range(expected_n):
            party_file = os.path.join(dir_path, f"party_{i + 1}.txt")
            if not os.path.exists(party_file):
                result["errors"].append(f"Missing party file: party_{i + 1}.txt")
            else:
                party_files.append(party_file)

        if len(party_files) != expected_n:
            result["status"] = "FAIL"
            result["errors"].append(f"Expected {expected_n} party files, found {len(party_files)}")
            results.append(result)
            failed += 1
            print(f"FAIL ({len(party_files)}/{expected_n} files)")
            continue

        # Read all party sets and count IPs
        try:
            party_sets = []
            for party_file in party_files:
                with open(party_file, "r") as fh:
                    ips = [line.strip() for line in fh if line.strip()]
                party_sets.append(ips)

            # Validate party set sizes
            for i, ips in enumerate(party_sets):
                if len(ips) != expected_s:
                    result["errors"].append(
                        f"party_{i + 1}.txt has {len(ips)} IPs, expected {expected_s}"
                    )

            # Count IP occurrences across all parties
            ip_counts = {}
            for party in party_sets:
                for ip in set(party):
                    ip_counts[ip] = ip_counts.get(ip, 0) + 1

            # Validate common IP structure
            common_ips = [ip for ip, count in ip_counts.items() if count == expected_t]
            unique_ips = [ip for ip, count in ip_counts.items() if count == 1]
            if expected_t == 1:
                other_ips = [ip for ip, count in ip_counts.items() if count != 1]
            else:
                other_ips = [ip for ip, count in ip_counts.items() if count not in (1, expected_t)]

            if len(common_ips) != effective_expected_c:
                result["errors"].append(
                    f"Expected {effective_expected_c} common IPs (appearing {expected_t} times), "
                    f"found {len(common_ips)}"
                )

            if other_ips:
                if expected_t == 1:
                    result["errors"].append(
                        f"Found {len(other_ips)} IPs with invalid occurrence counts (not 1): "
                        f"{other_ips[:5]}{'' if len(other_ips) > 5 else ''}"
                    )
                else:
                    result["errors"].append(
                        f"Found {len(other_ips)} IPs with invalid occurrence counts (not 1 or {expected_t}): "
                        f"{other_ips[:5]}{'' if len(other_ips) > 5 else ''}"
                    )

            # Check total IPs
            total_ips_in_sets = sum(len(party) for party in party_sets)
            expected_total = expected_n * expected_s
            if total_ips_in_sets != expected_total:
                result["errors"].append(
                    f"Total IPs mismatch: sum={total_ips_in_sets}, expected={expected_total}"
                )

        except Exception as e:
            result["status"] = "FAIL"
            result["errors"].append(f"Error reading/parsing party files: {e}")
            results.append(result)
            failed += 1
            print(f"FAIL (read error: {e})")
            continue

        # Determine final status
        if result["errors"]:
            result["status"] = "FAIL"
            failed += 1
            print(f"FAIL ({len(result['errors'])} error(s))")
        else:
            passed += 1
            print("PASS")

        results.append(result)

    # Print detailed results for failed datasets
    print(f"\n{'='*70}")
    if failed > 0:
        print("Failed Dataset Details:")
        print(f"{'='*70}\n")
        for result in results:
            if result["status"] == "FAIL":
                print(f"[FAIL] {result['name']}")
                for error in result["errors"]:
                    print(f"    |- {error}")
                print()

    print(f"{'='*70}")
    print(f"Summary: {passed} PASS, {failed} FAIL out of {len(dataset_dirs)} total")
    print(f"{'='*70}\n")


def run_menu(sample_config: dict, scaling_config: dict) -> None:
    """Interactive menu-driven interface."""
    while True:
        print("\n" + "="*50)
        print("Threshold PSI - Circuit TPSI")
        print("="*50)
        print("1) Generate input files")
        print("2) Run sample experiment")
        print("3) Run scaling experiments")
        print("4) Verify all cached datasets")
        print("5) Verify specific dataset")
        print("6) Exit")

        choice = input("\nSelect option: ").strip()
        if choice == "1":
            prepare_input_files(sample_config, scaling_config)
        elif choice == "2":
            chosen_config = prompt_sample_parameters(sample_config)
            run_sample_demo(chosen_config)
        elif choice == "3":
            print("\n--- Scaling Experiments ---")
            sweep_choice = _prompt_numbered_choice(
                "Choose scaling sweep",
                [
                    "all (all 4 sweeps)",
                    "parties (number of parties)",
                    "threshold (threshold values)",
                    "set_size (party set sizes)",
                    "common_ips (common IP counts)",
                ],
                default_index=0,
            )
            # Map user choice to sweep name
            sweep_map = {
                "all (all 4 sweeps)": "all",
                "parties (number of parties)": "parties",
                "threshold (threshold values)": "threshold",
                "set_size (party set sizes)": "set_size",
                "common_ips (common IP counts)": "common_ips",
            }
            selected_sweep = sweep_map.get(sweep_choice, "all")
            
            run_scaling_experiments(
                false_positive_rate=float(
                    scaling_config.get("false_positive_rate", sample_config.get("false_positive_rate", 0.0005))
                ),
                max_parties=int(scaling_config.get("max_parties", 10)),
                output_dir=str(scaling_config.get("output_dir", "benchmark_outputs")),
                scaling_sweeps=scaling_config.get("scaling_sweeps"),
                only_sweep=selected_sweep,
            )
        elif choice == "4":
            verify_all_cached_datasets()
        elif choice == "5":
            dataset_name = input("\nEnter dataset name (e.g., n20_t10_c10_s1000000): ").strip()
            if dataset_name:
                verify_single_dataset(dataset_name)
            else:
                print("No dataset name provided.")
        elif choice == "6":
            print("\nExiting.")
            break
        else:
            print("Invalid choice. Try again.")


def load_runtime_config() -> dict:
    """Load runtime configuration strictly from party_sets/party_sets_meta.json."""
    os.makedirs(ThresholdCircuit.PARTY_SETS_DIR, exist_ok=True)
    config_path = os.path.join(
        ThresholdCircuit.PARTY_SETS_DIR,
        ThresholdCircuit.PARTY_SETS_META_FILE,
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Runtime config file not found: {config_path}. "
            "Please create it with 'sample_config' and 'scaling_config'."
        )

    with open(config_path, "r") as fh:
        config = json.load(fh)

    if not isinstance(config, dict):
        raise ValueError("Runtime config must be a JSON object.")
    if "sample_config" not in config or not isinstance(config["sample_config"], dict):
        raise ValueError("Runtime config must include object field 'sample_config'.")
    if "scaling_config" not in config or not isinstance(config["scaling_config"], dict):
        raise ValueError("Runtime config must include object field 'scaling_config'.")

    config["script_mode"] = "circuit_tpsi"
    return config


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    log_path = setup_realtime_logging("party_sets/party_sets_runtime.log")
    runtime = load_runtime_config()
    print(f"[INIT] Runtime config loaded from {os.path.join(ThresholdCircuit.PARTY_SETS_DIR, ThresholdCircuit.PARTY_SETS_META_FILE)}")
    print(f"[INIT] Realtime append logging enabled at {log_path}")
    
    if USE_MENU:
        # Interactive menu mode
        print("[INIT] Starting in interactive menu mode")
        run_menu(runtime["sample_config"], runtime["scaling_config"])
    else:
        # Standalone script mode - run sample demo and optionally scaling experiments
        print("\n" + "="*60)
        print("Threshold PSI - Circuit TPSI (Standalone Script Mode)")
        print("="*60)
        
        # Run sample demo
        print("\n[1/2] Running sample experiment")
        run_sample_demo(runtime["sample_config"])
        
        # Optionally run scaling experiments
        print("\n[2/2] Running all scaling experiments")
        run_scaling_experiments(
            false_positive_rate=float(
                runtime["scaling_config"].get("false_positive_rate", 0.0005)
            ),
            max_parties=int(runtime["scaling_config"].get("max_parties", 10)),
            output_dir=str(runtime["scaling_config"].get("output_dir", "benchmark_outputs")),
            scaling_sweeps=runtime["scaling_config"].get("scaling_sweeps"),
            only_sweep="all",
        )
        
        print("\n" + "="*60)
        print("All operations completed successfully!")
        print("="*60)



