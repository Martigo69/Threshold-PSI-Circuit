#!/usr/bin/env python
import argparse
import itertools
import random
import os
import json
import array
import time
import math
import re
import gc
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime

import numpy as np
from probables import BloomFilter

from openfhe import *  # noqa: F401,F403

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


def configure_output_mode(use_realtime_log: bool, log_file_path: str = "party_sets/party_sets_runtime.log"):
    """Configure runtime output destination and return log path when file-redirect is active."""
    if not use_realtime_log:
        return None

    return setup_realtime_logging(log_file_path)


def _format_seconds(value) -> str:
    """Format a seconds value for tables, returning N/A for missing values."""
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


def _render_hierarchical_timing_table(title: str, rows: list) -> None:
    """
    Render a hierarchical timing table with grouped categories:
    - Preprocessing: Load | Build BF | Encrypt | Total
    - Circuit: Creation | Optimization | Total  
    - Computation: Encrypted Eval | Decrypt | Extract IPs | Total
    """
    if not rows:
        print(f"\n{title}: no rows")
        return

    # Headers with hierarchical structure
    prep_load = "Load"
    prep_bf = "Build BF"
    prep_enc = "FHE Prep"
    prep_total = "Total"
    
    circ_create = "Creation"
    circ_optim = "Optimization"
    circ_total = "Total"
    
    comp_eval = "Eval"
    comp_dec = "Decrypt"
    comp_extract = "Extract IPs"
    comp_total = "Total"

    print(f"\n{title}")
    print("=" * 180)
    
    # Main section headers
    print(f"{'Run':<20} | {'Preprocessing':<60} | {'Circuit':<45} | {'Computation':<50} |")
    
    # Sub-headers
    print(f"{'-'*20}-+-{'-'*60}-+-{'-'*45}-+-{'-'*50}-+")
    print(f"{'':20} | "
          f"{prep_load:<12} {prep_bf:<12} {prep_enc:<12} {prep_total:<20} | "
          f"{circ_create:<14} {circ_optim:<15} {circ_total:<12} | "
          f"{comp_eval:<10} {comp_dec:<10} {comp_extract:<12} {comp_total:<12} |")
    print("=" * 180)
    
    # Data rows
    for row in rows:
        run_name = str(row.get("run", "-"))[:20]
        
        # Preprocessing components
        load_time = _format_seconds(row.get("load_dataset_time_s"))
        bf_time = _format_seconds(row.get("build_bloom_filters_time_s"))
        enc_time = _format_seconds(row.get("input_encryption_time_s"))
        prep_total_time = _format_seconds(row.get("preprocessing_total_time_s"))
        
        # Circuit components
        circ_create_time = _format_seconds(row.get("circuit_creation_time_s"))
        circ_optim_time = _format_seconds(row.get("circuit_optimization_time_s"))
        circ_total_time = _format_seconds(row.get("circuit_total_time_s"))
        
        # Computation components
        encrypted_eval_time = _format_seconds(row.get("encrypted_computation_time_s"))
        dec_time = _format_seconds(row.get("decryption_time_s"))
        extract_time = _format_seconds(row.get("encrypted_extract_time_s"))
        comp_total_time = _format_seconds(row.get("encrypted_computation_and_decryption_time_s"))
        
        print(f"{run_name:<20} | "
              f"{load_time:<12} {bf_time:<12} {enc_time:<12} {prep_total_time:<20} | "
              f"{circ_create_time:<14} {circ_optim_time:<15} {circ_total_time:<12} | "
              f"{encrypted_eval_time:<10} {dec_time:<10} {extract_time:<12} {comp_total_time:<12} |")
    
    print("=" * 180)


# ============================================================================
# ThresholdCircuit - Core PSI Implementation
# ============================================================================

class ThresholdCircuit:
    """Threshold-PSI workflow using Bloom filters and batched OpenFHE BFV."""

    PARTY_SETS_DIR = "party_sets"
    PARTY_SETS_META_FILE = "party_sets_meta.json"
    BLOOM_FALSE_POSITIVE_RATE = 1e-4
    BFV_PLAINTEXT_MODULUS = 65537
    DEFAULT_BATCH_LANES = 2048

    def __init__(
        self,
        num_parties: int,
        threshold: int,
        num_common_ips: int = 5,
        party_set_size: int = 10**5,
    ):
        self.num_parties = num_parties
        self.threshold = threshold
        self.num_common_ips = num_common_ips
        self.party_set_size = party_set_size

        self.requested_false_positive_rate = self.BLOOM_FALSE_POSITIVE_RATE

        _template = BloomFilter(
            est_elements=party_set_size,
            false_positive_rate=self.requested_false_positive_rate,
        )

        self.false_positive_rate = self.requested_false_positive_rate
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
        """Build a stable dataset identifier from current N/T/C/S parameters."""
        return (
            f"n{self.num_parties}_t{self.threshold}_"
            f"c{self._effective_num_common_ips()}_s{self.party_set_size}"
        )

    def _dataset_dir(self) -> str:
        """Return the on-disk directory path for the current dataset tag."""
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
        """Return metadata JSON path for the active dataset directory."""
        return os.path.join(self._dataset_dir(), self.PARTY_SETS_META_FILE)

    def _current_party_set_metadata(self) -> dict:
        """Build metadata payload describing the current dataset parameters."""
        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "false_positive_rate": self.false_positive_rate,
            "num_common_ips": self._effective_num_common_ips(),
            "party_set_size": self.party_set_size,
        }

    def _write_party_set_metadata(self) -> None:
        """Write current dataset metadata JSON to disk."""
        os.makedirs(self._dataset_dir(), exist_ok=True)
        with open(self._party_set_metadata_path(), "w") as fh:
            json.dump(self._current_party_set_metadata(), fh, indent=2)

    def _read_party_set_metadata(self):
        """Read dataset metadata JSON from disk, returning None when missing."""
        path = self._party_set_metadata_path()
        if not os.path.exists(path):
            return None
        with open(path, "r") as fh:
            return json.load(fh)

    def _party_set_file_paths(self) -> list:
        """List all party data file paths for the current dataset directory."""
        dataset_dir = self._dataset_dir()
        if not os.path.exists(dataset_dir):
            return []
        files = []
        for name in os.listdir(dataset_dir):
            if name.startswith("party_") and name.endswith(".txt"):
                files.append(os.path.join(dataset_dir, name))
        return sorted(files)

    def _count_non_empty_lines(self, file_path: str) -> int:
        """Count non-empty lines in a text file."""
        with open(file_path, "r") as fh:
            return sum(1 for line in fh if line.strip())

    def _item_counts(self, party_sets: list) -> dict:
        """Count in how many parties each unique item appears."""
        counts = {}
        for party in party_sets:
            for item in set(party):
                counts[item] = counts.get(item, 0) + 1
        return counts

    def _get_party_preprocess_workers(self) -> int:
        """Resolve worker count for party-level preprocessing stages."""
        raw = os.getenv("TPSI_PARTY_PREPROCESS_WORKERS") or os.getenv("TPSI_PARALLEL_WORKERS")
        try:
            workers = int(raw) if raw is not None else 1
        except (TypeError, ValueError):
            workers = 1
        return max(1, min(workers, self.num_parties))

    def _is_party_set_cache_valid(self) -> bool:
        """Validate cached party files and metadata against current parameters."""
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
        """Remove all cached files for the current dataset directory."""
        dataset_dir = self._dataset_dir()
        if not os.path.exists(dataset_dir):
            return
        for name in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    def _read_party_sets(self):
        """Read all cached party set files, or return None if any are missing."""
        expected_files = [
            os.path.join(self._dataset_dir(), f"party_{i + 1}.txt")
            for i in range(self.num_parties)
        ]
        if not all(os.path.exists(p) for p in expected_files):
            return None

        workers = self._get_party_preprocess_workers()
        if workers <= 1 or len(expected_files) <= 1:
            return [_read_party_file(file_path) for file_path in expected_files]

        with ThreadPoolExecutor(max_workers=min(workers, len(expected_files))) as executor:
            return list(executor.map(_read_party_file, expected_files))

    def load_or_create_party_sets(self) -> list:
        """Load cached party sets when valid; otherwise generate and persist them."""
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
        """Build one Bloom-filter bit vector for each party set."""
        print(f"Building Bloom filters for '{self._dataset_tag()}' from party input files")
        workers = self._get_party_preprocess_workers()
        if workers <= 1 or len(party_sets) <= 1:
            return [self.build_bloom_filter(party_set) for party_set in party_sets]

        print(f"[PREPROCESS] Parallel Bloom construction across {min(workers, len(party_sets))} workers")
        worker_args = [
            (
                party_set,
                self.party_set_size,
                self.false_positive_rate,
                self._bits_per_chunk,
                self.num_bloom_bits,
            )
            for party_set in party_sets
        ]
        with ProcessPoolExecutor(max_workers=min(workers, len(worker_args))) as executor:
            return list(executor.map(_build_bloom_filter_bits_worker, worker_args))

    def generate_party_sets(self) -> list:
        """Generate synthetic party sets with a controlled threshold intersection."""
        party_sets = [set() for _ in range(self.num_parties)]
        used_ips: set = set()

        if self.threshold == 1:
            for party_set in party_sets:
                while len(party_set) < self.party_set_size:
                    party_set.add(self._random_unique_ip(used_ips))

            party_lists = [list(s) for s in party_sets]
            self._write_party_sets(party_lists)
            self._write_party_set_metadata()
            return party_lists

        common_ips = [
            self._random_unique_ip(used_ips) for _ in range(self.num_common_ips)
        ]
        chosen_parties = random.sample(range(self.num_parties), self.threshold)
        for ip in common_ips:
            for party_idx in chosen_parties:
                party_sets[party_idx].add(ip)

        for party_idx, party_set in enumerate(party_sets, start=1):
            while len(party_set) < self.party_set_size:
                party_set.add(self._random_unique_ip(used_ips))

        party_lists = [list(s) for s in party_sets]
        self._write_party_sets(party_lists)
        self._write_party_set_metadata()
        return party_lists

    def _time_stage(self, timings: dict, key: str, fn, *args, **kwargs):
        """Run a pipeline stage, store elapsed seconds, and return the result."""
        started_at = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[key] = time.perf_counter() - started_at
        return result

    def run_detailed_pipeline(self) -> dict:
        """Run the full TPSI pipeline and return results plus timing details."""
        timings = {
            "load_dataset_time_s": None,
            "build_bloom_filters_time_s": None,
            "input_encryption_time_s": None,
            "preprocessing_total_time_s": None,
            "circuit_creation_time_s": None,
            "circuit_optimization_time_s": None,
            "circuit_total_time_s": None,
            "plaintext_computation_time_s": None,
            "plaintext_extract_time_s": None,
            "encrypted_computation_time_s": None,
            "decryption_time_s": None,
            "encrypted_extract_time_s": None,
            "encrypted_computation_and_decryption_time_s": None,
        }

        party_sets = self._time_stage(
            timings,
            "load_dataset_time_s",
            self.load_or_create_party_sets,
        )
        bloom_filters = self._time_stage(
            timings,
            "build_bloom_filters_time_s",
            self.build_bloom_filters_from_party_sets,
            party_sets,
        )

        circuit_started_at = time.perf_counter()
        canonical_circuit = self._time_stage(
            timings,
            "circuit_creation_time_s",
            self.build_canonical_circuit,
        )
        optimized_circuit = self._time_stage(
            timings,
            "circuit_optimization_time_s",
            self.optimize_circuit,
            canonical_circuit,
        )
        timings["circuit_total_time_s"] = time.perf_counter() - circuit_started_at

        plaintext_result = self._time_stage(
            timings,
            "plaintext_computation_time_s",
            self.evaluate_plaintext_circuit,
            optimized_circuit,
            bloom_filters,
        )
        self._time_stage(
            timings,
            "plaintext_extract_time_s",
            self.extract_candidates_from_intersection_bloom,
            plaintext_result,
            party_sets,
        )

        encrypted_bloom_filters = self._time_stage(
            timings,
            "input_encryption_time_s",
            self.prepare_bloom_filters_for_openfhe,
            bloom_filters,
        )
        encrypted_result = self._time_stage(
            timings,
            "encrypted_computation_time_s",
            self.evaluate_encrypted_circuit,
            optimized_circuit,
            encrypted_bloom_filters,
        )
        if getattr(self, "_encrypted_eval_returned_plaintext", False):
            decrypted_result = encrypted_result
            timings["decryption_time_s"] = 0.0
            print("[FHE] Decryption was streamed during encrypted evaluation.")
        else:
            decrypted_result = self._time_stage(
                timings,
                "decryption_time_s",
                self.decrypt_bloom_filter,
                encrypted_result,
            )
        self._time_stage(
            timings,
            "encrypted_extract_time_s",
            self.extract_candidates_from_intersection_bloom,
            decrypted_result,
            party_sets,
        )

        encrypted_matches_plaintext = decrypted_result == plaintext_result
        print(
            "[CHECK] Encrypted result matches plaintext: "
            f"{'PASS' if encrypted_matches_plaintext else 'FAIL'}"
        )

        timings["preprocessing_total_time_s"] = (
            timings["load_dataset_time_s"]
            + timings["build_bloom_filters_time_s"]
            + timings["input_encryption_time_s"]
        )
        timings["encrypted_computation_and_decryption_time_s"] = (
            timings["encrypted_computation_time_s"]
            + timings["decryption_time_s"]
            + timings["encrypted_extract_time_s"]
        )

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
        """Build and unpack a party Bloom filter into a flat bit vector."""
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
        """Build the full canonical threshold circuit before minimization."""

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
        """Reduce the threshold circuit to positive T-combination SOP terms."""
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
        """Evaluate the optimized threshold circuit over plaintext Bloom bits."""
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
        """Return benchmark timings and correctness metrics for one configuration."""
        details = self.run_detailed_pipeline()
        timings = details["timings"]
        encrypted_match = details["encrypted_matches_plaintext"]

        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "num_common_ips": self.num_common_ips,
            "party_set_size": self.party_set_size,
            "num_bloom_bits": self.num_bloom_bits,
            "load_dataset_time_s": timings["load_dataset_time_s"],
            "build_bloom_filters_time_s": timings["build_bloom_filters_time_s"],
            "input_encryption_time_s": timings["input_encryption_time_s"],
            "preprocessing_total_time_s": timings["preprocessing_total_time_s"],
            "circuit_creation_time_s": timings["circuit_creation_time_s"],
            "circuit_optimization_time_s": timings["circuit_optimization_time_s"],
            "circuit_total_time_s": timings["circuit_total_time_s"],
            "plaintext_computation_time_s": timings["plaintext_computation_time_s"],
            "plaintext_extract_time_s": timings["plaintext_extract_time_s"],
            "encrypted_computation_time_s": timings["encrypted_computation_time_s"],
            "decryption_time_s": timings["decryption_time_s"],
            "encrypted_extract_time_s": timings["encrypted_extract_time_s"],
            "encrypted_computation_and_decryption_time_s": timings["encrypted_computation_and_decryption_time_s"],
            "encrypted_matches_plaintext": encrypted_match,
            "dataset_source": details["dataset_source"],
        }

    # ------------------------------------------------------------------
    # OpenFHE threshold BFV layer: joint key, encrypt, evaluate, decrypt
    # ------------------------------------------------------------------

    def _parse_positive_sop_terms(self, circuit: str) -> list:
        """Parse a positive SOP circuit such as 'AB + AC' into party-index terms."""
        terms = [term.strip() for term in circuit.split("+") if term.strip()]

        parsed_terms = []
        for term in terms:
            party_indices = []
            for party_label in term:
                if party_label.isalpha():
                    party_indices.append(ord(party_label.upper()) - ord("A"))
            if party_indices:
                parsed_terms.append(party_indices)

        return parsed_terms

    def _build_packed_threshold_evaluator(self, parsed_terms: list):
        """Create a slotwise OpenFHE evaluator for the threshold SOP circuit."""

        def evaluate_threshold(cc, encrypted_party_bits):
            term_values = []
            for term in parsed_terms:
                term_value = encrypted_party_bits[term[0]]
                for idx in term[1:]:
                    term_value = cc.EvalMult(term_value, encrypted_party_bits[idx])
                term_values.append(term_value)

            if not term_values:
                return encrypted_party_bits[0]

            # OR over binary slots: a OR b = a + b - ab.
            while len(term_values) > 1:
                next_level = []
                for left_idx in range(0, len(term_values), 2):
                    if left_idx + 1 >= len(term_values):
                        next_level.append(term_values[left_idx])
                        continue

                    left = term_values[left_idx]
                    right = term_values[left_idx + 1]
                    next_level.append(cc.EvalSub(cc.EvalAdd(left, right), cc.EvalMult(left, right)))
                term_values = next_level

            return term_values[0]

        return evaluate_threshold

    def _ensure_openfhe_context(self, circuit: str, batch_lanes: int) -> None:
        """Create and cache the OpenFHE context for a circuit/lane count."""
        if not hasattr(self, "_openfhe_contexts"):
            self._openfhe_contexts = {}

        cache_key = (circuit, batch_lanes)

        if cache_key in self._openfhe_contexts:
            return

        try:
            parsed_terms = self._parse_positive_sop_terms(circuit)

            batch_size = max(8, int(2 ** math.ceil(math.log2(max(1, batch_lanes)))))
            term_depth = max((len(term) - 1 for term in parsed_terms), default=1)
            or_depth = int(math.ceil(math.log2(max(1, len(parsed_terms))))) if parsed_terms else 0
            mult_depth = max(2, term_depth + or_depth + 1)

            print(
                f"[FHE] Parsed {len(parsed_terms)} positive SOP terms for encrypted evaluation "
                f"with batch lanes={batch_lanes}."
            )
            print(f"[FHE] OpenFHE parameters: bits={self.num_bloom_bits}, batch_size={batch_size}")
            print(
                f"[FHE] Circuit depth: term_depth={term_depth}, "
                f"or_depth={or_depth}, multiplicative_depth={mult_depth}"
            )

            params = CCParamsBFVRNS()
            params.SetPlaintextModulus(self.BFV_PLAINTEXT_MODULUS)
            params.SetMultiplicativeDepth(mult_depth)
            params.SetBatchSize(batch_size)

            if hasattr(params, "SetSecurityLevel"):
                params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
            if hasattr(params, "SetThresholdNumOfParties"):
                params.SetThresholdNumOfParties(self.num_parties)
            if hasattr(params, "SetMultipartyMode"):
                params.SetMultipartyMode(NOISE_FLOODING_MULTIPARTY)

            cc = GenCryptoContext(params)
            cc.Enable(PKE)
            cc.Enable(KEYSWITCH)
            cc.Enable(LEVELEDSHE)
            cc.Enable(ADVANCEDSHE)
            cc.Enable(MULTIPARTY)

            print("[FHE] Generating FHE keys...")
            keypairs = [cc.KeyGen()]
            for _ in range(1, self.num_parties):
                keypairs.append(cc.MultipartyKeyGen(keypairs[-1].publicKey))

            eval_mult_key = self._build_multiparty_eval_mult_key(cc, keypairs)
            cc.InsertEvalMultKey([eval_mult_key])

            self._openfhe_contexts[cache_key] = {
                "cc": cc,
                "keypairs": keypairs,
                "public_key": keypairs[-1].publicKey,
                "parsed_terms": parsed_terms,
                "batch_size": batch_size,
                "term_depth": term_depth,
                "or_depth": or_depth,
                "multiplicative_depth": mult_depth,
                "multiparty_enabled": True,
            }
            print("[FHE] OpenFHE circuit context is ready (multiparty).")
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialise OpenFHE FHE circuit. "
                "Ensure openfhe is installed and configured.\n"
                f"Underlying error: {exc}"
            ) from exc

    def _build_multiparty_eval_mult_key(self, cc, keypairs):
        """Build the public EvalMult key needed by encrypted AND gates."""
        eval_key = cc.KeySwitchGen(keypairs[0].secretKey, keypairs[0].secretKey)
        for party_idx in range(1, self.num_parties):
            eval_key_share = cc.MultiKeySwitchGen(
                keypairs[party_idx].secretKey,
                keypairs[party_idx].secretKey,
                eval_key,
            )
            eval_key = cc.MultiAddEvalKeys(
                eval_key,
                eval_key_share,
                keypairs[party_idx].publicKey.GetKeyTag(),
            )

        eval_mult_key = cc.MultiMultEvalKey(
            keypairs[0].secretKey,
            eval_key,
            keypairs[-1].publicKey.GetKeyTag(),
        )
        for party_idx in range(1, self.num_parties):
            eval_mult_share = cc.MultiMultEvalKey(
                keypairs[party_idx].secretKey,
                eval_key,
                keypairs[-1].publicKey.GetKeyTag(),
            )
            eval_mult_key = cc.MultiAddEvalMultKeys(
                eval_mult_key,
                eval_mult_share,
                eval_key.GetKeyTag(),
            )
        return eval_mult_key

    def _resolve_batch_lane_candidates(self, bit_count: int) -> list:
        """Return preferred batch lane sizes from fastest to safest."""
        env_value = os.getenv("TPSI_FHE_BATCH_LANES", str(self.DEFAULT_BATCH_LANES))
        try:
            requested = int(env_value)
        except (TypeError, ValueError):
            requested = 0

        if requested > 0:
            requested = max(1, min(bit_count, requested))
            fallbacks = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            return [requested] + [c for c in fallbacks if c < requested and c <= bit_count]

        candidates = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        filtered = [c for c in candidates if c <= bit_count]
        if not filtered:
            return [1]
        return filtered

    def prepare_bloom_filter_for_openfhe(self, bloom_filter_bits: list) -> np.ndarray:
        """Convert Bloom bits to int64; OpenFHE encryption happens during evaluation."""
        return np.array(bloom_filter_bits, dtype=np.int64)

    def prepare_bloom_filters_for_openfhe(self, bloom_filters: list) -> list:
        """Prepare all party Bloom filters for packed OpenFHE evaluation."""
        workers = self._get_party_preprocess_workers()
        if workers <= 1 or len(bloom_filters) <= 1:
            return [self.prepare_bloom_filter_for_openfhe(bits) for bits in bloom_filters]

        print(f"[PREPROCESS] Parallel OpenFHE input prep across {min(workers, len(bloom_filters))} workers")
        with ProcessPoolExecutor(max_workers=min(workers, len(bloom_filters))) as executor:
            return list(executor.map(_prepare_bloom_filter_bits_worker, bloom_filters))

    def encrypt_bloom_filter(self, bloom_filter_bits, context: dict, batch_lanes: int, bit_count: int) -> list:
        """Encrypt one Bloom filter into packed BFV ciphertext chunks."""
        encrypted_chunks = []

        for start in range(0, bit_count, batch_lanes):
            encrypted_chunks.append(
                self.encrypt_bloom_filter_chunk(
                    bloom_filter_bits,
                    context,
                    batch_lanes,
                    start,
                    bit_count,
                )
            )

        return encrypted_chunks

    def encrypt_bloom_filter_chunk(
        self,
        bloom_filter_bits,
        context: dict,
        batch_lanes: int,
        start: int,
        bit_count: int,
    ):
        """Encrypt one packed chunk of a party Bloom filter."""
        cc = context["cc"]
        public_key = context["public_key"]
        stop = min(start + batch_lanes, bit_count)
        span = stop - start
        packed_values = [0] * batch_lanes
        packed_values[:span] = [int(value) for value in bloom_filter_bits[start:stop]]
        plaintext = cc.MakePackedPlaintext(packed_values)
        return cc.Encrypt(public_key, plaintext)

    def encrypt_bloom_filters(self, bloom_filters: list, context: dict, batch_lanes: int, bit_count: int) -> list:
        """Encrypt every party Bloom filter using the joint BFV public key."""
        return [
            self.encrypt_bloom_filter(bloom_filter_bits, context, batch_lanes, bit_count)
            for bloom_filter_bits in bloom_filters
        ]

    def decrypt_bloom_filter(self, encrypted_chunks: list) -> list:
        """Collaboratively decrypt packed result chunks and unpack Bloom bits."""
        if not hasattr(self, "_active_openfhe_context"):
            raise RuntimeError("No active OpenFHE circuit found for decryption.")

        target_bit_count = getattr(self, "_active_encrypted_bit_count", None)
        if target_bit_count is None:
            target_bit_count = len(encrypted_chunks)
        batch_lanes = getattr(self, "_active_batch_lanes", 1)

        decrypted = []

        print(f"[FHE] Decrypting result bits ({target_bit_count} bits)...")

        for encrypted_chunk in encrypted_chunks:
            remaining = target_bit_count - len(decrypted)
            decrypted.extend(self.decrypt_bloom_filter_chunk(encrypted_chunk, min(batch_lanes, remaining)))
            if len(decrypted) >= target_bit_count:
                break

        print("[FHE] Decryption complete")
        return decrypted

    def decrypt_bloom_filter_chunk(self, encrypted_chunk, max_values: int) -> list:
        """Collaboratively decrypt one packed result chunk."""
        context = self._active_openfhe_context
        cc = context["cc"]
        keypairs = context["keypairs"]
        batch_lanes = getattr(self, "_active_batch_lanes", 1)
        debug_mode = os.getenv("TPSI_FHE_DEBUG", "0") == "1"

        partials = []
        for party_idx in range(len(keypairs)):
            if party_idx == 0:
                part = cc.MultipartyDecryptLead([encrypted_chunk], keypairs[party_idx].secretKey)
            else:
                part = cc.MultipartyDecryptMain([encrypted_chunk], keypairs[party_idx].secretKey)
            if part and len(part) > 0:
                partials.append(part[0])

        if not partials:
            return [0] * max_values

        decoded = cc.MultipartyDecryptFusion(partials)
        if hasattr(decoded, "SetLength"):
            decoded.SetLength(batch_lanes)

        bits = []
        for value in decoded.GetPackedValue():
            value = int(value)
            if debug_mode and len(bits) < 5:
                print(f"[FHE] Fused plaintext value[{len(bits)}]={value}")

            bit = 1 if (value % self.BFV_PLAINTEXT_MODULUS) == 1 else 0
            bits.append(bit)
            if len(bits) >= max_values:
                break

        return bits

    def evaluate_encrypted_circuit(self, circuit: str, encrypted_bloom_filters: list) -> list:
        """Pack Bloom positions into BFV slots and evaluate the threshold circuit."""
        if not encrypted_bloom_filters:
            return []

        bit_count = len(encrypted_bloom_filters[0])
        if bit_count == 0:
            return []

        batch_lane_candidates = self._resolve_batch_lane_candidates(bit_count)
        context = None
        batch_lanes = None

        for lanes in batch_lane_candidates:
            try:
                self._ensure_openfhe_context(circuit, lanes)
                context = self._openfhe_contexts[(circuit, lanes)]
                batch_lanes = lanes
                break
            except RuntimeError as exc:
                print(f"[FHE] Batch lanes={lanes} unavailable ({exc}). Trying smaller lanes...")

        if context is None or batch_lanes is None:
            raise RuntimeError("Failed to initialize any OpenFHE batched configuration.")

        self._active_openfhe_context = context
        self._active_encrypted_bit_count = bit_count
        self._active_batch_lanes = batch_lanes

        cc = context["cc"]
        parsed_terms = context["parsed_terms"]
        evaluate_threshold = self._build_packed_threshold_evaluator(parsed_terms)
        self._encrypted_eval_returned_plaintext = False

        chunk_count = (bit_count + batch_lanes - 1) // batch_lanes
        stream_decrypt = os.getenv("TPSI_FHE_STREAM_DECRYPT", "1") != "0"
        encrypted_result_chunks = [] if not stream_decrypt else None
        decrypted_result_bits = [] if stream_decrypt else None

        progress_every_raw = os.getenv("TPSI_FHE_PROGRESS_EVERY", "0")
        try:
            progress_every = max(0, int(progress_every_raw))
        except (TypeError, ValueError):
            progress_every = 5000

        print(
            f"[FHE][{self._dataset_tag()}][pid={os.getpid()}] "
            f"Encrypting and evaluating Bloom bits ({bit_count} bits) "
            f"with batch lanes={batch_lanes} ({chunk_count} chunks)..."
        )
        if progress_every > 0:
            print(f"[FHE][{self._dataset_tag()}] Progress logging every {progress_every} bits")
        if stream_decrypt:
            print("[FHE] Streaming collaborative decryption per evaluated chunk to reduce memory.")

        bit_loop_start = time.perf_counter()
        for chunk_idx in range(chunk_count):
            chunk_start = chunk_idx * batch_lanes
            encrypted_party_bits = [
                self.encrypt_bloom_filter_chunk(
                    encrypted_bloom_filters[party_idx],
                    context,
                    batch_lanes,
                    chunk_start,
                    bit_count,
                )
                for party_idx in range(self.num_parties)
            ]
            encrypted_result_chunk = evaluate_threshold(cc, encrypted_party_bits)
            if stream_decrypt:
                remaining = bit_count - len(decrypted_result_bits)
                decrypted_result_bits.extend(
                    self.decrypt_bloom_filter_chunk(
                        encrypted_result_chunk,
                        min(batch_lanes, remaining),
                    )
                )
            else:
                encrypted_result_chunks.append(encrypted_result_chunk)
            del encrypted_party_bits
            del encrypted_result_chunk
            if chunk_idx % 10 == 0:
                gc.collect()

            completed = min((chunk_idx + 1) * batch_lanes, bit_count)
            if progress_every > 0 and (
                completed == 1
                or completed == bit_count
                or (completed % progress_every == 0)
            ):
                elapsed = time.perf_counter() - bit_loop_start
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (bit_count - completed) / rate if rate > 0 else float("inf")
                remaining_text = f"{remaining:.1f}s" if np.isfinite(remaining) else "N/A"
                print(
                    f"[FHE][{self._dataset_tag()}] bit {completed}/{bit_count} "
                    f"({100.0 * completed / bit_count:.2f}%) | elapsed={elapsed:.1f}s | eta={remaining_text}"
                )

        print(f"[FHE][{self._dataset_tag()}] Encrypted evaluation complete")
        if stream_decrypt:
            self._encrypted_eval_returned_plaintext = True
            return decrypted_result_bits[:bit_count]

        return encrypted_result_chunks

    def extract_candidates_from_intersection_bloom(self, intersection_bits: list, party_sets: list) -> list:
        """Recover candidate IPs by checking party-set union against result Bloom bits."""
        helper = self._reconstruct_bloom_filter_from_bits(intersection_bits)

        universe = sorted({ip for party in party_sets for ip in party})
        candidates = []
        for item in universe:
            if helper.check(item):
                candidates.append(item)
        return candidates

    def exact_threshold_intersection(self, party_sets: list) -> list:
        """Compute the exact threshold intersection directly from plaintext sets."""
        counts = {}
        for party in party_sets:
            for item in set(party):
                counts[item] = counts.get(item, 0) + 1
        return sorted([item for item, count in counts.items() if count >= self.threshold])


# ============================================================================
# Utility Functions
# ============================================================================

def _expand_range_spec(range_spec):
    """Expand a range spec string/list/tuple into an inclusive integer list."""
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
    """Expand sequence config using explicit values, range spec, or defaults."""
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


def _read_party_file(file_path: str) -> list:
    """Load one party file from disk and return non-empty IP lines."""
    with open(file_path, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def _build_bloom_filter_bits_worker(args: tuple) -> list:
    """Build one party Bloom filter bit-array in a worker process."""
    party_set, party_set_size, false_positive_rate, bits_per_chunk, num_bloom_bits = args
    bloom_filter = BloomFilter(
        est_elements=int(party_set_size),
        false_positive_rate=float(false_positive_rate),
    )
    for ip in party_set:
        bloom_filter.add(ip)

    result_bits = []
    for chunk in bloom_filter.bloom:
        for bit_pos in range(int(bits_per_chunk)):
            result_bits.append((chunk >> bit_pos) & 1)
    return result_bits[: int(num_bloom_bits)]


def _prepare_bloom_filter_bits_worker(bloom_filter_bits: list) -> np.ndarray:
    """Convert one party Bloom bit array to int64 in a worker process."""
    return np.array(bloom_filter_bits, dtype=np.int64)


# ============================================================================
# CLI Helpers
# ============================================================================

def parse_config_key(config_key: str) -> dict:
    """Parse config key string (nX_tY_cZ_sW) and return validated config dict."""
    candidate = (config_key or "").strip().lower()
    match = re.fullmatch(r"n(\d+)_t(\d+)_c(\d+)_s(\d+)", candidate)
    if not match:
        raise ValueError("Invalid config format. Expected: nX_tY_cZ_sW (example: n3_t2_c5_s1000)")

    num_parties = int(match.group(1))
    threshold = int(match.group(2))
    num_common_ips = int(match.group(3))
    party_set_size = int(match.group(4))

    if num_parties < 2:
        raise ValueError("Invalid config: num_parties must be >= 2")
    if threshold < 1 or threshold > num_parties:
        raise ValueError("Invalid config: threshold must satisfy 1 <= threshold <= num_parties")
    if num_common_ips < 0:
        raise ValueError("Invalid config: num_common_ips must be >= 0")
    if party_set_size < 1:
        raise ValueError("Invalid config: party_set_size must be >= 1")

    return {
        "num_parties": num_parties,
        "threshold": threshold,
        "num_common_ips": num_common_ips,
        "party_set_size": party_set_size,
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Build command-line parser for non-interactive utility usage."""
    parser = argparse.ArgumentParser(description="Threshold PSI utility")
    parser.add_argument(
        "--generate-input",
        action="store_true",
        help="Generate/cache all input datasets from party_sets_meta.json",
    )
    parser.add_argument(
        "--verify-input",
        action="store_true",
        help="Verify all cached datasets under party_sets/",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Run one configuration using key format nX_tY_cZ_sW (example: -c n3_t2_c5_s1000)",
    )
    return parser


def run_sample_demo(config: dict) -> None:
    """Execute one full TPSI sample run and print detailed outputs."""
    tc = ThresholdCircuit(**config)
    print("\n[RUN] Starting custom/sample execution")
    print(f"[RUN] Configuration: N={tc.num_parties}, T={tc.threshold}, C={tc.num_common_ips}, S={tc.party_set_size}")
    print(f"[RUN] Bloom filter size M={tc.num_bloom_bits:,} bits, hash functions K={tc.num_hash_funcs}")
    print(
        "[RUN] False positive rate "
        f"target={tc.requested_false_positive_rate:.3e} (fixed), "
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

    print("\n--- OpenFHE Threshold BFV Encrypted Evaluation ---")
    if decrypted_result is None:
        print("[RUN] OpenFHE encrypted evaluation not available on this machine")
    else:
        print(f"[RUN] Decrypted intersection bits: {decrypted_result}")
        matches = decrypted_result == plaintext_result
        print(f"[RUN] Matches plaintext result: {matches}")
        recovered_candidates = tc.extract_candidates_from_intersection_bloom(
            decrypted_result,
            party_sets,
        )
        print(f"[RUN] Recovered candidates from decrypted bloom filter: {recovered_candidates}")

    _render_hierarchical_timing_table(
        "Timing Table - Custom Run",
        [
            {
                "run": tc._dataset_tag(),
                "dataset_source": details["dataset_source"],
                "load_dataset_time_s": timings.get("load_dataset_time_s"),
                "build_bloom_filters_time_s": timings.get("build_bloom_filters_time_s"),
                "input_encryption_time_s": timings.get("input_encryption_time_s"),
                "preprocessing_total_time_s": timings.get("preprocessing_total_time_s"),
                "circuit_creation_time_s": timings.get("circuit_creation_time_s"),
                "circuit_optimization_time_s": timings.get("circuit_optimization_time_s"),
                "circuit_total_time_s": timings.get("circuit_total_time_s"),
                "plaintext_computation_time_s": timings.get("plaintext_computation_time_s"),
                "plaintext_extract_time_s": timings.get("plaintext_extract_time_s"),
                "encrypted_computation_time_s": timings.get("encrypted_computation_time_s"),
                "decryption_time_s": timings.get("decryption_time_s"),
                "encrypted_extract_time_s": timings.get("encrypted_extract_time_s"),
                "encrypted_computation_and_decryption_time_s": timings.get("encrypted_computation_and_decryption_time_s"),
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
    """Prepare and cache all datasets required by sample and scaling runs."""
    print("\nPreparing input IP files for sample run and configured datasets")

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
    parties_threshold_values_cfg = parties_cfg.get("threshold_values")
    if parties_threshold_values_cfg is not None:
        parties_t_values = [
            int(v)
            for v in _expand_sequence_spec(parties_threshold_values_cfg, [3, 5, 7])
            if int(v) >= 1
        ]
        for t in parties_t_values:
            for n in parties_n_values:
                if t <= n:
                    configs.add((n, t, parties_common_ips, parties_set_size))
    else:
        parties_mode = str(parties_cfg.get("threshold_mode", "half"))
        parties_fixed_t = int(parties_cfg.get("threshold", 1))
        for n in parties_n_values:
            t = max(1, n // 2) if parties_mode == "half" else max(1, min(parties_fixed_t, n))
            configs.add((n, t, parties_common_ips, parties_set_size))

    threshold_cfg = sweeps_cfg.get("threshold", {}) if isinstance(sweeps_cfg, dict) else {}
    threshold_n_values = [
        int(v)
        for v in _expand_sequence_spec(threshold_cfg.get("n_values"), list(range(10, max_parties + 1)))
    ]
    threshold_t_values = [
        int(v)
        for v in _expand_sequence_spec(threshold_cfg.get("t_values"), list(range(1, max_parties + 1)))
    ]
    threshold_set_size = int(threshold_cfg.get("party_set_size", 10**4))
    threshold_common_ips = int(threshold_cfg.get("num_common_ips", 10))
    for n in threshold_n_values:
        for t in threshold_t_values:
            if 1 <= t <= n:
                configs.add((n, t, threshold_common_ips, threshold_set_size))

    set_size_cfg = sweeps_cfg.get("set_size", {}) if isinstance(sweeps_cfg, dict) else {}
    set_sizes = [int(v) for v in _expand_sequence_spec(set_size_cfg.get("set_sizes"), [10, 100, 1000, 10000, 100000, 1000000, 10000000])]
    n_for_size = int(set_size_cfg.get("num_parties", 10))
    set_size_threshold_values_cfg = set_size_cfg.get("threshold_values")
    if set_size_threshold_values_cfg is not None:
        t_values_for_size = [
            int(v)
            for v in _expand_sequence_spec(set_size_threshold_values_cfg, [3, 5, 7])
            if 1 <= int(v) <= n_for_size
        ]
    else:
        t_values_for_size = [max(1, min(int(set_size_cfg.get("threshold", 5)), n_for_size))]
    c_for_size = int(set_size_cfg.get("num_common_ips", 1))
    for t_for_size in t_values_for_size:
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
        f"parties_threshold_values={parties_cfg.get('threshold_values', 'half-mode')}, "
        f"threshold_n_values={threshold_n_values}, threshold_t_values=1..N, threshold_set_size={threshold_set_size}, "
        f"set_size_values={set_sizes}, set_size_threshold_values={set_size_cfg.get('threshold_values', [set_size_cfg.get('threshold', 5)])}, "
        f"common_n={n_for_common}, common_t={t_for_common}, common_set_size={set_size_for_common}, "
        f"common_values={common_values}"
    )

    ordered_configs = sorted(
        configs,
        key=lambda cfg: (cfg[0], cfg[1], cfg[2], cfg[3]),
        reverse=True,
    )
    cooldown_seconds = float(scaling_config.get("cooldown_seconds_between_datasets", 0.0)) if isinstance(scaling_config, dict) else 0.0

    _print_prep_preflight_stats(ordered_configs, ordered_configs, 0)

    # Heuristic workload proxy: roughly proportional to in-memory set footprint.
    workload_sizes = [n * s for (n, _t, _c, s) in ordered_configs]
    max_workload = max(workload_sizes) if workload_sizes else 0

    print(
        f"Preparing {len(ordered_configs)} dataset configurations sequentially "
        f"(largest workload proxy N*set_size={max_workload:,})"
    )

    def _prepare_one(tc: ThresholdCircuit):
        """Prepare one dataset and return (dataset_tag, elapsed_seconds)."""
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


def verify_all_cached_datasets_verbose() -> None:
    """Run detailed single-dataset verification for every cached dataset."""
    if not os.path.exists(ThresholdCircuit.PARTY_SETS_DIR):
        print(f"\nNo datasets found. Directory '{ThresholdCircuit.PARTY_SETS_DIR}' does not exist.")
        return

    dataset_names = [
        name
        for name in sorted(os.listdir(ThresholdCircuit.PARTY_SETS_DIR))
        if os.path.isdir(os.path.join(ThresholdCircuit.PARTY_SETS_DIR, name))
    ]

    if not dataset_names:
        print(f"\nNo datasets found in '{ThresholdCircuit.PARTY_SETS_DIR}'.")
        return

    print(f"\n{'='*70}")
    print(f"Detailed verification for {len(dataset_names)} cached dataset(s)")
    print(f"{'='*70}")

    for idx, dataset_name in enumerate(dataset_names, start=1):
        print(f"\n[{idx}/{len(dataset_names)}] Running detailed check for {dataset_name}")
        verify_single_dataset(dataset_name)

    print(f"\n{'='*70}")
    print("Detailed verification completed for all cached datasets")
    print(f"{'='*70}\n")


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

    config["sample_config"].pop("false_positive_rate", None)
    config["scaling_config"].pop("false_positive_rate", None)

    config["script_mode"] = "circuit_tpsi"
    return config


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    args = build_cli_parser().parse_args()

    log_file_name = "runtime.log"
    if args.config:
        normalized_config = args.config.strip().lower()
        if re.fullmatch(r"n\d+_t\d+_c\d+_s\d+", normalized_config):
            log_file_name = f"{normalized_config}_runtime.log"

    log_path = configure_output_mode(
        use_realtime_log=True,
        log_file_path=os.path.join("benchmark_outputs", log_file_name),
    )
    runtime = load_runtime_config()
    print(f"[INIT] Runtime config loaded from {os.path.join(ThresholdCircuit.PARTY_SETS_DIR, ThresholdCircuit.PARTY_SETS_META_FILE)}")
    if log_path:
        print(f"[INIT] Realtime append logging enabled at {log_path}")
    else:
        print("[INIT] Normal console logging enabled")

    did_anything = False

    if args.generate_input:
        did_anything = True
        print("\n[ACTION] Generating all configured input datasets")
        prepare_input_files(runtime["sample_config"], runtime["scaling_config"])

    if args.verify_input:
        did_anything = True
        print("\n[ACTION] Verifying all cached input datasets")
        verify_all_cached_datasets_verbose()

    if args.config:
        did_anything = True
        print(f"\n[ACTION] Running configuration: {args.config}")
        parsed_config = parse_config_key(args.config)

        tc = ThresholdCircuit(**parsed_config)
        resolved_tag = tc._dataset_tag()
        if resolved_tag != args.config.strip().lower():
            print(f"[INFO] Resolved dataset tag: {resolved_tag} (from input: {args.config})")

        tc.load_or_create_party_sets()

        verify_single_dataset(resolved_tag)
        run_sample_demo(parsed_config)

    if not did_anything:
        print("\nNo action selected. Use one or more of: --generate-input, --verify-input, -c <config>")
        print("Example: python Circuit_TPSI.py --generate-input")



