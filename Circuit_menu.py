import itertools
import random
import os
import json
import array
import time
import threading
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from probables import BloomFilter
from concrete import fhe
import matplotlib.pyplot as plt


class _LiveSpinner:
    """Render a single-line spinner for long blocking steps."""

    def __init__(self, label: str, interval: float = 0.2):
        self.label = label
        self.interval = interval
        self._start = 0.0
        self._stop_event = threading.Event()
        self._thread = None
        self._last_line_len = 0

    def _write_line(self, text: str) -> None:
        padded = text.ljust(self._last_line_len)
        self._last_line_len = len(padded)
        sys.stdout.write("\r" + padded)
        sys.stdout.flush()

    def _run(self) -> None:
        frames = "|/-\\"
        idx = 0
        while not self._stop_event.wait(self.interval):
            elapsed = time.perf_counter() - self._start
            self._write_line(f"[FHE] {self.label} {frames[idx % len(frames)]}  elapsed {elapsed:6.1f}s")
            idx += 1

    def __enter__(self):
        self._start = time.perf_counter()
        self._write_line(f"[FHE] {self.label} ...")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        elapsed = time.perf_counter() - self._start
        status = "done" if exc is None else "failed"
        self._write_line(f"[FHE] {self.label} {status}  elapsed {elapsed:6.1f}s")
        sys.stdout.write("\n")
        sys.stdout.flush()


class _LiveProgress:
    """Render throttled single-line progress for iterative encrypted steps."""

    def __init__(self, label: str, total: int, interval: float = 0.2):
        self.label = label
        self.total = max(total, 1)
        self.interval = interval
        self._start = time.perf_counter()
        self._last_render = 0.0
        self._last_line_len = 0

    def _write_line(self, text: str) -> None:
        padded = text.ljust(self._last_line_len)
        self._last_line_len = len(padded)
        sys.stdout.write("\r" + padded)
        sys.stdout.flush()

    def update(self, current: int, detail: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        if not force and current < self.total and (now - self._last_render) < self.interval:
            return

        elapsed = now - self._start
        progress = current / self.total
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 else float("inf")
        eta_text = f"{eta:6.1f}s" if eta != float("inf") else "   n/a"
        detail_text = f"  {detail}" if detail else ""
        self._write_line(
            f"[FHE] {self.label} {current}/{self.total} ({progress * 100:5.1f}%)"
            f"  elapsed {elapsed:6.1f}s  eta {eta_text}{detail_text}"
        )
        self._last_render = now

    def finish(self, detail: str = "done") -> None:
        self.update(self.total, detail=detail, force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# ThresholdCircuit
# ---------------------------------------------------------------------------

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
        Acceptable Bloom-filter false-positive probability (default 0.05 %).
    num_common_ips : int
        Number of IP addresses that will be seeded into exactly T parties.
    party_set_size : int
        Number of IPs in each party's full set.
    """

    PARTY_SETS_DIR = "party_sets_menu"
    PARTY_SETS_META_FILE = "party_sets_meta.json"
    BLOOM_META_FILE = "bloom_meta.json"

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
        self.false_positive_rate = false_positive_rate
        self.num_common_ips = num_common_ips
        self.party_set_size = party_set_size

        # Derive optimal Bloom-filter parameters via pyprobables.
        _template = BloomFilter(
            est_elements=party_set_size,
            false_positive_rate=false_positive_rate,
        )
        self.num_bloom_bits = _template.number_bits   # M
        self.num_hash_funcs = _template.number_hashes # K
        self._bits_per_chunk = int(_template._bits_per_elm)
        self._bloom_typecode = _template.bloom.typecode

    def _dataset_tag(self) -> str:
        return (
            f"n{self.num_parties}_t{self.threshold}_"
            f"c{self.num_common_ips}_s{self.party_set_size}"
        )

    def _dataset_dir(self) -> str:
        return os.path.join(self.PARTY_SETS_DIR, self._dataset_tag())

    def _bloom_cache_name(self) -> str:
        fpr_token = str(self.false_positive_rate).replace(".", "p")
        return f"bloom_fpr_{fpr_token}.npz"

    def _bloom_meta_name(self) -> str:
        fpr_token = str(self.false_positive_rate).replace(".", "p")
        return f"bloom_meta_fpr_{fpr_token}.json"

    # ------------------------------------------------------------------
    # Data layer  party set persistence
    # ------------------------------------------------------------------

    def _random_unique_ip(self, used_ips: set) -> str:
        """Return a random IPv4 address that does not appear in used_ips."""
        while True:
            ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
            if ip not in used_ips:
                used_ips.add(ip)
                return ip

    def _write_party_sets(self, party_sets: list) -> None:
        """Persist party_sets to individual text files (one IP per line)."""
        os.makedirs(self._dataset_dir(), exist_ok=True)
        for idx, party_set in enumerate(party_sets):
            file_path = os.path.join(self._dataset_dir(), f"party_{idx + 1}.txt")
            with open(file_path, "w") as fh:
                for ip in party_set:
                    fh.write(ip + "\n")

    def _party_set_metadata_path(self) -> str:
        return os.path.join(self._dataset_dir(), self.PARTY_SETS_META_FILE)

    def _current_party_set_metadata(self) -> dict:
        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "false_positive_rate": self.false_positive_rate,
            "num_common_ips": self.num_common_ips,
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

    def _read_metadata_from_dir(self, dataset_dir: str):
        meta_path = os.path.join(dataset_dir, self.PARTY_SETS_META_FILE)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r") as fh:
            return json.load(fh)

    def _read_party_sets_from_dir(self, dataset_dir: str, num_parties: int):
        expected_files = [
            os.path.join(dataset_dir, f"party_{i + 1}.txt")
            for i in range(num_parties)
        ]
        if not all(os.path.exists(path) for path in expected_files):
            return None

        party_sets = []
        for file_path in expected_files:
            with open(file_path, "r") as fh:
                party_sets.append([line.strip() for line in fh if line.strip()])
        return party_sets

    def _item_counts(self, party_sets: list) -> dict:
        counts = {}
        for party in party_sets:
            for item in set(party):
                counts[item] = counts.get(item, 0) + 1
        return counts

    def _validate_party_sets_shape(self, party_sets: list) -> bool:
        if len(party_sets) != self.num_parties:
            return False
        if any(len(party) != self.party_set_size for party in party_sets):
            return False

        counts = self._item_counts(party_sets)
        if self.threshold == 1:
            return all(count == 1 for count in counts.values())

        shared_count = sum(1 for count in counts.values() if count == self.threshold)
        allowed_counts = all(count in (1, self.threshold) for count in counts.values())
        return shared_count == self.num_common_ips and allowed_counts

    def _find_adaptation_source(self):
        if not os.path.exists(self.PARTY_SETS_DIR):
            return None

        candidates = []
        for name in os.listdir(self.PARTY_SETS_DIR):
            dataset_dir = os.path.join(self.PARTY_SETS_DIR, name)
            if not os.path.isdir(dataset_dir):
                continue
            metadata = self._read_metadata_from_dir(dataset_dir)
            if metadata is None:
                continue
            if metadata.get("num_parties") != self.num_parties:
                continue
            if metadata.get("threshold") != self.threshold:
                continue

            score = (
                abs(metadata.get("party_set_size", 0) - self.party_set_size),
                abs(metadata.get("num_common_ips", 0) - self.num_common_ips),
            )
            candidates.append((score, dataset_dir, metadata))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, dataset_dir, metadata = candidates[0]
        return dataset_dir, metadata

    def _adjust_common_ips_minimally(self, party_sets: list, used_ips: set) -> bool:
        if self.threshold == 1:
            return True

        counts = self._item_counts(party_sets)
        current_shared = sorted([item for item, count in counts.items() if count == self.threshold])

        if len(current_shared) > self.num_common_ips:
            for item in current_shared[self.num_common_ips:]:
                owners = [idx for idx, party in enumerate(party_sets) if item in party]
                for owner in owners:
                    party_sets[owner].remove(item)
                    party_sets[owner].add(self._random_unique_ip(used_ips))

        elif len(current_shared) < self.num_common_ips:
            deficit = self.num_common_ips - len(current_shared)
            for _ in range(deficit):
                shared_ip = self._random_unique_ip(used_ips)
                owners = random.sample(range(self.num_parties), self.threshold)
                for owner in owners:
                    party_sets[owner].add(shared_ip)

        return True

    def _adjust_party_size_minimally(self, party_sets: list, used_ips: set) -> bool:
        for party in party_sets:
            while len(party) < self.party_set_size:
                party.add(self._random_unique_ip(used_ips))

        for party in party_sets:
            while len(party) > self.party_set_size:
                counts = self._item_counts(party_sets)
                removable = next((item for item in sorted(party) if counts.get(item) == 1), None)
                if removable is None:
                    return False
                party.remove(removable)

        return True

    def _adapt_party_sets(self, party_sets: list) -> list:
        adapted = [set(party) for party in party_sets]
        used_ips = {item for party in adapted for item in party}

        if not self._adjust_common_ips_minimally(adapted, used_ips):
            raise RuntimeError("Unable to adjust common IPs minimally for requested parameters.")
        if not self._adjust_party_size_minimally(adapted, used_ips):
            raise RuntimeError("Unable to adjust party sizes minimally for requested parameters.")

        sorted_sets = [sorted(party) for party in adapted]
        if not self._validate_party_sets_shape(sorted_sets):
            raise RuntimeError("Adjusted input files do not satisfy the requested parameters.")
        return sorted_sets

    def _party_sets_digest(self, party_sets: list) -> str:
        hasher = hashlib.sha256()
        for party in party_sets:
            for ip in party:
                hasher.update(ip.encode("ascii"))
                hasher.update(b"\n")
            hasher.update(b"--party--\n")
        return hasher.hexdigest()

    def _bloom_cache_paths(self) -> tuple:
        dataset_dir = self._dataset_dir()
        return (
            os.path.join(dataset_dir, self._bloom_cache_name()),
            os.path.join(dataset_dir, self._bloom_meta_name()),
        )

    def _bloom_cache_metadata(self, party_sets: list) -> dict:
        return {
            "false_positive_rate": self.false_positive_rate,
            "num_bloom_bits": self.num_bloom_bits,
            "num_hash_funcs": self.num_hash_funcs,
            "party_sets_digest": self._party_sets_digest(party_sets),
        }

    def _is_bloom_cache_valid(self, party_sets: list) -> bool:
        data_path, meta_path = self._bloom_cache_paths()
        if not (os.path.exists(data_path) and os.path.exists(meta_path)):
            return False
        with open(meta_path, "r") as fh:
            metadata = json.load(fh)
        return metadata == self._bloom_cache_metadata(party_sets)

    def _load_cached_bloom_filters(self) -> list:
        data_path, _ = self._bloom_cache_paths()
        matrix = np.load(data_path, allow_pickle=False)["bloom_filters"]
        return matrix.astype(np.int64).tolist()

    def _write_bloom_cache(self, party_sets: list, bloom_filters: list) -> None:
        data_path, meta_path = self._bloom_cache_paths()
        np.savez_compressed(data_path, bloom_filters=np.array(bloom_filters, dtype=np.uint8))
        with open(meta_path, "w") as fh:
            json.dump(self._bloom_cache_metadata(party_sets), fh, indent=2)

    def _is_party_set_cache_valid(self) -> bool:
        # Validate metadata first (parameter-level compatibility check).
        metadata = self._read_party_set_metadata()
        if metadata != self._current_party_set_metadata():
            return False

        # Validate file count and expected names.
        expected_files = {
            os.path.join(self._dataset_dir(), f"party_{i + 1}.txt")
            for i in range(self.num_parties)
        }
        actual_files = set(self._party_set_file_paths())
        if actual_files != expected_files:
            return False

        # Validate each file has exactly party_set_size non-empty lines.
        for file_path in sorted(actual_files):
            with open(file_path, "r") as fh:
                line_count = sum(1 for line in fh if line.strip())
            if line_count != self.party_set_size:
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
                ips = [line.strip() for line in fh if line.strip()]
            party_sets.append(ips)
        return party_sets

    def load_or_create_party_sets(self, quiet: bool = False) -> list:
        """
        Return party sets from disk if they already exist, otherwise generate
        them, write them to disk, and return them.
        """
        if self._is_party_set_cache_valid():
            existing = self._read_party_sets()
            if existing is not None:
                if not quiet:
                    print(
                        f"Party set cache is valid for '{self._dataset_tag()}'. "
                        f"Loaded {self.num_parties} files."
                    )
                return existing

        source = self._find_adaptation_source()
        if source is not None:
            source_dir, source_meta = source
            source_sets = self._read_party_sets_from_dir(
                source_dir,
                source_meta["num_parties"],
            )
            if source_sets is not None:
                try:
                    if not quiet:
                        print(
                            f"Adapting cached input files from '{os.path.basename(source_dir)}' "
                            f"to '{self._dataset_tag()}'."
                        )
                    adapted_sets = self._adapt_party_sets(source_sets)
                    self._clear_party_set_cache()
                    self._write_party_sets(adapted_sets)
                    self._write_party_set_metadata()
                    return adapted_sets
                except RuntimeError:
                    pass

        if not quiet:
            print(
                f"Party set cache missing/mismatched for '{self._dataset_tag()}'. "
                "Regenerating from scratch..."
            )
        self._clear_party_set_cache()
        return self.generate_party_sets()

    def load_or_build_bloom_filters(self, party_sets: list, quiet: bool = False) -> list:
        """Load Bloom filters from disk when valid, otherwise build and persist them."""
        os.makedirs(self._dataset_dir(), exist_ok=True)
        if self._is_bloom_cache_valid(party_sets):
            bloom_filters = self._load_cached_bloom_filters()
            if not quiet:
                print(f"Bloom cache is valid for '{self._dataset_tag()}'. Loaded from disk.")
            return bloom_filters

        if not quiet:
            print(f"Bloom cache missing/mismatched for '{self._dataset_tag()}'. Rebuilding...")
        bloom_filters = [self.build_bloom_filter(party_set) for party_set in party_sets]
        self._write_bloom_cache(party_sets, bloom_filters)
        return bloom_filters

    def generate_party_sets(self) -> list:
        """
        Generate N party IP sets with controlled intersection.

        Process
        -------
        1. Create num_common_ips IP addresses and assign each to exactly
           threshold randomly chosen parties.
        2. Fill every party's set with unique IPs until it reaches
           party_set_size entries.
        3. Write all sets to disk.

        Returns
        -------
        list of sorted IP-address lists, one per party.
        """
        party_sets = [set() for _ in range(self.num_parties)]
        used_ips: set = set()

        # Seed common IPs into exactly T parties each.
        common_ips = [
            self._random_unique_ip(used_ips) for _ in range(self.num_common_ips)
        ]
        for ip in common_ips:
            chosen_parties = random.sample(range(self.num_parties), self.threshold)
            for party_idx in chosen_parties:
                party_sets[party_idx].add(ip)

        # Fill remaining slots with unique IPs.
        for party_idx in range(self.num_parties):
            while len(party_sets[party_idx]) < self.party_set_size:
                party_sets[party_idx].add(self._random_unique_ip(used_ips))

        sorted_sets = [sorted(s) for s in party_sets]
        self._write_party_sets(sorted_sets)
        self._write_party_set_metadata()
        return sorted_sets

    # ------------------------------------------------------------------
    # Bloom filter layer
    # ------------------------------------------------------------------

    def build_bloom_filter(self, party_set: list) -> list:
        """
        Build a Bloom filter for party_set and return its contents as a flat
        list of num_bloom_bits individual bits (each 0 or 1).

        pyprobables stores the bit array as packed 32-bit integers; we unpack
        them so that each position maps to a single bit, which is the format
        required by both plaintext and encrypted circuit evaluation.
        """
        bf = BloomFilter(
            est_elements=self.party_set_size,
            false_positive_rate=self.false_positive_rate,
        )
        for ip in party_set:
            bf.add(ip)

        # Unpack each chunk into individual bits using pyprobables' chunk width.
        bits = []
        for chunk in bf.bloom:
            for bit_pos in range(self._bits_per_chunk):
                bits.append((chunk >> bit_pos) & 1)
        return bits[: self.num_bloom_bits]

    def _reconstruct_bloom_filter_from_bits(self, bits: list) -> BloomFilter:
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
                if bit_idx < len(bits) and bits[bit_idx] == 1:
                    chunk_val |= (1 << bit_pos)
            chunks.append(chunk_val)

        helper._bloom = array.array(self._bloom_typecode, chunks)
        return helper

    # ------------------------------------------------------------------
    # Circuit layer  build, optimise, evaluate plaintext
    # ------------------------------------------------------------------

    def build_canonical_circuit(self) -> str:
        """
        Build the canonical threshold circuit for num_parties parties and threshold T.

        Includes every minterm where at least T party literals are positive,
        preserving a fixed alphabetical variable order.

        Example (N=3, T=2): AB!C + A!BC + !ABC + ABC

        Returns
        -------
        str  Sum-of-Products string in compact notation (e.g. AB!C).
        """
        if not (1 <= self.threshold <= self.num_parties):
            raise ValueError("threshold T must satisfy 1 <= T <= N")

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
        Minimise a threshold circuit to its compact SOP form.

        For a T-of-N threshold function the minimum SOP consists of all
        C(N, T) combinations of exactly T positive literals  no negations
        remain. This exploits the symmetric structure of threshold functions.

        Example: AB!C + A!BC + !ABC + ABC  ->  AB + AC + BC

        Parameters
        ----------
        circuit : str  Canonical SOP string from build_canonical_circuit().

        Returns
        -------
        str  Optimised SOP string containing only positive literals.
        """
        terms = [t.strip() for t in circuit.split("+") if t.strip()]
        if not terms:
            return ""

        # Collect all variable names.
        variables = sorted({ch for term in terms for ch in term if ch.isalpha()})

        # Threshold = minimum number of positive literals in any single term.
        threshold = min(
            sum(
                1
                for pos, ch in enumerate(term)
                if ch.isalpha() and (pos == 0 or term[pos - 1] != "!")
            )
            for term in terms
        )

        optimized_terms = [
            "".join(combo) for combo in itertools.combinations(variables, threshold)
        ]
        return " + ".join(optimized_terms)

    def evaluate_plaintext_circuit(
        self, circuit: str, bloom_filters: list
    ) -> list:
        """
        Evaluate the optimised circuit over plaintext Bloom-filter bit arrays.

        For each bit position the circuit AND/OR logic is applied across the
        parties' bits. Used for testing and validation against the encrypted path.

        Parameters
        ----------
        circuit : str
            Optimised SOP circuit (only positive literals), e.g. AB + AC + BC.
        bloom_filters : list of lists
            Raw bit arrays from build_bloom_filter(), one per party in
            alphabetical order (index 0 = Party A, 1 = Party B, ...).

        Returns
        -------
        list of int  Result bit array of length num_bloom_bits.
        """
        terms = [t.strip() for t in circuit.split("+") if t.strip()]
        if not terms:
            return [0] * self.num_bloom_bits

        bit_count = len(bloom_filters[0]) if bloom_filters else self.num_bloom_bits
        results = []
        for bit_idx in range(bit_count):
            output_bit = 0
            for term in terms:
                term_bit = 1
                for ch in term:
                    if ch.isalpha():
                        party_idx = ord(ch.upper()) - ord("A")
                        term_bit &= bloom_filters[party_idx][bit_idx]
                output_bit |= term_bit
            results.append(output_bit)

        return results



    def benchmark_computation_times_from_party_sets(
        self,
        sample_bits_cap: int = 100_000,
        include_encrypted: bool = True,
        encrypted_sample_bits_cap: int = 256,
        quiet: bool = False,
    ) -> dict:
        """
        Benchmark computation time using real generated/loaded party sets.

        This path captures effects from data-generation parameters like
        num_common_ips, unlike synthetic random bit generation.
        """
        prep_start = time.perf_counter()
        party_sets = self.load_or_create_party_sets(quiet=quiet)
        bloom_filters_full = self.load_or_build_bloom_filters(party_sets, quiet=quiet)
        prep_elapsed = time.perf_counter() - prep_start

        sampled_bits = min(self.num_bloom_bits, sample_bits_cap)
        scale_factor = self.num_bloom_bits / sampled_bits
        bloom_filters = [bf[:sampled_bits] for bf in bloom_filters_full]

        canonical_circuit = self.build_canonical_circuit()
        optimized_circuit = self.optimize_circuit(canonical_circuit)

        start_plain = time.perf_counter()
        plain_result = self.evaluate_plaintext_circuit(optimized_circuit, bloom_filters)
        plain_elapsed = time.perf_counter() - start_plain

        encrypted_elapsed = None
        encrypted_match = None
        encrypted_sampled_bits = min(sampled_bits, encrypted_sample_bits_cap)
        encrypted_scale_factor = self.num_bloom_bits / encrypted_sampled_bits
        if include_encrypted:
            try:
                start_enc = time.perf_counter()
                encrypted_bloom_filters = [
                    self.encrypt_bloom_filter(bits[:encrypted_sampled_bits])
                    for bits in bloom_filters
                ]
                enc_result = self.evaluate_encrypted_circuit(
                    optimized_circuit,
                    encrypted_bloom_filters,
                )
                dec_result = self.decrypt_bloom_filter(enc_result)
                encrypted_elapsed = time.perf_counter() - start_enc
                encrypted_match = (dec_result == plain_result[:encrypted_sampled_bits])
            except RuntimeError:
                encrypted_elapsed = None
                encrypted_match = None

        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "num_common_ips": self.num_common_ips,
            "party_set_size": self.party_set_size,
            "num_bloom_bits": self.num_bloom_bits,
            "sampled_bits": sampled_bits,
            "scale_factor": scale_factor,
            "encrypted_sampled_bits": encrypted_sampled_bits,
            "encrypted_scale_factor": encrypted_scale_factor,
            "preprocessing_time_s": prep_elapsed,
            "plaintext_time_sampled_s": plain_elapsed,
            "plaintext_time_with_preprocessing_s": prep_elapsed + plain_elapsed,
            "plaintext_time_estimated_full_s": plain_elapsed * scale_factor,
            "encrypted_time_sampled_s": encrypted_elapsed,
            "encrypted_time_with_preprocessing_s": (
                None if encrypted_elapsed is None else prep_elapsed + encrypted_elapsed
            ),
            "encrypted_time_estimated_full_s": (
                None if encrypted_elapsed is None else encrypted_elapsed * encrypted_scale_factor
            ),
            "encrypted_matches_plaintext": encrypted_match,
        }

    # ------------------------------------------------------------------
    # Concrete-python FHE layer  compile, encrypt, decrypt, evaluate
    # ------------------------------------------------------------------

    def _parse_positive_sop_terms(self, circuit: str) -> list:
        """Parse an optimised positive-literal SOP string into party-index terms."""
        terms = [t.strip() for t in circuit.split("+") if t.strip()]
        parsed = []
        for term in terms:
            parsed.append([ord(ch.upper()) - ord("A") for ch in term if ch.isalpha()])
        return parsed

    def _build_concrete_threshold_function(self, parsed_terms: list):
        """Create a concrete-python compiled function for one bit position."""

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
        Lazily compile and keygen a concrete-python bit-circuit for this SOP.
        """
        if not hasattr(self, "_compiled_circuits"):
            self._compiled_circuits = {}

        if circuit in self._compiled_circuits:
            return

        try:
            parsed_terms = self._parse_positive_sop_terms(circuit)
            compiler = self._build_concrete_threshold_function(parsed_terms)
            print(f"[FHE] Parsed {len(parsed_terms)} positive SOP terms for encrypted evaluation.")

            # Two boundary vectors are sufficient: inputs are strictly binary [0,1].
            inputset = [
                np.array([0] * self.num_parties, dtype=np.int64),
                np.array([1] * self.num_parties, dtype=np.int64),
            ]

            with _LiveSpinner("Compiling Concrete circuit"):
                compiled = compiler.compile(inputset)
            with _LiveSpinner("Generating FHE keys"):
                compiled.keygen()
            self._compiled_circuits[circuit] = compiled
            print("[FHE] Concrete circuit context is ready.")
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialise concrete-python FHE circuit. "
                "Ensure concrete-python is installed and configured.\n"
                f"Underlying error: {exc}"
            ) from exc

    def encrypt_bloom_filter(self, bits: list) -> np.ndarray:
        """
        Prepare a Bloom-filter bit array for concrete-python encrypted evaluation.

        Parameters
        ----------
        bits : list of int  Plain bit array from build_bloom_filter().

        Returns
        -------
        np.ndarray  Integer bit vector consumed by concrete-python encrypt() calls.
        """
        return np.array(bits, dtype=np.int64)

    def decrypt_bloom_filter(self, ciphertext: list) -> list:
        """
        Decrypt a concrete-python encrypted bit-result list.

        Parameters
        ----------
        ciphertext : list  Encrypted per-bit outputs returned by evaluate_encrypted_circuit.

        Returns
        -------
        list of int  Decrypted bit array.
        """
        if not hasattr(self, "_active_compiled_circuit"):
            raise RuntimeError("No active concrete-python circuit found for decryption.")

        compiled = self._active_compiled_circuit
        progress = _LiveProgress("Decrypting result bits", len(ciphertext))
        decrypted = [0] * len(ciphertext)
        completed = 0

        def _decrypt_one(indexed_item):
            idx, enc_bit = indexed_item
            return idx, int(compiled.decrypt(enc_bit))

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            future_list = [pool.submit(_decrypt_one, item) for item in enumerate(ciphertext)]
            for future in as_completed(future_list):
                idx, value = future.result()
                decrypted[idx] = value
                completed += 1
                progress.update(completed)

        progress.finish("decryption complete")
        return decrypted

    def evaluate_encrypted_circuit(
        self,
        circuit: str,
        encrypted_bloom_filters: list,
    ) -> list:
        """
        Evaluate the optimised boolean circuit over encrypted Bloom filters
        using concrete-python over one bit position at a time.

        Parameters
        ----------
        circuit : str
            Optimised SOP circuit (positive literals only), e.g. AB + AC + BC.
        encrypted_bloom_filters : list of np.ndarray
            Party bit arrays, one per party in alphabetical order.

        Returns
        -------
        list
            Encrypted per-bit outputs for the threshold intersection.
        """
        self._init_fhe_context(circuit)
        compiled = self._compiled_circuits[circuit]
        self._active_compiled_circuit = compiled

        if not encrypted_bloom_filters:
            return []

        bit_count = len(encrypted_bloom_filters[0])
        if bit_count == 0:
            return []

        results = [None] * bit_count
        progress = _LiveProgress("Encrypting and evaluating Bloom bits", bit_count)
        completed = 0

        def _eval_bit(bit_idx: int):
            bit_inputs = np.array(
                [int(encrypted_bloom_filters[p][bit_idx]) for p in range(self.num_parties)],
                dtype=np.int64,
            )
            return compiled.run(compiled.encrypt(bit_inputs))

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            future_to_idx = {pool.submit(_eval_bit, i): i for i in range(bit_count)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
                completed += 1
                progress.update(completed)

        progress.finish("encrypted evaluation complete")
        return results

    def extract_candidates_from_intersection_bloom(
        self, intersection_bits: list, party_sets: list
    ) -> list:
        """
        Recover candidate elements from a decrypted intersection Bloom filter.

        This checks every candidate in the union of party sets against the
        decrypted intersection bloom bits using the same Bloom hash mapping.
        Because Bloom filters are probabilistic, the output can include
        false positives.
        """
        helper = self._reconstruct_bloom_filter_from_bits(intersection_bits)

        universe = sorted({ip for party in party_sets for ip in party})
        candidates = []
        for item in universe:
            if helper.check(item):
                candidates.append(item)
        return candidates

    def exact_threshold_intersection(self, party_sets: list) -> list:
        """Compute exact threshold intersection from plaintext party sets."""
        counts = {}
        for party in party_sets:
            for item in set(party):
                counts[item] = counts.get(item, 0) + 1
        return sorted([item for item, count in counts.items() if count >= self.threshold])


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
    plt.ylabel("Estimated full computation time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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


def run_scaling_experiments(
    false_positive_rate: float = 0.0005,
    max_parties: int = 10,
    sample_bits_cap: int = 100_000,
    output_dir: str = "benchmark_outputs",
    scaling_sweeps: dict = None,
):
    """
    Run scaling sweeps and save plots:
    1) parties sweep (2..max_parties)
    2) threshold sweep (1..N)
    3) party set size sweep (10..10^7)
    4) common IPs sweep (fixed N, T, set size)
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

    # ------------------------------------------------------------------
    # 1) Parties sweep
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    parties_x, parties_plain, parties_enc = [], [], []
    parties_cfg = default_sweeps["parties"]
    parties_n_values = [int(v) for v in _expand_sequence_spec(parties_cfg.get("n_values"), list(range(2, max_parties + 1)))]
    fixed_set_size_for_parties = int(parties_cfg.get("party_set_size", 10**3))
    parties_common_ips = int(parties_cfg.get("num_common_ips", 10))
    threshold_mode = str(parties_cfg.get("threshold_mode", "half"))
    _w = len(str(max(parties_n_values) if parties_n_values else max_parties))
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
        # Print detailed parameters and calculated values
        print(f"\n      [{idx}/{len(parties_n_values)}] Experiment: N={n}, T={t}")
        print(f"        Parameters: party_set_size={fixed_set_size_for_parties}, num_common_ips={parties_common_ips}, FPR={false_positive_rate}")
        print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
        print(f"        Running benchmark...", end='', flush=True)
        _exp_start = time.perf_counter()
        metrics = tc.benchmark_computation_times_from_party_sets(sample_bits_cap=sample_bits_cap, include_encrypted=True, quiet=True)
        _exp_time = time.perf_counter() - _exp_start
        parties_x.append(n)
        parties_plain.append(metrics["plaintext_time_estimated_full_s"])
        parties_enc.append(metrics["encrypted_time_estimated_full_s"])
        print(f" done ({_exp_time:.2f}s)  |  plaintext: {metrics['plaintext_time_estimated_full_s']:.3f}s, encrypted: {metrics['encrypted_time_estimated_full_s']:.3f}s")
    print(f"\n[1/4] Parties sweep  done  ({time.perf_counter()-_t0:.1f}s)")
    _plot_scaling_graph(
        parties_x, parties_plain, parties_enc,
        title="Computation Time vs Number of Parties",
        x_label="Number of parties (N)",
        output_path=os.path.join(output_dir, "scaling_num_parties.png"),
    )

    # ------------------------------------------------------------------
    # 2) Threshold sweep
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    threshold_x, threshold_plain, threshold_enc = [], [], []
    threshold_cfg = default_sweeps["threshold"]
    fixed_set_size_for_threshold = int(threshold_cfg.get("party_set_size", 10**3))
    n_fixed = int(threshold_cfg.get("n_fixed", max_parties))
    threshold_common_ips = int(threshold_cfg.get("num_common_ips", 10))
    t_values = [int(v) for v in _expand_sequence_spec(threshold_cfg.get("t_values"), list(range(1, n_fixed + 1)))]
    t_values = [t for t in t_values if 1 <= t <= n_fixed]
    _w = len(str(n_fixed))
    print(f"[2/4] Threshold sweep  T values: {t_values}  N={n_fixed}  (real IP files -- disk I/O per step)")
    for idx, t in enumerate(t_values, 1):
        tc = ThresholdCircuit(
            num_parties=n_fixed,
            threshold=t,
            false_positive_rate=false_positive_rate,
            num_common_ips=threshold_common_ips,
            party_set_size=fixed_set_size_for_threshold,
        )
        # Print detailed parameters and calculated values
        print(f"\n      [{idx}/{len(t_values)}] Experiment: N={n_fixed}, T={t}")
        print(f"        Parameters: party_set_size={fixed_set_size_for_threshold}, num_common_ips={threshold_common_ips}, FPR={false_positive_rate}")
        print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
        print(f"        Running benchmark...", end='', flush=True)
        _exp_start = time.perf_counter()
        metrics = tc.benchmark_computation_times_from_party_sets(sample_bits_cap=sample_bits_cap, include_encrypted=True, quiet=True)
        _exp_time = time.perf_counter() - _exp_start
        threshold_x.append(t)
        threshold_plain.append(metrics["plaintext_time_estimated_full_s"])
        threshold_enc.append(metrics["encrypted_time_estimated_full_s"])
        print(f" done ({_exp_time:.2f}s)  |  plaintext: {metrics['plaintext_time_estimated_full_s']:.3f}s, encrypted: {metrics['encrypted_time_estimated_full_s']:.3f}s")
    print(f"\n[2/4] Threshold sweep  done  ({time.perf_counter()-_t0:.1f}s)")
    _plot_scaling_graph(
        threshold_x, threshold_plain, threshold_enc,
        title=f"Computation Time vs Threshold (N={n_fixed})",
        x_label="Threshold (T)",
        output_path=os.path.join(output_dir, "scaling_threshold.png"),
    )

    # ------------------------------------------------------------------
    # 3) Set-size sweep
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    size_cfg = default_sweeps["set_size"]
    size_x = [int(v) for v in _expand_sequence_spec(size_cfg.get("set_sizes"), [10, 100, 1000])]
    size_plain, size_enc = [], []
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
        # Print detailed parameters and calculated values
        print(f"\n      [{i}/{len(size_x)}] Experiment: N={n_for_size}, T={t_for_size}, party_set_size={set_size:,}")
        print(f"        Parameters: num_common_ips={c_for_size}, FPR={false_positive_rate}")
        print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
        print(f"        Running benchmark...", end='', flush=True)
        _exp_start = time.perf_counter()
        metrics = tc.benchmark_computation_times_from_party_sets(sample_bits_cap=sample_bits_cap, include_encrypted=True, quiet=True)
        _exp_time = time.perf_counter() - _exp_start
        size_plain.append(metrics["plaintext_time_estimated_full_s"])
        size_enc.append(metrics["encrypted_time_estimated_full_s"])
        print(f" done ({_exp_time:.2f}s)  |  plaintext: {metrics['plaintext_time_estimated_full_s']:.3f}s, encrypted: {metrics['encrypted_time_estimated_full_s']:.3f}s")
    print(f"\n[3/4] Set-size sweep  done  ({time.perf_counter()-_t0:.1f}s)")
    _plot_scaling_graph(
        size_x, size_plain, size_enc,
        title=f"Computation Time vs Party Set Size (N={n_for_size}, T={t_for_size})",
        x_label="Party set size",
        output_path=os.path.join(output_dir, "scaling_party_set_size.png"),
    )

    # ------------------------------------------------------------------
    # 4) Common-IPs sweep
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    common_cfg = default_sweeps["common_ips"]
    common_x = [int(v) for v in _expand_sequence_spec(common_cfg.get("common_values"), [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])]
    common_plain, common_enc = [], []
    n_for_common = int(common_cfg.get("num_parties", 10))
    t_for_common = int(common_cfg.get("threshold", 5))
    set_size_for_common = int(common_cfg.get("party_set_size", 10**3))
    print(f"[4/4] Common-IPs sweep  N={n_for_common}  T={t_for_common}  set_size={set_size_for_common:,}")
    print( "      (party files are written to disk and regenerated whenever params change)")
    for i, common_ips in enumerate(common_x, 1):
        tc = ThresholdCircuit(
            num_parties=n_for_common,
            threshold=t_for_common,
            false_positive_rate=false_positive_rate,
            num_common_ips=common_ips,
            party_set_size=set_size_for_common,
        )
        # Print detailed parameters and calculated values
        print(f"\n      [{i}/{len(common_x)}] Experiment: N={n_for_common}, T={t_for_common}, num_common_ips={common_ips}")
        print(f"        Parameters: party_set_size={set_size_for_common:,}, FPR={false_positive_rate}")
        print(f"        Calculated: M={tc.num_bloom_bits:,} bits, K={tc.num_hash_funcs} hash functions")
        print(f"        Running benchmark...", end='', flush=True)
        _exp_start = time.perf_counter()
        metrics = tc.benchmark_computation_times_from_party_sets(
            sample_bits_cap=sample_bits_cap,
            include_encrypted=True,
            quiet=True,
        )
        _exp_time = time.perf_counter() - _exp_start
        common_plain.append(metrics["plaintext_time_estimated_full_s"])
        common_enc.append(metrics["encrypted_time_estimated_full_s"])
        print(f" done ({_exp_time:.2f}s)  |  plaintext: {metrics['plaintext_time_estimated_full_s']:.3f}s, encrypted: {metrics['encrypted_time_estimated_full_s']:.3f}s")
    print(f"\n[4/4] Common-IPs sweep  done  ({time.perf_counter()-_t0:.1f}s)")
    _plot_scaling_graph(
        common_x, common_plain, common_enc,
        title=(
            "Computation Time vs Number of Common IPs "
            f"(N={n_for_common}, T={t_for_common}, SetSize={set_size_for_common})"
        ),
        x_label="Number of common IPs",
        output_path=os.path.join(output_dir, "scaling_num_common_ips.png"),
    )

    print(f"\nAll 4 sweeps done in {time.perf_counter()-_all_start:.1f}s  --  plots saved in '{output_dir}'")


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
    false_positive_rate = _prompt_numbered_choice(
        "Choose false positive rate",
        [0.01, 0.005, 0.001, 0.0005],
        default_index=3,
    )

    return {
        "num_parties": num_parties,
        "threshold": threshold,
        "false_positive_rate": false_positive_rate,
        "num_common_ips": num_common_ips,
        "party_set_size": party_set_size,
    }


def run_sample_demo(config: dict) -> None:
    tc = ThresholdCircuit(**config)
    print(f"\nDataset tag            = {tc._dataset_tag()}")
    print(f"Bloom filter size      M = {tc.num_bloom_bits:,} bits")
    print(f"Number of hash funcs   K = {tc.num_hash_funcs}")

    party_sets = tc.load_or_create_party_sets()
    bloom_filters = tc.load_or_build_bloom_filters(party_sets)
    print(f"\nBuilt {len(bloom_filters)} Bloom filters  "
          f"({tc.num_bloom_bits} bits each, K={tc.num_hash_funcs} hash functions, "
          f"FPR={tc.false_positive_rate*100:.2f}%)")

    canonical_circuit = tc.build_canonical_circuit()
    print(f"\nCanonical circuit: {canonical_circuit}")

    optimized_circuit = tc.optimize_circuit(canonical_circuit)
    print(f"Optimized circuit: {optimized_circuit}")

    plaintext_result = tc.evaluate_plaintext_circuit(optimized_circuit, bloom_filters)
    ones_count = sum(plaintext_result)
    print(f"\nIntersection Bloom filter: {len(plaintext_result)} bits,  "
          f"{ones_count} set ({100*ones_count/len(plaintext_result):.1f}% density)")

    recovered_from_plain = tc.extract_candidates_from_intersection_bloom(
        plaintext_result,
        party_sets,
    )
    exact_elements = tc.exact_threshold_intersection(party_sets)
    print(f"Recovered intersection candidates from plaintext bloom result: {recovered_from_plain}")
    print(f"Exact threshold intersection from plaintext sets: {exact_elements}")

    print("\n--- Concrete-Python Encrypted Evaluation ---")
    try:
        encrypted_bloom_filters = [tc.encrypt_bloom_filter(bits) for bits in bloom_filters]
        encrypted_result = tc.evaluate_encrypted_circuit(
            optimized_circuit,
            encrypted_bloom_filters,
        )
        decrypted_result = tc.decrypt_bloom_filter(encrypted_result)
        print(f"Decrypted result (first 30 bits): {decrypted_result[:30]}")
        matches = decrypted_result == plaintext_result
        print(f"Matches plaintext result: {matches}")
        recovered_candidates = tc.extract_candidates_from_intersection_bloom(
            decrypted_result,
            party_sets,
        )
        print(f"Recovered candidates from decrypted bloom filter: {recovered_candidates}")
    except RuntimeError as exc:
        print(f"Concrete-python FHE unavailable on this machine: {exc}")


def prepare_input_files(sample_config: dict, scaling_config: dict) -> None:
    print("\nPreparing cached input files for sample run and scaling experiments...")

    sweeps_cfg = scaling_config.get("scaling_sweeps", {}) if isinstance(scaling_config, dict) else {}
    max_parties = int(scaling_config.get("max_parties", 10)) if isinstance(scaling_config, dict) else 10
    false_positive_rate = float(
        scaling_config.get("false_positive_rate", sample_config["false_positive_rate"])
    ) if isinstance(scaling_config, dict) else float(sample_config["false_positive_rate"])

    configs = {
        (
            int(sample_config["num_parties"]),
            int(sample_config["threshold"]),
            float(sample_config["false_positive_rate"]),
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
        configs.add((n, t, false_positive_rate, parties_common_ips, parties_set_size))

    threshold_cfg = sweeps_cfg.get("threshold", {}) if isinstance(sweeps_cfg, dict) else {}
    n_fixed = int(threshold_cfg.get("n_fixed", max_parties))
    t_values = [int(v) for v in _expand_sequence_spec(threshold_cfg.get("t_values"), list(range(1, n_fixed + 1)))]
    threshold_set_size = int(threshold_cfg.get("party_set_size", 10**3))
    threshold_common_ips = int(threshold_cfg.get("num_common_ips", 10))
    for t in t_values:
        if 1 <= t <= n_fixed:
            configs.add((n_fixed, t, false_positive_rate, threshold_common_ips, threshold_set_size))

    set_size_cfg = sweeps_cfg.get("set_size", {}) if isinstance(sweeps_cfg, dict) else {}
    set_sizes = [int(v) for v in _expand_sequence_spec(set_size_cfg.get("set_sizes"), [10, 100, 1000])]
    n_for_size = int(set_size_cfg.get("num_parties", 10))
    t_for_size = int(set_size_cfg.get("threshold", 5))
    c_for_size = int(set_size_cfg.get("num_common_ips", 1))
    for set_size in set_sizes:
        configs.add((n_for_size, t_for_size, false_positive_rate, c_for_size, set_size))

    common_cfg = sweeps_cfg.get("common_ips", {}) if isinstance(sweeps_cfg, dict) else {}
    common_values = [int(v) for v in _expand_sequence_spec(common_cfg.get("common_values"), [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])]
    n_for_common = int(common_cfg.get("num_parties", 10))
    t_for_common = int(common_cfg.get("threshold", 5))
    set_size_for_common = int(common_cfg.get("party_set_size", 10**3))
    for common_ips in common_values:
        configs.add((n_for_common, t_for_common, false_positive_rate, common_ips, set_size_for_common))

    ordered_configs = sorted(configs)
    for idx, config in enumerate(ordered_configs, start=1):
        num_parties, threshold, fpr, num_common_ips, party_set_size = config
        tc = ThresholdCircuit(
            num_parties=num_parties,
            threshold=threshold,
            false_positive_rate=fpr,
            num_common_ips=num_common_ips,
            party_set_size=party_set_size,
        )
        print(f"[{idx}/{len(ordered_configs)}] Preparing {tc._dataset_tag()}...")
        party_sets = tc.load_or_create_party_sets(quiet=True)
        tc.load_or_build_bloom_filters(party_sets, quiet=True)

    print("All requested input files and Bloom caches are ready.")


def run_menu(sample_config: dict, scaling_config: dict) -> None:
    while True:
        print("\nThreshold PSI Mini Program")
        print("1) Generate input files")
        print("2) Run sample experiment")
        print("3) Run scaling experiments")
        print("4) Exit")

        choice = input("Select option: ").strip()
        if choice == "1":
            prepare_input_files(sample_config, scaling_config)
        elif choice == "2":
            chosen_config = prompt_sample_parameters(sample_config)
            run_sample_demo(chosen_config)
        elif choice == "3":
            print("\n--- Scaling Experiments ---")
            run_scaling_experiments(
                false_positive_rate=float(
                    scaling_config.get("false_positive_rate", sample_config["false_positive_rate"])
                ),
                max_parties=int(scaling_config.get("max_parties", 10)),
                sample_bits_cap=int(scaling_config.get("sample_bits_cap", 100_000)),
                output_dir=str(scaling_config.get("output_dir", "benchmark_outputs")),
                scaling_sweeps=scaling_config.get("scaling_sweeps"),
            )
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_menu_runtime_config() -> dict:
    defaults = {
        "script_mode": "circuit_menu",
        "sample_config": {
            "num_parties": 3,
            "threshold": 2,
            "false_positive_rate": 0.0005,
            "num_common_ips": 5,
            "party_set_size": 10**2,
        },
        "scaling_config": {
            "false_positive_rate": 0.0005,
            "max_parties": 10,
            "sample_bits_cap": 100_000,
            "output_dir": "benchmark_outputs",
            "scaling_sweeps": {
                "parties": {
                    "n_values": {"range": [2, 10]},
                    "threshold_mode": "half",
                    "party_set_size": 1000,
                    "num_common_ips": 10
                },
                "threshold": {
                    "n_fixed": 10,
                    "t_values": {"range": "1..10"},
                    "party_set_size": 1000,
                    "num_common_ips": 10
                },
                "set_size": {
                    "set_sizes": [10, 100, 1000],
                    "num_parties": 10,
                    "threshold": 5,
                    "num_common_ips": 1
                },
                "common_ips": {
                    "common_values": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                    "num_parties": 10,
                    "threshold": 5,
                    "party_set_size": 1000
                }
            }
        },
    }
    os.makedirs(ThresholdCircuit.PARTY_SETS_DIR, exist_ok=True)
    config_path = os.path.join(
        ThresholdCircuit.PARTY_SETS_DIR,
        ThresholdCircuit.PARTY_SETS_META_FILE,
    )

    config = defaults.copy()
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            existing = json.load(fh)
        if isinstance(existing, dict):
            config.update(existing)

    sample_config = defaults["sample_config"].copy()
    sample_config.update(config.get("sample_config", {}))
    config["sample_config"] = sample_config

    scaling_config = defaults["scaling_config"].copy()
    scaling_config.update(config.get("scaling_config", {}))
    config["scaling_config"] = scaling_config

    config["script_mode"] = "circuit_menu"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    return config

if __name__ == "__main__":
    runtime = load_menu_runtime_config()
    run_menu(runtime["sample_config"], runtime["scaling_config"])
