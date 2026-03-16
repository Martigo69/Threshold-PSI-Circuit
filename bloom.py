import itertools
import random
import os
import json
import array
import time

import numpy as np
from probables import BloomFilter
try:
    from concrete import fhe
    _FHE_AVAILABLE = True
except ModuleNotFoundError:
    fhe = None  # type: ignore
    _FHE_AVAILABLE = False
import matplotlib.pyplot as plt


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
        os.makedirs(self.PARTY_SETS_DIR, exist_ok=True)
        for idx, party_set in enumerate(party_sets):
            file_path = os.path.join(self.PARTY_SETS_DIR, f"party_{idx + 1}.txt")
            with open(file_path, "w") as fh:
                for ip in party_set:
                    fh.write(ip + "\n")

    def _party_set_metadata_path(self) -> str:
        return os.path.join(self.PARTY_SETS_DIR, self.PARTY_SETS_META_FILE)

    def _current_party_set_metadata(self) -> dict:
        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "false_positive_rate": self.false_positive_rate,
            "num_common_ips": self.num_common_ips,
            "party_set_size": self.party_set_size,
        }

    def _write_party_set_metadata(self) -> None:
        os.makedirs(self.PARTY_SETS_DIR, exist_ok=True)
        with open(self._party_set_metadata_path(), "w") as fh:
            json.dump(self._current_party_set_metadata(), fh, indent=2)

    def _read_party_set_metadata(self):
        path = self._party_set_metadata_path()
        if not os.path.exists(path):
            return None
        with open(path, "r") as fh:
            return json.load(fh)

    def _party_set_file_paths(self) -> list:
        if not os.path.exists(self.PARTY_SETS_DIR):
            return []
        files = []
        for name in os.listdir(self.PARTY_SETS_DIR):
            if name.startswith("party_") and name.endswith(".txt"):
                files.append(os.path.join(self.PARTY_SETS_DIR, name))
        return sorted(files)

    def _is_party_set_cache_valid(self) -> bool:
        # Validate metadata first (parameter-level compatibility check).
        metadata = self._read_party_set_metadata()
        if metadata != self._current_party_set_metadata():
            return False

        # Validate file count and expected names.
        expected_files = {
            os.path.join(self.PARTY_SETS_DIR, f"party_{i + 1}.txt")
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
        if not os.path.exists(self.PARTY_SETS_DIR):
            return
        for name in os.listdir(self.PARTY_SETS_DIR):
            path = os.path.join(self.PARTY_SETS_DIR, name)
            if os.path.isfile(path):
                os.remove(path)

    def _read_party_sets(self):
        """
        Load party sets from disk if all N files exist.
        Returns a list of IP-address lists, or None if any file is missing.
        """
        expected_files = [
            os.path.join(self.PARTY_SETS_DIR, f"party_{i + 1}.txt")
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
                    print(f"Party set cache is valid. Loaded {self.num_parties} files.")
                return existing

        if not quiet:
            print("Party set cache missing/mismatched. Regenerating from scratch...")
        self._clear_party_set_cache()
        return self.generate_party_sets()

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
        party_sets = self.load_or_create_party_sets(quiet=quiet)
        bloom_filters_full = [self.build_bloom_filter(party_set) for party_set in party_sets]

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
            "plaintext_time_sampled_s": plain_elapsed,
            "plaintext_time_estimated_full_s": plain_elapsed * scale_factor,
            "encrypted_time_sampled_s": encrypted_elapsed,
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

            # Inputset over binary vectors of length N to infer exact integer ranges.
            inputset = [
                np.array([0] * self.num_parties, dtype=np.int64),
                np.array([1] * self.num_parties, dtype=np.int64),
            ]
            rng = random.Random(42)
            for _ in range(min(256, 1 << min(self.num_parties, 12))):
                sample = np.array(
                    [rng.randint(0, 1) for _ in range(self.num_parties)],
                    dtype=np.int64,
                )
                inputset.append(sample)

            compiled = compiler.compile(inputset)
            compiled.keygen()
            self._compiled_circuits[circuit] = compiled
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
        return [int(compiled.decrypt(enc_bit)) for enc_bit in ciphertext]

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

        terms = [t.strip() for t in circuit.split("+") if t.strip()]
        if not terms:
            raise ValueError("Circuit string is empty  nothing to evaluate.")

        encrypted_result = []
        for bit_idx in range(bit_count):
            bit_inputs = np.array(
                [int(encrypted_bloom_filters[p][bit_idx]) for p in range(self.num_parties)],
                dtype=np.int64,
            )
            enc_in = compiled.encrypt(bit_inputs)
            enc_out = compiled.run(enc_in)
            encrypted_result.append(enc_out)

        return encrypted_result

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


def run_scaling_experiments(
    false_positive_rate: float = 0.0005,
    max_parties: int = 10,
    sample_bits_cap: int = 100_000,
    output_dir: str = "benchmark_outputs",
):
    """
    Run scaling sweeps and save plots:
    1) parties sweep (2..max_parties)
    2) threshold sweep (1..N)
    3) party set size sweep (10..10^7)
    4) common IPs sweep (fixed N, T, set size)
    """
    os.makedirs(output_dir, exist_ok=True)
    _all_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1) Parties sweep  N=2..max_parties  T~N/2  (synthetic Bloom bits)
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    parties_x, parties_plain, parties_enc = [], [], []
    fixed_set_size_for_parties = 10**3
    _w = len(str(max_parties))
    print(f"[1/4] Parties sweep  N=2..{max_parties}  T~N/2  (real IP files -- disk I/O per step)")
    for n in range(2, max_parties + 1):
        print(f"      N={n:{_w}}/{max_parties}  generating {n} party files...", end='\r', flush=True)
        t = max(1, n // 2)
        tc = ThresholdCircuit(
            num_parties=n,
            threshold=t,
            false_positive_rate=false_positive_rate,
            num_common_ips=10,
            party_set_size=fixed_set_size_for_parties,
        )
        metrics = tc.benchmark_computation_times_from_party_sets(sample_bits_cap=sample_bits_cap, include_encrypted=False, quiet=True)
        parties_x.append(n)
        parties_plain.append(metrics["plaintext_time_estimated_full_s"])
        parties_enc.append(metrics["encrypted_time_estimated_full_s"])
    print(f"[1/4] Parties sweep  done  ({time.perf_counter()-_t0:.1f}s)                    ")
    _plot_scaling_graph(
        parties_x, parties_plain, parties_enc,
        title="Computation Time vs Number of Parties",
        x_label="Number of parties (N)",
        output_path=os.path.join(output_dir, "scaling_num_parties.png"),
    )

    # ------------------------------------------------------------------
    # 2) Threshold sweep  T=1..N  fixed N=max_parties  (synthetic Bloom bits)
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    threshold_x, threshold_plain, threshold_enc = [], [], []
    fixed_set_size_for_threshold = 10**3
    n_fixed = max_parties
    _w = len(str(n_fixed))
    print(f"[2/4] Threshold sweep  T=1..{n_fixed}  N={n_fixed}  (real IP files -- disk I/O per step)")
    for t in range(1, n_fixed + 1):
        print(f"      T={t:{_w}}/{n_fixed}  generating {n_fixed} party files...", end='\r', flush=True)
        tc = ThresholdCircuit(
            num_parties=n_fixed,
            threshold=t,
            false_positive_rate=false_positive_rate,
            num_common_ips=10,
            party_set_size=fixed_set_size_for_threshold,
        )
        metrics = tc.benchmark_computation_times_from_party_sets(sample_bits_cap=sample_bits_cap, include_encrypted=False, quiet=True)
        threshold_x.append(t)
        threshold_plain.append(metrics["plaintext_time_estimated_full_s"])
        threshold_enc.append(metrics["encrypted_time_estimated_full_s"])
    print(f"[2/4] Threshold sweep  done  ({time.perf_counter()-_t0:.1f}s)                    ")
    _plot_scaling_graph(
        threshold_x, threshold_plain, threshold_enc,
        title=f"Computation Time vs Threshold (N={n_fixed})",
        x_label="Threshold (T)",
        output_path=os.path.join(output_dir, "scaling_threshold.png"),
    )

    # ------------------------------------------------------------------
    # 3) Set-size sweep  10..10^7  (synthetic Bloom bits)
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    size_x = [10**i for i in range(1, 4)]  # 10 to 1,000
    size_plain, size_enc = [], []
    n_for_size, t_for_size = 10, 5
    print(f"[3/4] Set-size sweep  10..10^3  N={n_for_size}  T={t_for_size}  (real IP files -- disk I/O per step)")
    for i, set_size in enumerate(size_x, 1):
        print(f"      [{i}/{len(size_x)}] set_size={set_size:,}  generating {n_for_size} party files...", end='\r', flush=True)
        tc = ThresholdCircuit(
            num_parties=n_for_size,
            threshold=t_for_size,
            false_positive_rate=false_positive_rate,
            num_common_ips=1,
            party_set_size=set_size,
        )
        metrics = tc.benchmark_computation_times_from_party_sets(sample_bits_cap=sample_bits_cap, include_encrypted=False, quiet=True)
        size_plain.append(metrics["plaintext_time_estimated_full_s"])
        size_enc.append(metrics["encrypted_time_estimated_full_s"])
    print(f"[3/4] Set-size sweep  done  ({time.perf_counter()-_t0:.1f}s)                    ")
    _plot_scaling_graph(
        size_x, size_plain, size_enc,
        title=f"Computation Time vs Party Set Size (N={n_for_size}, T={t_for_size})",
        x_label="Party set size",
        output_path=os.path.join(output_dir, "scaling_party_set_size.png"),
    )

    # ------------------------------------------------------------------
    # 4) Common-IPs sweep  (REAL party sets -- regenerated each iteration)
    # ------------------------------------------------------------------
    _t0 = time.perf_counter()
    common_x = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    common_plain, common_enc = [], []
    n_for_common, t_for_common, set_size_for_common = 10, 5, 10**3
    print(f"[4/4] Common-IPs sweep  N={n_for_common}  T={t_for_common}  set_size={set_size_for_common:,}")
    print( "      (party files are written to disk and regenerated whenever params change)")
    for i, common_ips in enumerate(common_x, 1):
        print(f"      [{i}/{len(common_x)}] common_ips={common_ips:<4}  generating {n_for_common} party files...", end='\r', flush=True)
        tc = ThresholdCircuit(
            num_parties=n_for_common,
            threshold=t_for_common,
            false_positive_rate=false_positive_rate,
            num_common_ips=common_ips,
            party_set_size=set_size_for_common,
        )
        metrics = tc.benchmark_computation_times_from_party_sets(
            sample_bits_cap=sample_bits_cap,
            include_encrypted=False,
            quiet=True,
        )
        common_plain.append(metrics["plaintext_time_estimated_full_s"])
        common_enc.append(metrics["encrypted_time_estimated_full_s"])
    print(f"[4/4] Common-IPs sweep  done  ({time.perf_counter()-_t0:.1f}s)                    ")
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



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RUN_SCALING_EXPERIMENTS = True

    # ---- Parameters --------------------------------------------------------
    NUM_PARTIES = 5
    THRESHOLD = 5
    FALSE_POSITIVE_RATE = 0.0005
    NUM_COMMON_IPS = 5
    PARTY_SET_SIZE = 10**3

    # ---- Setup -------------------------------------------------------------
    tc = ThresholdCircuit(
        num_parties=NUM_PARTIES,
        threshold=THRESHOLD,
        false_positive_rate=FALSE_POSITIVE_RATE,
        num_common_ips=NUM_COMMON_IPS,
        party_set_size=PARTY_SET_SIZE,
    )
    print(f"Bloom filter size      M = {tc.num_bloom_bits:,} bits")
    print(f"Number of hash funcs   K = {tc.num_hash_funcs}")

    # ---- Party sets --------------------------------------------------------
    party_sets = tc.load_or_create_party_sets()

    # ---- Bloom filters (plaintext) -----------------------------------------
    bloom_filters = [tc.build_bloom_filter(party_set) for party_set in party_sets]
    print(f"\nBuilt {len(bloom_filters)} Bloom filters  "
          f"({tc.num_bloom_bits} bits each, K={tc.num_hash_funcs} hash functions, "
          f"FPR={tc.false_positive_rate*100:.2f}%)")

    # ---- Circuit -----------------------------------------------------------
    canonical_circuit = tc.build_canonical_circuit()
    print(f"\nCanonical circuit: {canonical_circuit}...")

    optimized_circuit = tc.optimize_circuit(canonical_circuit)
    print(f"Optimized circuit: {optimized_circuit}")

    # ---- Plaintext evaluation ----------------------------------------------
    plaintext_result = tc.evaluate_plaintext_circuit(optimized_circuit, bloom_filters)
    _ones = sum(plaintext_result)
    print(f"\nIntersection Bloom filter: {len(plaintext_result)} bits,  "
          f"{_ones} set ({100*_ones/len(plaintext_result):.1f}% density)")

    # Recover candidate elements from the (plaintext) threshold-intersection Bloom bits.
    recovered_from_plain = tc.extract_candidates_from_intersection_bloom(
        plaintext_result,
        party_sets,
    )
    exact_elements = tc.exact_threshold_intersection(party_sets)
    print(f"Recovered intersection candidates from plaintext bloom result: {recovered_from_plain}")
    print(f"Exact threshold intersection from plaintext sets: {exact_elements}")

    # ---- Concrete-python encrypted evaluation (temporarily disabled) ------
    # print("\n--- Concrete-Python Encrypted Evaluation ---")
    # try:
    #     encrypted_bloom_filters = [
    #         tc.encrypt_bloom_filter(bf) for bf in bloom_filters
    #     ]
    #     encrypted_result = tc.evaluate_encrypted_circuit(
    #         optimized_circuit, encrypted_bloom_filters
    #     )
    #     decrypted_result = tc.decrypt_bloom_filter(encrypted_result)
    #     print(f"Decrypted result (first 30 bits): {decrypted_result[:30]}")
    #     matches = decrypted_result == plaintext_result
    #     print(f"Matches plaintext result: {matches}")
    #     recovered_candidates = tc.extract_candidates_from_intersection_bloom(
    #         decrypted_result, party_sets,
    #     )
    #     print(f"Recovered candidates from decrypted bloom filter: {recovered_candidates}")
    # except RuntimeError as exc:
    #     print(f"Concrete-python FHE unavailable on this machine: {exc}")

    if RUN_SCALING_EXPERIMENTS:
        print("\n--- Scaling Experiments ---")
        run_scaling_experiments(
            false_positive_rate=FALSE_POSITIVE_RATE,
            max_parties=10,
            sample_bits_cap=100_000,
            output_dir="benchmark_outputs",
        )
