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
import threading
import sys
import shutil
import re
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from probables import BloomFilter
from concrete import fhe
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - Set to True to enable interactive menu, False for script mode
# ============================================================================
USE_MENU = True  # Set to False to run in standalone script mode


# ============================================================================
# Utility Classes for Progress Display
# ============================================================================

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
        self._write_line(f"[FHE] {self.label} ")
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
    MAX_PARTY_IO_WORKERS = max(1, min(32, os.cpu_count() or 1))

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
        if self.num_parties <= 0:
            raise ValueError("num_parties must be >= 1")
        if self.party_set_size <= 0:
            raise ValueError("party_set_size must be >= 1")

        # Enforce a deterministic FPR based on dataset size.
        self.requested_false_positive_rate = 1.0 / self.party_set_size
        computed_fpr = self.requested_false_positive_rate
        if computed_fpr <= 0.0:
            computed_fpr = np.nextafter(0.0, 1.0)
        elif computed_fpr >= 1.0:
            computed_fpr = np.nextafter(1.0, 0.0)

        # Derive optimal Bloom-filter parameters via pyprobables.
        # Some probables versions reject ultra-small FPRs; relax minimally until valid.
        _template = None
        for _ in range(400):
            try:
                _template = BloomFilter(
                    est_elements=party_set_size,
                    false_positive_rate=computed_fpr,
                )
                break
            except ValueError as exc:
                if "math domain error" not in str(exc):
                    raise
                computed_fpr *= 10.0
                if computed_fpr >= 1.0:
                    raise ValueError(
                        "Unable to derive valid Bloom parameters for this dataset scale. "
                        "Try reducing party_set_size or num_parties."
                    ) from exc

        self.false_positive_rate = computed_fpr
        self.num_bloom_bits = _template.number_bits   # M
        self.num_hash_funcs = _template.number_hashes # K
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
        # Sequential probing avoids expensive per-IP random generation in large datasets.
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
                # Build one payload and write once to avoid per-line I/O overhead.
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

    def _party_set_file_paths_from_dir(self, dataset_dir: str, num_parties: int) -> list:
        """Get party file paths from a specific dataset directory."""
        if not os.path.exists(dataset_dir):
            return []
        files = []
        for i in range(num_parties):
            file_path = os.path.join(dataset_dir, f"party_{i + 1}.txt")
            if os.path.exists(file_path):
                files.append(file_path)
        return files

    def _count_non_empty_lines(self, file_path: str) -> int:
        with open(file_path, "r") as fh:
            return sum(1 for line in fh if line.strip())

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

    def _score_adaptation_cost(self, source_meta: dict) -> int:
        """Estimate adaptation work from a cached source (lower is better)."""
        source_parties = int(source_meta.get("num_parties", 0))
        source_set_size = int(source_meta.get("party_set_size", 0))
        source_common = int(source_meta.get("num_common_ips", 0))
        source_threshold = int(source_meta.get("threshold", 0))
        target_parties = self.num_parties
        target_set_size = self.party_set_size
        target_common = self.num_common_ips
        target_threshold = self.threshold

        if source_parties <= 0 or source_set_size <= 0 or source_threshold <= 0 or target_threshold <= 0:
            return 10**18
        if source_parties < target_parties:
            return 10**18
        if source_set_size < target_set_size:
            return 10**18
        if source_common < target_common:
            return 10**18
        if source_threshold < target_threshold:
            return 10**18

        # Weighted cost: prefer nearest compatible source in all dimensions.
        n_delta = source_parties - target_parties
        t_delta = source_threshold - target_threshold
        c_delta = source_common - target_common
        s_delta = source_set_size - target_set_size
        return (n_delta * 1_000_000) + (t_delta * 100_000) + (c_delta * 100) + (s_delta // 1_000)

    def _find_adaptation_source(self):
        """
        Find best adaptation source by scoring all cached datasets.
        Returns (source_dir, metadata, total_cost).
        """
        if not os.path.exists(self.PARTY_SETS_DIR):
            return None

        dataset_dirs = []
        for name in os.listdir(self.PARTY_SETS_DIR):
            dataset_dir = os.path.join(self.PARTY_SETS_DIR, name)
            if os.path.isdir(dataset_dir):
                dataset_dirs.append(dataset_dir)

        if not dataset_dirs:
            return None

        def _score_candidate(dataset_dir: str):
            metadata = self._read_metadata_from_dir(dataset_dir)
            if metadata is None:
                return None
            cost = self._score_adaptation_cost(metadata)
            if cost >= 10**18:
                return None
            return cost, dataset_dir, metadata

        candidates = []
        max_workers = min(len(dataset_dirs), self.MAX_PARTY_IO_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for item in pool.map(_score_candidate, dataset_dirs):
                if item is not None:
                    candidates.append(item)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        cost, dataset_dir, metadata = candidates[0]
        return dataset_dir, metadata, cost

    def _adapt_from_source(self, source_sets: list, source_meta: dict) -> list:
        """
        Adapt from cached source by down-scaling dimensions (N, T, C, S):
        - source N/T/C/S must be >= target N/T/C/S
        - common IPs end at exact target threshold count
        - all non-common IPs remain unique (count 1)
        
        OPTIMIZATION: Counts computed once and updated incrementally (not 3+ recounts).
        Unique-only IPs pre-cached to avoid linear searches. Batch IP generation.
        """
        source_num_parties = int(source_meta.get("num_parties", 0))
        source_set_size = int(source_meta.get("party_set_size", 0))
        source_common = int(source_meta.get("num_common_ips", 0))
        source_threshold = int(source_meta.get("threshold", 0))

        if (
            source_num_parties < self.num_parties
            or source_set_size < self.party_set_size
            or source_common < self.num_common_ips
            or source_threshold < self.threshold
        ):
            raise RuntimeError("Source is not a valid down-adaptation superset.")

        adapted = [set(party) for party in source_sets]

        # Identify source common pool (items currently shared by source threshold owners).
        # COMPUTE ONCE and maintain incrementally instead of recounting 3+ times.
        counts = self._item_counts(source_sets)
        source_shared = [ip for ip, cnt in counts.items() if cnt == source_threshold]
        if len(source_shared) < source_common:
            raise RuntimeError("Source shared-IP structure inconsistent with metadata.")
        source_shared = source_shared[:source_common]

        # Keep exactly target C common IPs from source shared pool.
        target_shared = source_shared[: self.num_common_ips]
        target_shared_set = set(target_shared)

        # Reduce number of parties first by dropping least/most shared owners as needed.
        while len(adapted) > self.num_parties:
            shared_hits = [sum(1 for ip in target_shared if ip in party) for party in adapted]
            if source_threshold > self.threshold:
                drop_idx = max(range(len(adapted)), key=lambda idx: shared_hits[idx])
            else:
                drop_idx = min(range(len(adapted)), key=lambda idx: shared_hits[idx])
            del adapted[drop_idx]

        used_ips = {ip for party in adapted for ip in party}

        # Demote source-shared IPs not needed in target common set to unique (count 1).
        for ip in source_shared:
            if ip in target_shared_set:
                continue
            owners = [idx for idx, party in enumerate(adapted) if ip in party]
            while len(owners) > 1:
                owner = owners.pop()
                adapted[owner].remove(ip)
                counts[ip] -= 1
            counts[ip] = 1 if owners else 0

        # Pre-cache unique-only IPs to avoid linear search in normalize loop.
        unique_only_ips = {ip for ip, cnt in counts.items() if cnt == 1 and ip not in target_shared_set}

        # Force each target common IP to appear exactly T times.
        for ip in target_shared:
            owners = [idx for idx, party in enumerate(adapted) if ip in party]
            if len(owners) > self.threshold:
                remove_n = len(owners) - self.threshold
                for owner in random.sample(owners, remove_n):
                    adapted[owner].remove(ip)
                    counts[ip] -= 1
            elif len(owners) < self.threshold:
                non_owners = [idx for idx in range(self.num_parties) if ip not in adapted[idx]]
                add_n = self.threshold - len(owners)
                if add_n > len(non_owners):
                    raise RuntimeError("Not enough non-owner parties for threshold repair.")
                for owner in random.sample(non_owners, add_n):
                    adapted[owner].add(ip)
                    counts[ip] += 1

        # Normalize party sizes to target S using pre-cached unique_only_ips.
        for party in adapted:
            while len(party) > self.party_set_size:
                # Intersect party with unique-only IPs for O(min size) lookup
                removables = party & unique_only_ips
                if not removables:
                    raise RuntimeError("Unable to shrink party size without breaking common structure.")
                removable = next(iter(removables))
                party.remove(removable)
                counts[removable] = 0
                unique_only_ips.discard(removable)
                used_ips.discard(removable)

            if len(party) < self.party_set_size:
                needed = self.party_set_size - len(party)
                for _ in range(needed):
                    new_ip = self._random_unique_ip(used_ips)
                    party.add(new_ip)
                    counts[new_ip] = 1
                    unique_only_ips.add(new_ip)

        adapted_lists = [list(party) for party in adapted]
        
        # Lightweight validation using counts dict (already computed) instead of recounting 20M IPs.
        # Check: All parties exist and have target size
        if len(adapted_lists) != self.num_parties:
            raise RuntimeError(f"Adaptation produced {len(adapted_lists)} parties, expected {self.num_parties}")
        if any(len(party) != self.party_set_size for party in adapted_lists):
            raise RuntimeError("Adaptation produced incorrect party sizes")
        
        # Check: Common IPs appear exactly T times, all others appear 1 time
        common_ips_found = [ip for ip, cnt in counts.items() if cnt == self.threshold]
        other_ips = [ip for ip, cnt in counts.items() if cnt not in (1, self.threshold, 0)]
        if len(common_ips_found) != self.num_common_ips:
            raise RuntimeError(f"Adaptation produced {len(common_ips_found)} common IPs, expected {self.num_common_ips}")
        if other_ips:
            raise RuntimeError(f"Adaptation has IPs with invalid counts: {len(other_ips)}")
        
        return adapted_lists

    def _verify_and_report_dataset_integrity(self) -> bool:
        """
        Verify dataset structure matches parameters. Returns True if valid, False if corrupted.
        Reports validation status clearly.
        """
        dataset_dir = self._dataset_dir()
        dataset_name = self._dataset_tag()
        
        # Check files exist
        expected_files = {
            os.path.join(dataset_dir, f"party_{i + 1}.txt")
            for i in range(self.num_parties)
        }
        actual_files = set(self._party_set_file_paths())
        if actual_files != expected_files:
            print(f"  [WARN] Integrity check FAILED: Expected {self.num_parties} files, found {len(actual_files)}")
            return False

        # Read all party sets
        try:
            party_sets = []
            ordered_files = sorted(actual_files)
            for file_path in ordered_files:
                with open(file_path, "r") as fh:
                    ips = [line.strip() for line in fh if line.strip()]
                party_sets.append(ips)
            
            # Validate sizes
            for i, party in enumerate(party_sets):
                if len(party) != self.party_set_size:
                    print(f"  [WARN] Integrity check FAILED: party_{i+1} has {len(party)} IPs, expected {self.party_set_size}")
                    return False
            
            # Validate IP structure
            counts = self._item_counts(party_sets)
            common_ips = [ip for ip, cnt in counts.items() if cnt == self.threshold]
            if self.threshold == 1:
                other_ips = [ip for ip, cnt in counts.items() if cnt != 1]
            else:
                other_ips = [ip for ip, cnt in counts.items() if cnt not in (1, self.threshold)]
            
            if self.threshold != 1 and len(common_ips) != self.num_common_ips:
                print(f"  [WARN] Integrity check FAILED: Expected {self.num_common_ips} common IPs (at T={self.threshold}), found {len(common_ips)}")
                return False
            
            if other_ips:
                print(f"  [WARN] Integrity check FAILED: Found {len(other_ips)} IPs with invalid occurrence counts")
                return False
            
            print(f"  [OK] Integrity check PASSED: {dataset_name} is valid")
            return True
            
        except Exception as e:
            print(f"  [WARN] Integrity check FAILED: {e}")
            return False

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

    def load_or_create_party_sets(self, quiet: bool = False) -> list:
        """
        Return party sets from disk if they already exist, otherwise generate
        them fresh, write them to disk, and return them.
        
        No adaptation, no smart caching. Just generate based on parameters.
        """
        # Quick cache check: if files already exist with correct metadata, load them
        if self._is_party_set_cache_valid():
            if not quiet:
                print(f"  [OK] Loading cached {self._dataset_tag()}")
            existing = self._read_party_sets()
            if existing is not None:
                return existing

        # Cache doesn't exist or is invalid: generate fresh based on parameters
        if not quiet:
            print(
                f"  ├─ GENERATING fresh dataset\n"
                f"  └─ TARGET: {self._dataset_tag()}"
            )
        self._clear_party_set_cache()
        return self.generate_party_sets()


    def build_bloom_filters_from_party_sets(self, party_sets: list, quiet: bool = False) -> list:
        """Build Bloom filters directly from in-memory party sets (no Bloom-cache files)."""
        if not quiet:
            print(f"Building Bloom filters for '{self._dataset_tag()}' from party input files")
        return [self.build_bloom_filter(party_set) for party_set in party_sets]

    def generate_party_sets(self) -> list:
        """
        Generate N party IP sets with controlled intersection.

        Process
        -------
          1. Create num_common_ips IP addresses and assign all of them to the
              same threshold randomly chosen parties.
        2. Fill every party's set with unique IPs until it reaches
           party_set_size entries.
        3. Write all sets to disk.

        Returns
        -------
        list of sorted IP-address lists, one per party.
        """
        party_sets = [set() for _ in range(self.num_parties)]
        used_ips: set = set()

        # Seed common IPs into one fixed owner group of exactly T parties.
        common_ips = [
            self._random_unique_ip(used_ips) for _ in range(self.num_common_ips)
        ]
        chosen_parties = random.sample(range(self.num_parties), self.threshold)
        for ip in common_ips:
            for party_idx in chosen_parties:
                party_sets[party_idx].add(ip)

        # Fill remaining slots with unique IPs sequentially.
        total_workload = self.num_parties * self.party_set_size
        if total_workload >= 2_000_000:
            print(
                f"  [INFO] Large fresh generation path for {self._dataset_tag()} "
                f"(N*S={total_workload:,})"
            )

        for party_idx, party_set in enumerate(party_sets, start=1):
            while len(party_set) < self.party_set_size:
                party_set.add(self._random_unique_ip(used_ips))
            if total_workload >= 2_000_000:
                print(f"  [INFO] Filled party {party_idx}/{self.num_parties}")

        party_lists = [list(s) for s in party_sets]
        self._write_party_sets(party_lists)
        self._write_party_set_metadata()
        return party_lists

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
            alphabetical order (index 0 = Party A, 1 = Party B, ).

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
        bloom_filters_full = self.build_bloom_filters_from_party_sets(party_sets, quiet=quiet)
        prep_elapsed = time.perf_counter() - prep_start

        sampled_bits = min(self.num_bloom_bits, sample_bits_cap)
        bloom_filters = [bf[:sampled_bits] for bf in bloom_filters_full]

        canonical_circuit = self.build_canonical_circuit()
        optimized_circuit = self.optimize_circuit(canonical_circuit)

        start_plain = time.perf_counter()
        plain_result = self.evaluate_plaintext_circuit(optimized_circuit, bloom_filters)
        plain_elapsed = time.perf_counter() - start_plain

        encrypted_elapsed = None
        encryption_elapsed = None
        encrypted_eval_elapsed = None
        decryption_elapsed = None
        encrypted_match = None
        encrypted_sampled_bits = min(sampled_bits, encrypted_sample_bits_cap)
        if include_encrypted:
            try:
                start_encrypt = time.perf_counter()
                encrypted_bloom_filters = [
                    self.encrypt_bloom_filter(bits[:encrypted_sampled_bits])
                    for bits in bloom_filters
                ]
                encryption_elapsed = time.perf_counter() - start_encrypt

                start_eval = time.perf_counter()
                enc_result = self.evaluate_encrypted_circuit(
                    optimized_circuit,
                    encrypted_bloom_filters,
                )
                encrypted_eval_elapsed = time.perf_counter() - start_eval

                start_decrypt = time.perf_counter()
                dec_result = self.decrypt_bloom_filter(enc_result)
                decryption_elapsed = time.perf_counter() - start_decrypt

                encrypted_elapsed = encryption_elapsed + encrypted_eval_elapsed + decryption_elapsed
                encrypted_match = (dec_result == plain_result[:encrypted_sampled_bits])
            except RuntimeError:
                encrypted_elapsed = None
                encryption_elapsed = None
                encrypted_eval_elapsed = None
                decryption_elapsed = None
                encrypted_match = None

        return {
            "num_parties": self.num_parties,
            "threshold": self.threshold,
            "num_common_ips": self.num_common_ips,
            "party_set_size": self.party_set_size,
            "num_bloom_bits": self.num_bloom_bits,
            "sampled_bits": sampled_bits,
            "encrypted_sampled_bits": encrypted_sampled_bits,
            "preprocessing_time_s": prep_elapsed,
            "plaintext_time_sampled_s": plain_elapsed,
            "plaintext_time_measured_s": plain_elapsed,
            "encrypted_time_sampled_s": encrypted_elapsed,
            "encrypted_time_measured_s": encrypted_elapsed,
            "encryption_time_sampled_s": encryption_elapsed,
            "encrypted_eval_time_sampled_s": encrypted_eval_elapsed,
            "decryption_time_sampled_s": decryption_elapsed,
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
    sample_bits_cap: int = 100_000,
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
            print("        Running benchmark", end="", flush=True)

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets(
                sample_bits_cap=sample_bits_cap,
                include_encrypted=True,
                quiet=True,
            )
            _exp_time = time.perf_counter() - _exp_start

            parties_x.append(n)
            parties_plain.append(metrics["plaintext_time_measured_s"])
            parties_enc.append(metrics["encrypted_time_measured_s"])
            parties_preenc.append(
                None
                if metrics["encryption_time_sampled_s"] is None
                else metrics["preprocessing_time_s"] + metrics["encryption_time_sampled_s"]
            )
            parties_eval.append(metrics["encrypted_eval_time_sampled_s"])
            parties_dec.append(metrics["decryption_time_sampled_s"])
            print(
                f" done ({_exp_time:.2f}s)  |  "
                f"plaintext: {metrics['plaintext_time_measured_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_time_measured_s']:.3f}s"
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

    def _run_threshold_sweep():
        _t0 = time.perf_counter()
        threshold_x, threshold_plain, threshold_enc = [], [], []
        threshold_preenc, threshold_eval, threshold_dec = [], [], []

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
            print("        Running benchmark", end="", flush=True)

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets(
                sample_bits_cap=sample_bits_cap,
                include_encrypted=True,
                quiet=True,
            )
            _exp_time = time.perf_counter() - _exp_start

            threshold_x.append(t)
            threshold_plain.append(metrics["plaintext_time_measured_s"])
            threshold_enc.append(metrics["encrypted_time_measured_s"])
            threshold_preenc.append(
                None
                if metrics["encryption_time_sampled_s"] is None
                else metrics["preprocessing_time_s"] + metrics["encryption_time_sampled_s"]
            )
            threshold_eval.append(metrics["encrypted_eval_time_sampled_s"])
            threshold_dec.append(metrics["decryption_time_sampled_s"])
            print(
                f" done ({_exp_time:.2f}s)  |  "
                f"plaintext: {metrics['plaintext_time_measured_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_time_measured_s']:.3f}s"
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

    def _run_set_size_sweep():
        _t0 = time.perf_counter()
        size_cfg = default_sweeps["set_size"]
        size_x = [int(v) for v in _expand_sequence_spec(size_cfg.get("set_sizes"), [10, 100, 1000])]
        size_plain, size_enc = [], []
        size_preenc, size_eval, size_dec = [], [], []

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
            print("        Running benchmark", end="", flush=True)

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets(
                sample_bits_cap=sample_bits_cap,
                include_encrypted=True,
                quiet=True,
            )
            _exp_time = time.perf_counter() - _exp_start

            size_plain.append(metrics["plaintext_time_measured_s"])
            size_enc.append(metrics["encrypted_time_measured_s"])
            size_preenc.append(
                None
                if metrics["encryption_time_sampled_s"] is None
                else metrics["preprocessing_time_s"] + metrics["encryption_time_sampled_s"]
            )
            size_eval.append(metrics["encrypted_eval_time_sampled_s"])
            size_dec.append(metrics["decryption_time_sampled_s"])
            print(
                f" done ({_exp_time:.2f}s)  |  "
                f"plaintext: {metrics['plaintext_time_measured_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_time_measured_s']:.3f}s"
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

    def _run_common_ips_sweep():
        _t0 = time.perf_counter()
        common_cfg = default_sweeps["common_ips"]
        common_x = [int(v) for v in _expand_sequence_spec(common_cfg.get("common_values"), [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])]
        common_plain, common_enc = [], []
        common_preenc, common_eval, common_dec = [], [], []

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
            print("        Running benchmark", end="", flush=True)

            _exp_start = time.perf_counter()
            metrics = tc.benchmark_computation_times_from_party_sets(
                sample_bits_cap=sample_bits_cap,
                include_encrypted=True,
                quiet=True,
            )
            _exp_time = time.perf_counter() - _exp_start

            common_plain.append(metrics["plaintext_time_measured_s"])
            common_enc.append(metrics["encrypted_time_measured_s"])
            common_preenc.append(
                None
                if metrics["encryption_time_sampled_s"] is None
                else metrics["preprocessing_time_s"] + metrics["encryption_time_sampled_s"]
            )
            common_eval.append(metrics["encrypted_eval_time_sampled_s"])
            common_dec.append(metrics["decryption_time_sampled_s"])
            print(
                f" done ({_exp_time:.2f}s)  |  "
                f"plaintext: {metrics['plaintext_time_measured_s']:.3f}s, "
                f"encrypted: {metrics['encrypted_time_measured_s']:.3f}s"
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
    print(f"\nDataset tag            = {tc._dataset_tag()}")
    print(f"Bloom filter size      M = {tc.num_bloom_bits:,} bits")
    print(f"Number of hash funcs   K = {tc.num_hash_funcs}")
    print(f"False positive rate    target={tc.requested_false_positive_rate:.3e} (1/party_set_size), effective={tc.false_positive_rate:.3e}")

    party_sets = tc.load_or_create_party_sets()
    bloom_filters = tc.build_bloom_filters_from_party_sets(party_sets)
    print(f"\nBuilt {len(bloom_filters)} Bloom filters  "
          f"({tc.num_bloom_bits} bits each, K={tc.num_hash_funcs} hash functions, "
            f"FPR={tc.false_positive_rate:.3e})")

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
        print(f"Decrypted result: {decrypted_result}")
        matches = decrypted_result == plaintext_result
        print(f"Matches plaintext result: {matches}")
        recovered_candidates = tc.extract_candidates_from_intersection_bloom(
            decrypted_result,
            party_sets,
        )
        print(f"Recovered candidates from decrypted bloom filter: {recovered_candidates}")
    except RuntimeError as exc:
        print(f"Concrete-python FHE unavailable on this machine: {exc}")


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
        tc.load_or_create_party_sets(quiet=False)
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
            print(f"    └─ {err}")
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
                print(f"    └─ {err}")
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
            f"  └─ Common IPs (appearing {expected_t} times): "
            f"{len(common_ips)} (expected {effective_expected_c})"
        )
        print(f"  └─ Unique IPs (appearing 1 time): {len(unique_ips)}")
        
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
                print(f"    └─ {ip}: appears {cnt} times")
            if len(invalid_ips) > 5:
                print(f"    └─  and {len(invalid_ips) - 5} more")
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
                    print(f"    └─ {error}")
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
                sample_bits_cap=int(scaling_config.get("sample_bits_cap", 100_000)),
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
    """Load or initialize configuration from party_sets directory."""
    defaults = {
        "script_mode": "circuit_tpsi",
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
            "parallel_dataset_workers": 0,
            "allow_unsafe_parallel_prep": False,
            "skip_oversized_prep": False,
            "max_prepare_workload": 0,
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

    config["script_mode"] = "circuit_tpsi"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    return config


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    runtime = load_runtime_config()
    
    if USE_MENU:
        # Interactive menu mode
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
            sample_bits_cap=int(runtime["scaling_config"].get("sample_bits_cap", 100_000)),
            output_dir=str(runtime["scaling_config"].get("output_dir", "benchmark_outputs")),
            scaling_sweeps=runtime["scaling_config"].get("scaling_sweeps"),
            only_sweep="all",
        )
        
        print("\n" + "="*60)
        print("All operations completed successfully!")
        print("="*60)

