# Threshold PSI Circuit

Threshold PSI Circuit is a research prototype for threshold private set intersection over Bloom-filter-encoded IPv4 datasets. It evaluates a threshold `T-of-N` circuit in plaintext and, when OpenFHE is available, through a batched BFV-based encrypted path.

The repository currently has one Python entry point: [Circuit_OpenFHE_Batched.py](Circuit_OpenFHE_Batched.py). It can generate datasets, verify cached inputs, run a single configuration, and write runtime logs for analysis.

## What it does

- Generates or reuses synthetic party datasets of IPv4 addresses.
- Encodes each party dataset into a Bloom filter.
- Evaluates a threshold `T-of-N` circuit on those filters.
- Runs the same workflow with OpenFHE batched BFV when the OpenFHE Python package is installed.
- Produces timing logs and final-result plots for benchmark analysis.

## Repository layout

- [Circuit_OpenFHE_Batched.py](Circuit_OpenFHE_Batched.py): main CLI script for dataset generation, verification, and TPSI runs.
- [party_sets/](party_sets/): cached party datasets and [party_sets_meta.json](party_sets/party_sets_meta.json), which stores the runtime configuration.
- [Final_Results/](Final_Results/): final PNG plots and pivot-table summaries.
- [requirements.txt](requirements.txt): Python dependencies.

## Requirements

- Python 3.10 or newer
- `numpy`
- `pyprobables`
- `matplotlib`
- OpenFHE Python bindings for encrypted execution

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

OpenFHE is distributed as platform-specific Python wheels. If `pip install -r requirements.txt` does not install OpenFHE on your system, install the wheel recommended by the OpenFHE Python project for your OS and Python version.

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Ensure the OpenFHE Python package imports correctly in the same environment.

Example on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the project

The script reads its runtime settings from [party_sets/party_sets_meta.json](party_sets/party_sets_meta.json).

Generate or refresh cached datasets:

```bash
python Circuit_OpenFHE_Batched.py --generate-input
```

Verify the cached datasets:

```bash
python Circuit_OpenFHE_Batched.py --verify-input
```

Run one configuration:

```bash
python Circuit_OpenFHE_Batched.py -c n3_t2_c5_s10
```

The `-c` argument uses this format:

```text
n<num_parties>_t<threshold>_c<num_common_ips>_s<party_set_size>
```

## Outputs

Running the script can produce:

- Cached datasets in [party_sets/](party_sets/)
- Runtime logs in [benchmark_outputs/](benchmark_outputs/) or [Final_Results/](Final_Results/), depending on the workflow used
- Final plots in [Final_Results/](Final_Results/)
- Pivot tables in [Final_Results/pivot_tables.txt](Final_Results/pivot_tables.txt)

Only [party_sets/party_sets_meta.json](party_sets/party_sets_meta.json) is tracked under `party_sets/`; generated dataset files remain local.

## Configuration

The runtime configuration file stores two sections:

- `sample_config`: parameters for one sample run
- `scaling_config`: parameters for dataset preparation and benchmark sweeps

Key values include:

- `num_parties`: number of participating parties
- `threshold`: minimum number of parties that must share an element
- `num_common_ips`: number of shared IPs seeded into exactly `threshold` parties
- `party_set_size`: number of IPs in each party dataset

## Notes

- The exact threshold intersection is computed from the plaintext party sets for comparison.
- Bloom filters can introduce false positives, so recovered candidates may exceed the exact intersection.
- The encrypted evaluation path is intended for experimentation and benchmarking, not production deployment.

## Troubleshooting

- If OpenFHE import fails, install the correct OpenFHE Python wheel for your platform before running encrypted workflows.
- If dataset generation is slow, reduce `party_set_size` or `num_parties` in [party_sets/party_sets_meta.json](party_sets/party_sets_meta.json).
- If you only need the final charts and tables, inspect the files in [Final_Results/](Final_Results/).

## License

No license file is included. Add one if you want to publish or share the project under explicit terms.
