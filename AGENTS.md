# AGENTS.md — jasmin-sns

SNS opinion-dynamics simulation (Python >=3.12, `uv`-managed). `README.md` is empty; design truth lives in `docs/md/設計①②③*.md` and the `F-/NFR-/A-` tags cited in each module docstring.

## Commands

- Setup: `uv sync`
- All tests: `uv run pytest`
- Single file / test: `uv run pytest tests/test_rng.py`, `uv run pytest tests/test_world.py::test_F07_ablation_does_not_mutate_params`
- No lint, typecheck, CI, or pre-commit config exists — pytest is the only verification step.

## Imports

- `pyproject.toml` sets `pythonpath = ["src"]`, so imports are top-level: `import rng`, `from config import Config, load`, `from world.build import build_world`.
- `src/` has no `__init__.py` (namespace package); do not add one or switch to `from src import ...` — `tests/test_rng.py` already mixes both styles and relies on this layout.

## Architecture (`src/`)

- `config.py` — load/validate YAML into frozen dataclasses (`Config`, `ModelConfig`, ...). `run_id()` = timestamp + 4-hex config hash.
- `rng.py` — `streams(master_seed)` derives exactly 4 `Generator`s (`network`, `init`, `persona`, `order`) via `SeedSequence.spawn`.
- `world/build.py` — `build_world(config, seed)` is the aggregation point; `World` is a frozen dataclass passed downstream.
- `world/{network,persona}.py`, `world/ablation.py` — components; `run/`, `layers/`, `io/`, `analysis/` are empty stubs. `configs/`, `scripts/` are empty.
- New tests: reuse the `small_config(tmp_path)` fixture in `tests/conftest.py` — it writes a minimal config + persona YAML pair.

## Invariants (do not break)

- **RNG**: only `rng.py` may call `np.random.default_rng()`. Everything else receives a `Generator` from `streams()`. Governance/ablation layers consume no RNG (FR-G3). `build_world` call order is fixed — `network → init → persona → order` — to keep cross-ablation comparability (NFR-2).
- **Immutability**: `O0`, `R0`, `update_order`, neighbor arrays, and all `Attrs` arrays are `writeable=False`. `apply_ablation()` must use `dataclasses.replace`, never mutate `params` — always route through it, even for `condition="none"`.
- **Config (NFR-5)**: no defaults in code; `configs/*.yaml` is canonical. Persona table path resolves relative to the config file's parent dir, not cwd (NFR-7). Validation: `S > 0` else `ConfigError` (`S > 2` = bistable warning, A3); `T_max >= 1`, `T_min <= T_max`, `eps > 0`; `k` even and `< n`; `p`, `W_values` in `[0,1]`; `max_tries >= 1`; `std >= 0`; `n_seeds >= 1` (`< 5` = warning); `master_seed_start >= 0`; malformed structure/keys/values raise `ConfigError` (never raw `KeyError`/`TypeError`/`ValueError`); ablation ∈ `none, ABL-SYS, ABL-SNS, ABL-REACT`; `update_mode` ∈ `gauss_seidel, jacobi`.
- **Persona**: `share` sum must be 1 (`atol=1e-6`); headcounts use largest-remainder method; labels are `rng.shuffle`d to avoid spatial bias on the WS ring; `theta_react ~ Beta(mu*nu, (1-mu)*nu)` via `scipy.stats.beta.rvs(random_state=rng)`, clipped to `[0,1]`.
- **Network**: pass the `Generator` directly as `seed=` to `nx.connected_watts_strogatz_graph` (requires NetworkX ≥ 3.3); wrap `NetworkXError` as `NetworkGenerationError`.
