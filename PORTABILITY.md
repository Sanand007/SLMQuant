# Reproduction release changes

The archived evidence is unchanged. Runtime code is a derivative of the original experiment source, with these explicitly scoped changes:

1. `src/v2/supervise.py` uses the current Python interpreter instead of an original machine-specific virtual-environment location. Operational wall-clock and disk limits come from the runtime protocol.
2. `src/v2/runner.py` fixes the published search allowance at 90 distinct candidates per method/seed. The original pilot could reduce its proposed allowance to fit remaining runtime. That is inappropriate for a reproduction of already completed results. Candidate scoring and genetic operators are unchanged. Its wall-clock guards read the runtime protocol.
3. `src/v3/mlx_affine_study.py` uses the corrected wide30-local held-out arrays, fresh cache keys, output filename and stage key. This implements the documented fresh reevaluation, not the superseded original routing. Search and finalist selection are unchanged.
4. `reproduce.py` is new orchestration: isolated source/data copies, local model path substitution, frozen runtime hashes, a repository-wide scorer lock, process-group interruption, model downloads and saved-evidence verification. No original SQLite cache or completed evaluation is copied to a new runtime.

Original versions of the three changed files are under `evidence/original_source/`. Original evidence metadata can contain historical absolute paths; these are provenance records, not paths used by the launcher. Prepared runtime paths are resolved locally. Dataset exclusions are retained without rerunning data sampling. The pinned requirements are the recorded original environment, not an independently re-resolved lock.

The published V2 frozen budget is preserved. Removing an administrative runtime cap does not enlarge the candidate allowance, bit-width options, model-size budget or number of seeds. V3's published wide30 runner is already uncapped; the new launcher rejects optional V2 limit arguments for V3. Neither study is permitted to adapt its search to held-out outcomes.

Validation of this release covers archived counts/uniqueness, checksums, fresh-data hashes, summary arithmetic, source syntax, and isolated preparation. It does not certify that the full original multi-hour GPU study has been rerun with this wrapper. A new execution remains necessary to verify new numerical results and inference/export behavior on another device.
