# Selection policies for mixed-precision quantization

Code, frozen inputs, and saved results for the completed **V2** and **corrected V3** experiments by Smit Anand (`smit.anand033@gmail.com`). This repository accompanies *Selection Policies for Equal-Byte Mixed-Precision Quantization of Small Language Models*. It does not include the ongoing V4–V6 studies.

## What question are we asking?

A small language model stores large tables of numbers called weight matrices. Quantization stores an approximation to each number using fewer bits. Four-bit quantization uses 16 possible codes per group, along with scale and offset information. It usually needs much less weight storage than a floating-point model.

Ordinary uniform four-bit MLX quantization assigns four bits to every eligible matrix. Some matrices may tolerate lower precision; others may benefit from higher precision. We ask whether choosing **which matrices receive 2, 4, or 8 bits** can improve a fixed-size model, and whether the way we search for this assignment matters.

Each assignment is a **genome**: one bit-width choice per eligible matrix. The experiment changes this assignment, then uses the same MLX affine quantizer. Tournament and Alternis are **selection rules for the search**, not new scalar quantizers. We do not train a language model from scratch, change the number of weights, or demonstrate a reduction in storage at unchanged accuracy.

A candidate is one complete assignment. Its score is negative log-likelihood (NLL): how poorly the model predicts the next token in the scoring text. **Lower NLL is better.** This is different from answering more questions correctly, and it is not a general measure of usefulness, factuality, or safety.

## What is included?

| Directory or file | Purpose |
| --- | --- |
| `src/` | Portable experiment code; shared earlier helpers are included because V2/V3 import them |
| `inputs/` | Frozen token blocks, question subsets, protocols and exclusion records |
| `evidence/v2/` | Original V2 search histories, results, selections and control results |
| `evidence/v3/` | Authoritative 500-candidate, 30-generation V3 histories and corrected results |
| `evidence/original_source/` | Original versions of the files modified for portability/corrected routing |
| `reproduce.py` | Download, prepare, launch, verify, and summarize commands |
| `tests/` | Lightweight packaging and frozen-evidence checks; no model scoring |
| `requirements-mlx.txt` | Exact dependency versions recorded in the original environment |
| `MANIFEST.json` | SHA-256 hashes for distributed files; excludes itself and generated runs |
| `PORTABILITY.md` | Changes made for this release and their scientific implications |
| `DATA_AND_LICENSES.md` | Dataset/model origins, revisions, and redistribution notes |

Model weights, virtual environments, model exports, machine caches, private account tokens, SQLite score caches, and active experiment state are omitted. Score histories and result JSON are included. New downloads go in `models/`; new outputs go in `runs/`. Both are ignored by Git.

## Quick start: verify saved results without running models

Unzip the repository, open a terminal in its top-level directory, and use Python 3.11:

```bash
python3 reproduce.py verify
python3 reproduce.py results
python3 -m unittest discover -s tests -q
```

No third-party Python packages are needed for these three commands. They work without MLX or a GPU. `verify` checks the distributed checksums, distinct search counts, V3 generation traces, and expected result counts. `results` recomputes the V3 paired-seed NLL differences from corrected saved outcomes. These checks do not rerun model inference or independently remeasure export reload or speed.

## Quick start: rerun a study on an Apple Silicon Mac

Use an Apple Silicon Mac with Metal, Python 3.11, adequate free disk and unified memory, and a reliable power source. The quantized model size is not peak memory usage: full-precision weights, quantized candidates, calibration arrays and inference buffers also take memory. Close other heavy GPU jobs. There is no supported CUDA, Intel-Mac, or Windows inference path in this release.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mlx.txt
python reproduce.py download-models
python reproduce.py prepare v2 --name v2_repeat
python reproduce.py run v2 --name v2_repeat
```

After V2 finishes, or independently using the same downloaded snapshots:

```bash
python reproduce.py prepare v3 --name v3_repeat
python reproduce.py run v3 --name v3_repeat
python reproduce.py results --folder runs/v3_repeat/v3/studies/wide30_mlx_affine_qwen3_1_7b
```

V3 does not need you to rerun V2 first: the necessary shared code and frozen inputs are supplied. Preparation copies source and data into an isolated runtime tree and substitutes the downloaded model paths. It does not copy historical score caches, finalists or completed-stage records. The old `prepare.py`/`prepare_data.py` files are historical data-building code; use the top-level commands rather than executing them directly.

The download command fetches these exact upstream snapshots:

| Use | Model | Revision |
| --- | --- | --- |
| V2 primary | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| V2 AWQ control and V3 primary | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

The first download needs internet access. Scoring uses the local model snapshots and frozen data. No hosted language-model API is called. Network availability and dependency availability are outside the reproduction scripts' control. Do not silently upgrade quantization libraries if the pins fail: record a separate environment and treat its results as a new reproduction.

## Stopping, resuming, and monitoring

- The launcher runs in the foreground and writes `runs/<name>/launcher.log`. Individual stages may write additional logs under the study directory.
- Press **Ctrl-C once** to stop. The launcher signals its process group and waits before terminating a process that has not stopped. Completed JSON and SQLite checkpoints are retained. Work not yet committed may be repeated.
- Resume with **exactly the same `run` command and name**. Search code replays deterministic state and reuses committed scores. Some interrupted non-search preparation work may have to run again.
- To start a truly fresh independent run, choose a new name. Never delete the evidence directory to restart.
- One repository-wide lock permits only one launcher at a time. Do not bypass it by launching lower-level scripts or using a second extracted copy concurrently.
- Check the study's `supervisor.json`, `run.json`, `completed_stages.json`, and `search_<method>_<seed>.json`. A status string alone is not proof a process is alive; inspect the launcher/process and recent checkpoint timestamps too. Interruption can leave the last status string behind.
- A sleeping or shut-down machine does not keep scoring. After restarting your terminal and activating the environment, run the same command to resume. Automatic launch at boot is not installed.

There is no default wall-clock or directory-size cap in these independent reproductions. The original V2 used a 12-hour/30-GB operational cap; V3 wide30 had those caps removed. Such limits are scheduling constraints, not model-size budgets. For V2 only, you may set `--max-hours` and `--max-disk-gb` at preparation. Hours are measured from the first start, including pauses; disk is the runtime directory, not the external model cache. The wrapper rejects these options for V3 instead of pretending its uncapped runner enforces them. Monitor actual free disk. The **quantized-parameter byte budget remains enforced** in both studies.

Source, protocol and prepared input hashes are checked when you resume. Do not edit a prepared run's scoring code or data. Use a new run directory and document a separate experiment if you intend to change the method.

## How the experiments differ

| Detail | V2 | Corrected V3 wide30 |
| --- | --- | --- |
| Primary model | Qwen3.5-2B | Qwen3-1.7B |
| Allocation variables | 186 eligible matrices | 196 matrices: 28 blocks × 7 projections |
| Choices | 2, 4, 8 bits | 2, 4, 8 bits |
| Quantizer | MLX affine, group size 64 | MLX affine, group size 64 |
| Methods | Random, tournament, Alternis | Random, tournament, Alternis |
| Seeds per method | 10 | 10 |
| Distinct scores per seed/method | 90 | 500 |
| Total search scores | 2,700 | 15,000 |
| Initial population | 30 | 18 |
| Later search schedule | 29 + 29 + 2 new candidates | 30 complete generations: usually 16 new candidates, 17 at generations 15 and 30 |
| Predicted search tokens | 16,384 | 8,192 |
| Predicted validation tokens | 65,536 | 32,768 |
| Primary held-out text | 131,072 predicted tokens | 32,768 predicted tokens each for fresh AG News and IMDb |
| Task check | ARC-Easy, 400 questions | ARC-Challenge, 400 questions |

V2's hybrid architecture includes 72 feed-forward, 24 full-attention and 90 linear-attention projection variables. V3 uses Q, K, V and output attention projections plus gate, up and down feed-forward projections in each block. These counts concern eligible matrices, not every tensor or individual neuron. V2 calibration/search interleaves WikiText and AG News blocks; its validation uses a separate AG News training partition and its text test uses the official AG News test split. The file named `arc_easy.json` in the V3 runtime actually holds the frozen **ARC-Challenge** subset; the name is inherited from the shared runner.

A seed is a number that makes the pseudorandom decisions reproducible. The ten paired seeds are `101, 211, 307, 401, 503, 601, 701, 809, 907, 1009`. Each method receives the same allowance and seed set. A generation is a round of producing new candidates. It is not a training epoch. V3 generation 0 is initialization, followed by 30 complete offspring generations. The total is 18 + 28×16 + 2×17 = 500.

## How a search works

1. Build the eligible matrix list and its exact byte costs, including quantization metadata. The remaining tensors contribute fixed cost.
2. Evaluate local sensitivity around uniform four-bit precision. V2 uses four calibration blocks (2,048 predicted tokens) that also occur in the search data; this is not independent validation.
3. Build the same initial population for each paired method: the uniform assignment, an informed assignment if distinct, nearby perturbations, then random assignments. The random bit prior in both versions is 0.15/0.80/0.05 for 2/4/8 bits.
4. Score complete candidate assignments on search text. Cache scores, reject repeated genomes within a run, and count distinct candidate scores rather than mutation attempts.
5. For genetic search, select parents, perform two-cut crossover, mutate each gene with probability 1/number-of-genes, and repair the result to the byte budget. Repair lowers random choices while over budget and then applies affordable promotions. It is shared across methods.
6. Retain the best candidate and repeat until the fixed allowance is exhausted.
7. Use separate validation data to select the finalist from the top three search candidates plus informed and uniform candidates, after removing duplicates. Ties use the frozen deterministic ordering.
8. Freeze choices before evaluating held-out text or task results. Do not select a new winner using those test outcomes.

**Random search** draws new assignments without fitness-based parent selection; it still has the shared initial candidates and byte repair. **Tournament selection** builds a parent pool from three independent binary contests, choosing the lower-loss entry in each. Winners can repeat. Sampling two different pool positions can therefore produce the same parent genotype. **Alternis** orders the population worst, best, next-worst, next-best and so on, then forms its pool from an interior center and neighboring positions. The pool is constructed once per generation. Consult `src/search.py` and `src/v2/operators.py` for exact tie, crossover, mutation and repair behavior.

This is an adapted comparison of parent-selection policies, not a claim that every possible implementation of a genetic algorithm has been tested.

## Additional V2 quantizers and controls

V2 includes higher precision, uniform 3/4/6/8-bit MLX, local-loss sensitivity allocation, GPTQ at four bits, and an Alternis-selected allocation followed by GPTQ. AWQ is evaluated on the separate supported Qwen3-1.7B control alongside its own higher-precision, uniform-four and GPTQ results. Compare AWQ only against controls on that same model and data. Its result is not a paired comparison against Qwen3.5-2B.

The V2 primary output has 38 rows; the AWQ-control output has four. V2's sensitivity baseline is slightly below the uniform-four byte budget. Do not extend the exact-equal-size statement for V3 representatives to every V2 baseline.

## The V3 correction you must know about

The original V3 text-evaluation code accidentally read earlier V3 text arrays instead of the fresh arrays prepared for wide30. The frozen finalists were subsequently reevaluated on the correct arrays, using separate cache keys. **`heldout_text_fresh.json` is the primary evidence.** `heldout_text.json` is retained only as superseded historical output; never average the two or describe the old result as fresh confirmation.

This release routes new evaluation to the study-local `news_test_tokens.npy` and `imdb_test_tokens.npy`, with `news_fresh` and `imdb_fresh` cache keys and a separate completion stage. It does not change the search, selected genomes, or original saved outcomes. The fresh arrays had been prepared in a superseded study but had not been evaluated there. Their hashes and task exclusion records are included.

## What the saved results support

For V3, all validation-selected random-search finalists were uniform four-bit. This does **not** mean random search never sampled mixed precision. Thus the paired comparison against random also equals the uniform-four comparison for these frozen representatives.

| Corrected V3 corpus | Tournament NLL reduction | Alternis NLL reduction |
| --- | --- | --- |
| AG News | about 0.06163; 10/10 seeds improved | about 0.00970; 5/10 improved |
| IMDb | about 0.01520; 10/10 seeds improved | about 0.00038; 3/10 improved |

`python reproduce.py results` prints the exact paired means and 95% Student-t intervals. A positive reduction means lower NLL than random/uniform. The interval is mean difference ± 2.262157 × sample standard deviation / √10. It describes variation across the ten search seeds on fixed evaluation corpora; it does not capture all uncertainty from other models, prompts, datasets or repeated exploratory comparisons.

The selected V3 representatives each use 968,001,536 quantized-parameter bytes. These results are a text-loss improvement at matched parameter storage, **not a percentage of weights removed or storage saved**. The task check did not improve: uniform-four answered 163/400 ARC-Challenge questions correctly (40.75%), tournament 156/400 (39%), and Alternis 158/400 (39.5%). Task and timing checks use one validation-selected global representative per method, not ten independent task/timing replicates.

Generation speed was measured with greedy decoding, a fixed prompt, at most 128 generated tokens, and three retained trials after a warm-up. This small benchmark does not establish a robust speed ranking. The exported selected model was reloaded and its last-position vocabulary logits compared with the in-memory model; the original maximum difference was zero under a 0.01 tolerance. Model exports are not bundled; rerunning generates new exports.
