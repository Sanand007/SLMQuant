# Data, models and rights

This repository bundles the exact prepared arrays and small question subsets used by the experiments so a rerun does not silently choose different evaluation text. They are derived third-party data, not newly authored datasets. Original preparation metadata is under `inputs/` and `evidence/`.

## Upstream data

| Source | Recorded revision | Purpose / source information |
| --- | --- | --- |
| [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) | `b08601e04326c79dfdd32d625aee71d232d685c3` | Calibration/search text; consult dataset card and underlying Wikipedia licensing |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | `eb185aade064a813bc0b7f42de02595523103ca4` | News-domain calibration/search/validation and held-out text; V2 uses its official test split; consult dataset card and upstream corpus terms |
| [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) | `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` | Review-domain text; consult dataset card and original review corpus terms |
| [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) | `210d026faf9955653af8916fad021475a3f00453` | ARC-Easy / ARC-Challenge questions; consult dataset card and upstream license |

The original preparation scripts specify the exact partitions, tokenization and exclusions. Use the shipped arrays for direct reproduction. `inputs/v3/studies/wide30_mlx_affine_qwen3_1_7b/fresh_data_manifest.json` binds the two corrected fresh arrays and selected ARC IDs to their hashes. Earlier task IDs are supplied as exclusion records. Token counts in the README count predicted tokens; each array row also contains the initial conditioning token.

## Models and libraries

Weights are not bundled. The downloader retrieves pinned revisions from [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) and [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Their upstream model cards and license files govern use. This package does not grant rights to third-party weights, data, MLX, MLX-LM, Transformers or other dependencies. The exact model revisions are recorded in README and `reproduce.py`.

## Project code and citation

Author: Smit Anand, smit.anand033@gmail.com. Accompanying manuscript: *Selection Policies for Equal-Byte Mixed-Precision Quantization of Small Language Models* (completed V2 and corrected V3 evidence).

A project software license and permanent publication identifier have not been supplied. No license or DOI has been invented. Before public distribution, the author should select a project license and confirm the terms for bundled derived data. If a dataset's terms require separate acquisition, distribute its preparation procedure and hashes instead of its arrays, clearly explaining that extra setup step. Do not represent this notice as a replacement for upstream licenses.
