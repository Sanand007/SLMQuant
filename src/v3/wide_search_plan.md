# Prospective long-generation Alternis study

This is a separate v3 follow-up to test whether Alternis needs more generations to outperform matched search. The original 188-candidate follow-up was interrupted at the user's request before validation or held-out evaluation; preserve its checkpoint directory as superseded and exclude it from the final paired analysis. Do not change or rerun v1, v2, or completed v3 strata. Run one supervised stratum at a time. The user removed the v3 wall-clock and storage caps for the restarted run on 2026-10-18; monitor available disk space but do not impose the former 12-hour or 30-GB limits.

## Primary comparison

- Use the Qwen3-1.7B MLX affine 2/4/8-bit per-matrix variant bank already validated in v3, without copying it. Use the same group size, calibration arrays, validation arrays, exact stored-model byte accounting, and uniform-four-bit byte cap for every arm.
- Compare uniform four-bit, the fixed local-sensitivity allocation, random search, tournament selection, and Alternis selection. The three searches receive the same 18-member initial population per seed and the same crossover, mutation, byte-budget repair, candidate scorer, and candidate allowance. Only parent selection differs between the two genetic algorithms.
- Run ten paired seeds and exactly 500 **distinct scored candidates per search** over 30 complete generations. Start with 18 shared candidates, retain one elite each generation, create 17 new offspring in generations 15 and 30, and 16 in each other generation: 18 + 2 × 17 + 28 × 16 = 500. The varying 17/18-member population schedule is identical for random, tournament, and Alternis. Score the common initial population once per method/seed for accounting, while reusing cached model scores where possible. Record best calibration loss and population diversity at every generation; reserve validation for finalist selection. Do not count duplicate proposals as scores.
- Freeze the 500-score, 30-generation schedule before the restarted search. Do not change it after seeing search or held-out results. Run to completion without the former time or storage caps, and resume from candidate-level checkpoints after interruption.
- Select one finalist per method/seed using validation only, with the same shortlist rule. Compare paired best-of-generation trajectories and final validation loss; quantify paired uncertainty across seeds. An Alternis-specific benefit requires improvement over **both** tournament and random search at equal candidate counts, not merely over uniform four-bit.

## Fresh final evaluation

Reserve a new, disjoint test partition before search: AG News shuffled training indices 25,000–30,000 (same shuffle seed as the v3 manifest), IMDb official test indices 5,000–10,000, and 400 ARC-Challenge IDs excluded from all earlier v3 and predecessor evaluations. Hash the selected IDs and token arrays and freeze the manifest before any new held-out score. Calibration and validation may reuse the frozen v3 partitions; never use prior or new held-out outcomes to tune operators, bit budgets, or candidate allowances.

Evaluate frozen finalists on the new text and task sets; measure stored bytes, load time, prompt and decode speed, peak memory, and preparation/search cost. Export and reload any selected model, checking its allocation and scores. Report both final quality and the cost of obtaining it. Distinguish any gain from ordinary mixed-precision search from a gain specific to Alternis.

## Replication and safeguards

After the primary comparison, consider a separate replication on the compatible Qwen3.5-2B MLX affine bank with the same predeclared rules and a newly disjoint test partition. Do not launch a second supervisor while another is active. If compatible fresh data or available physical disk space preclude replication, record the exact limitation rather than silently weakening the protocol.

Checkpoint every distinct candidate score, generation boundary, validation choice, held-out evaluation, benchmark, and export. On resume, skip completed work. Source or protocol changes after freezing require a comparability note before further held-out evaluation. The restarted 500-score study is a new protocol, not a continuation or directly pooled result of the superseded 188-score search.
