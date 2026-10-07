# Reproducing the optimization and QFI workflows

See [main.md](../main.md) for installation, scientific context and figure outputs. All commands below run from the repository root with `edrixs-combined` active. These are equivalent workflows reconstructed from the supplied scripts, BO JSON records and refinement archives; exact historical shell paths, scheduling and all numerical settings were not recorded.

## Inputs and preparation

Keep the configuration scripts and local Python modules in the repository root. Ce requires `Ce.npy`, `CeEmap.numpy`, `CeEmapEg.numpy` and the Ce scan data. Ta requires `Ta_Ldep.npz`, `Ta_Edep.npz`, `Ta_Edep_Large.npz` and the Ta scan data. The plotting notebook also needs its precomputed NPZ inputs, colormap and frozen Ce level branches.

Use a fresh working copy or preserve existing outputs before recalculation. BO records and refinement result files are appended when their names already exist. The combination script replaces its output, and the QFI scripts replace their output NPZs.

BO stores its JSON records and compressed evaluation logs in `Thalia/`, as configured by `output_dir`. Refinement reads those records **and the corresponding `.bz2` log archives**. Summary JSONs alone are insufficient for reconstructing the candidate list. Starting directly from the supplied collected parameter tables skips both fitting stages and the combination step.

### Ce BO bounds

Use the minimally corrected `config_Ce_t2g.py` supplied with this documentation. Only four lines in `pbounds` were changed from the uploaded repository config to match all nine historical Ce `test4` JSONs:

| Parameter | BO bounds |
| --- | --- |
| `Vdd_sigma` | `(-3.0, 0.0)` |
| `Vdd_pi` | `(-1.5, 0.0)` |
| `F4_dd` | `(true_values['F4_dd'] * 0.1, true_values['F4_dd'] * 1.5)` |
| `soc` | `(0.0, 0.5)` |

Retain the current Ce refinement settings, including `threshold_factor = 1.2`, `optname = '_subseq_optV4V'` and the current `greedy_bounds`. The available historical BO evaluation data select 22 candidates with these settings, as verified by the author using `--num-points`. The Ta BO bounds already match the historical records; retain its `threshold_factor = 1.15` and `optname = '_subseq_optV'`.

## Ce command chain

```bash
conda activate edrixs-combined

# Nine batches, each containing six BO optimizations.
for run in {1..9}; do
    seed=$((10 * (run - 1)))
    python BO_run.py --config-py config_Ce_t2g.py \
        --run-ind "$run" --init-seed "$seed"
done

# Count candidates without running refinement; expected historical count: 22.
python BO_refine.py --config-py config_Ce_t2g.py --num-points

# Starting indices are inclusive; ending indices are exclusive.
python BO_refine.py --config-py config_Ce_t2g.py 0 5
python BO_refine.py --config-py config_Ce_t2g.py 5 10
python BO_refine.py --config-py config_Ce_t2g.py 10 15
python BO_refine.py --config-py config_Ce_t2g.py 15 20
python BO_refine.py --config-py config_Ce_t2g.py 20 22

bash combine_greedy_results.sh \
    Ba3CeIr2O9_test4_subseq_optV4V \
    Ba3CeIr2O9_test4_collectedV4V.txt

python Ce_opt_momdep.py --work-dir ce_momentum_work --output-dir .
python Ce_opt_Edep.py --work-dir ce_energy_work --output-dir .

# Parity-sector calculation; seed 42 is a new reproducible choice.
python compute_ce_parity_maxima.py \
    --data-root . --work-dir calc_Ba3CeIr2O9 \
    --output-dir . --parity both --seed 42
```

If a fresh BO run produces a different candidate count, adjust refinement ranges to cover that list without gaps or overlaps. Preserve the historical 22-row table when reproducing the supplied notebook's row selections.

The final parity calculation was called the “S5 script” during workflow reconstruction. In the supplied manuscript, supplement and notebook, its results appear in **Supplementary Fig. S4**. Supplementary Fig. S5 instead shows the Ta incident-energy map. The command above follows the confirmed parity-script identity.

### Ce outputs and figure ensembles

| Stage | Output |
| --- | --- |
| BO | `Thalia/Ba3CeIr2O9_test4_run1.json` through `run9.json`, plus matching `.bz2` archives |
| Refinement | Five `Ba3CeIr2O9_test4_subseq_optV4V<start>_<end>.txt` files and matching `details.txt` files |
| Combination | `Ba3CeIr2O9_test4_collectedV4V.txt`, shape `(22, 12)` |
| Momentum calculation | `Ce_t2g_momdep_outputsV4Vr.npz`, `Ce_t2g_momdep_pol_outputsV4Vr.npz` |
| Energy calculation | `Ce_t2g_Edep_outputsV4Vr.npz`, `Ce_t2g_Edep_pol_outputsV4Vr.npz` |
| Parity calculation | `Ce_t2g_maxwal_P.npz`, `Ce_t2g_maxwal_M.npz` |

The archived refinement vectors concatenate exactly into the supplied Ce table. Both QFI scripts explicitly read that V4V table and write V4Vr output names. Unpolarized results retain all 22 rows; polarized results use fitted row **0**. The plotting notebook selects Ce rows **[0, 3, 4, 11, 14, 18]** after excluding physically unsuitable fits. It uses two intensity normalizations for each of these six fits, producing a 12-member ensemble for experimental/model means and standard deviations.

The archived V4Vr NPZs are byte-identical to the repository copies. This supports a shared parameter source; the suffix alone does not identify a change in numerical settings, and the NPZs do not store a full historical config snapshot.

## Ta command chain

```bash
conda activate edrixs-combined

for run in {1..9}; do
    seed=$((10 * (run - 1)))
    python BO_run.py --config-py config_Ta_Eg.py \
        --run-ind "$run" --init-seed "$seed"
done

# Historical refinement covered 24 selected candidates.
python BO_refine.py --config-py config_Ta_Eg.py --num-points
python BO_refine.py --config-py config_Ta_Eg.py 0 4
python BO_refine.py --config-py config_Ta_Eg.py 4 8
python BO_refine.py --config-py config_Ta_Eg.py 8 12
python BO_refine.py --config-py config_Ta_Eg.py 12 16
python BO_refine.py --config-py config_Ta_Eg.py 16 20
python BO_refine.py --config-py config_Ta_Eg.py 20 24

bash combine_greedy_results.sh \
    Ba3TaIr2O9_fitEg_subseq_optV \
    Ba3TaIr2O9_collectedV.txt

python Ta_opt_momdep.py --work-dir ta_momentum_work --output-dir .
python Ta_opt_Edep.py --work-dir ta_energy_work --output-dir .
```

The supplied Ta QFI scripts explicitly process 24 fitted rows. A different candidate count therefore requires reviewing those loops and the notebook's row selections before recalculation. No Ta parity-sector script is included.

| Stage | Output |
| --- | --- |
| BO | `Thalia/Ba3TaIr2O9_fitEg_run1.json` through `run9.json`, plus matching `.bz2` archives |
| Refinement | Six `Ba3TaIr2O9_fitEg_subseq_optV<start>_<end>.txt` files and matching `details.txt` files |
| Combination | `Ba3TaIr2O9_collectedV.txt`, shape `(24, 12)` |
| Momentum calculation | `Ta_Eg_momdep_outputsV.npz`, `Ta_Eg_momdep_pol_outputsV.npz` |
| Energy calculation | `Ta_Eg_Edep_outputsV.npz`, `Ta_Eg_Edep_pol_outputsV.npz` |

The archived vectors concatenate exactly into the supplied Ta table. All four archived Ta NPZs are byte-identical to the repository copies. Representative Ta curves and maps use fitted row **9**, which has the lowest recorded refinement objective. Polarized output row **0** represents fitted row **9**, not fitted row 0. The notebook retains all 24 fits: two normalizations give a 48-member momentum-QFI ensemble; incident-energy QFI uses one normalization and a 24-member ensemble.

## Parameter mapping and optimization stages

Both collected tables use the current `greedy_bounds` key order:

```text
Vdd_sigma, Vdd_pi, Vdd_delta, F2_dd, F2_dp, F4_dd,
G1_dp, G3_dp, tenDq, soc, soc_c, Gam_c
```

BO instead places `F4_dd` before `F2_dp` and does not optimize `tenDq` or `Gam_c`. Refinement maps BO candidates by parameter name and adds those two fitted variables. Preserve the collected-table column order: the QFI scripts interpret it using `list(cfg.greedy_bounds.keys())`.

`pbounds` controls BO, while `greedy_bounds` controls refinement. Bound values do not constrain the later spectra/QFI calculations, which use the saved fitted values. The `F2_dd >= F4_dd` filter selects starting candidates; it is not an enforced inequality throughout Py-BOBYQA refinement.

## Recorded counts and numerical reproducibility

| Material | BO invocations | Independent BO optimizations | Evaluations requested per optimization | Refinement batches | Refined points | Recorded refinement evaluations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ce | 9 | 54 | 1,000 | 5 | 22 | 65,131 |
| Ta | 9 | 54 | 1,000 | 6 | 24 | 60,468 |

Each BO optimization requests 10 initial random points and 990 Bayesian steps. For batch index `r`, `init_seed = 10 * (r - 1)`; the six internal seeds are `init_seed + 1` through `init_seed + 6`. These values match the saved JSON records. Refinement allows at most 3,600 evaluations per starting point; 11 archived Ce results and four Ta results reached that limit.

Candidate selection uses a strict distance threshold in the code (`distance < threshold_factor * minimum_distance`) followed by the configured filters. Ce uses 1.2 and Ta uses 1.15. These values agree with the candidate-selection description in the supplied supplement; the earlier Ce BO JSONs contain 1.15 as metadata, but that field does not control BO itself.

The ordinary QFI-bound scripts use 100 random L-BFGS-B restarts and do not expose a seed argument. The Ce parity script uses 500 restarts and supports `--seed`; 42 above is a newly chosen seed, not a recovered historical value. Historical random starts, package builds and machine settings were not fully recorded, so bit-for-bit agreement is not guaranteed. Computation times and installation have not been benchmarked or tested for this documentation.

### Settings to keep visible

- **Ta temperature:** the supplied Ta QFI-generation scripts use 9 K, while the plotting notebook uses 11 K for its measured-QFI integration. This documentation preserves the existing code and records the difference; it does not silently reconcile it.
- **Ce level branches:** the plotting notebook uses frozen published branches in `Ce_Fig5_published_branches.npz`; the QFI command chains do not regenerate them.
- **Plotting helper compatibility:** the notebook loads root-level `white_turbo.npy` itself. The unused `plot2.turbo_w()` instead expects `cmaps/white_turbo.npy`; `plot2.sgolay()` uses the removed `np.int` alias. The notebook uses SciPy's `savgol_filter` instead of that helper.
- **Full execution:** the supplied plotting notebook recomputes model spectra for several figure panels; having the NPZ files does not make every cell a simple plotting operation.

## Plot after recalculation

```bash
jupyter lab RIXS_QFI_paper_figures.ipynb
```

Select the analysis environment's kernel and run the notebook in order. Output filenames and figure numbering are listed in [main.md](../main.md).
