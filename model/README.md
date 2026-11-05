# model/ — the prediction model

| file | what it does |
|---|---|
| `metanet_model.py` | `MetanetModel`: METANET / Bounded-METANET built with `sym-metanet`, compiled to a CasADi function `F(x, u, d)` for the NMPC. Holds every calibrated parameter set and the geometry loaders for both roads. |
| `ExtraData.py` | Reads lengths, lanes and speed limits from a SUMO `.net.xml`. |

Parameter sets (pass as `MetanetParams(**PARAMS_...)`):

| name | road | use |
|---|---|---|
| `PARAMS_BM` | merge | **final model on the merge** (report Table 1); fitted by `experiments/expB_bounded/calibrate_bm.py` |
| `PARAMS_BM_JAM_DIS` | straight | **final model on the straight road** (report Table 1); fitted by `calibration/calibrate_jam.py` |
| `PARAMS_BM_JAM`, `PARAMS_BM_STRAIGHT` | straight | earlier straight-road fits (see `EXPERIMENTS.md`) |
| `PARAMS_S14` | merge | asymmetric-anticipation METANET from the bake-off; robustness only |
| defaults | merge | standard METANET, shown in report Fig. 1 only |

Bounded-METANET keeps only relaxation plus a look-ahead density, so speeds
stay in [0, v_free] (report §4.1–4.2). The sign enters as a cap on the target
speed, `(u / u_legal)·v_free` (`vsl_scaled=True`).

Any change here must pass `validation/test_bounded_parity.py` and
`validation/test_bounded_parity_straight.py` (~1e-12 against the numpy
reference in `experiments/expB_bounded/bmetanet.py`).
