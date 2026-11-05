# validation/ — metrics, model studies, tests

**Metrics (computed from SUMO, used by every run)**

| file | what it does |
|---|---|
| `wave_metrics.py` | Congestion on the 200 m × 8 s speed field: congested cells (the report's "congestion"), jam vehicle-hours, front reach. |
| `sumo_metrics.py` | TTS incl. entrance queue, SSM safety (hard braking/DRAC, TET, TIT), emissions. |
| `validate_wave_trajectories.py` | Checks the cell metrics against vehicle trajectories (`data/wave_validation.json`). |

**Prediction-model studies**

| file | what it does |
|---|---|
| `prediction_study.py` / `analyse_prediction_study.py` | 600 s forecast skill vs persistence, and model vs SUMO response to fixed sign plans (report Table 2, §7.5, Fig. 7). |
| `term_scale.py` | Matched weight of each cost term relative to `tts` (report Table 10). |
| `flat_objective_check.py`, `sign_check.py` | Does an objective's cost actually respond to the sign? |
| `metanet_vs_sumo.py`, `control_authority.py` | Helpers used by the above (open-loop comparison, sign authority). |

**Tests (no SUMO needed)**

```bash
python validation/test_bounded_parity.py           # CasADi vs numpy Bounded-METANET, merge
python validation/test_bounded_parity_straight.py  # same, straight road
python validation/test_antic_parity.py             # PARAMS_S14 model
python validation/test_driver_aware.py             # driver-aware NMPC options
```
