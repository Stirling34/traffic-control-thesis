# calibration/ — fitting the model to the straight road

| file | what it does |
|---|---|
| `make_straight_baseline.py` | Runs uncontrolled SUMO on the straight road and writes `data/straight_baseline.npz` (the fixed calibration target; already shared, do not regenerate casually). |
| `calibrate_straight.py` | Open-loop + 210 s prediction fit (differential evolution) → `PARAMS_BM_STRAIGHT`, `data/params_straight.json`. |
| `calibrate_jam.py` | The same objective plus jam clearance; with the discharge term (`JAM_W_DIS=1`) → **`PARAMS_BM_JAM_DIS`**, `data/params_jam_dis_s0.json` (report §4.3, Table 1). |

The merge-road model `PARAMS_BM` was fitted by
`experiments/expB_bounded/calibrate_bm.py` against `data/sumo_baseline.npz`.
Both fits use the numpy model in `experiments/expB_bounded/bmetanet.py`.
Copy the result into `model/metanet_model.py`, then run the parity tests.
