# control/ — controller, experiment runner, report analysis

| file | what it does |
|---|---|
| `nmpc_metanet.py` | The nonlinear MPC: builds the NLP over the prediction model with `csnlp`, solves with IPOPT. |
| `cost_functions.py` | Every cost term (`term_*`) and the `term:weight\|...` spec parser. |
| `phase_d.py` | `run_closed_loop`: one SUMO + controller run over TraCI (or no control, or the reactive rule), scored on SUMO output. |
| `sweep.py` | Experiment runner: one `build_*_manifest` per round, resumable workers, one writer per shard. Final config: `build_final_manifest`, `FINAL_W_*`. |
| `launch_sweep.ps1` | Starts N detached workers for a round (refuses if workers are already running). |
| `run_prediction_study.ps1`, `run_sign_check.ps1` | Launchers for `validation/prediction_study.py` and `validation/sign_check.py`. |
| `analyse_final.py` | Round 42 table and `data/sweep_final/summary.json` (report §7.1–7.2). |
| `analyse_final_ssm.py` | Safety table (report Table 8). |
| `analyse_robustness.py` | Sign / driver-compliance table (report Table 9). |
| `analyse_overnight.py` | Rounds 39–41; only round 41 (`data/sweep_bmjam`) is shared (report §5.5). |
| `analyse_deploy.py` | Rounds 23–32 pool; also provides `spv` (time per vehicle incl. queue). |
| `make_appendix_tables.py` | Appendix A LaTeX tables. |
| `plot_report_model.py` | Report Figs. 1–2 (uses `plot_report_part1.fig_models`). |
| `plot_report_results.py` | Report Figs. 3–6. |
| `plot_sign_response.py` | Report Fig. 7. |

Analysis and plot scripts are read-only and need no SUMO (except Fig. 2).
Outputs go to `docs/final_report/` (not tracked).
