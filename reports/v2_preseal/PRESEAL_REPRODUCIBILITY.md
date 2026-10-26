# V2 pre-seal reproducibility report

Post-preregistration implementation preserving frozen semantics.
This report is **not** V2 confirmatory evidence and does **not** authorize sealed access.

- created_at: `2026-10-26T01:31:02.999719+00:00`
- integrity_digest: `45f0e827e42552fcbc1c5af516f98a8f0d4b1fac62b48b66fa03dbf1983d1390`
- frozen commit: `6f4f815a7551244673cb8f0a853f238ad11b8544`
- research_config_hash: `96947d9901b5366246ab1073af7ec17a83cf2bd074ecb0f32f66fcad21a8fe1c`

## A. Frozen research hashes

- research_config_hash: `96947d9901b5366246ab1073af7ec17a83cf2bd074ecb0f32f66fcad21a8fe1c`
- candidate: `aa0ff14bfa5fe41d0b0767c97090c0b9df70ea8e687594e5c47deed272a55a28`
- universe: `73d3fcff3c0b7df72871e6ab72085c3e4afba998ce1f66e31e06705b36d473fb`
- feature_set: `cf4e9dab9b20d33cb5c51ecb5cf4c2bdad3282016afbe34e2a6ce9d94e7ad396`
- target: `baf60d37d71211e94a2bfeff0027d8a1b4a2e131cf1023e03ce1d289acf35381`
- strategy: `ce46464d3d33bd20e96c6f429ebad60a991dc284b331f769c457f121ea43b521`
- costs: `74114006e731780591dd2a0b8ea9110f1261fe99fe9e8bb7231d326b7b529329`
- random_null: `d8661aa1fcaf42ce239865049a9707d41dee2c598faf4de06de7630cfead977e`
- acceptance_gates: `5e32900f2c16065d98271b8706b528e8e1682601314207085bdc3b34a9bea6be`

Live hashes match freeze (asserted).

## B. Exact Model A equivalence results

Available: `True`  
Passed: `True`  
Role: implementation equivalence only (not V2 confirmation).

Checks: `{'canonical_n_rows': 11470, 'canonical_fold_ids': ['2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q2', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4'], 'canonical_C_by_fold': {'2019Q1': 0.01, '2019Q2': 0.01, '2019Q3': 0.01, '2019Q4': 0.01, '2020Q1': 0.01, '2020Q2': 0.01, '2020Q3': 0.01, '2020Q4': 0.01, '2021Q1': 0.01, '2021Q2': 0.01, '2021Q3': 0.01, '2021Q4': 0.01, '2022Q1': 0.01, '2022Q2': 0.01, '2022Q3': 0.01, '2022Q4': 0.01, '2023Q1': 0.01, '2023Q2': 0.01, '2023Q3': 0.01, '2023Q4': 0.01}, 'predictions': {'eligible_keys_equal': True, 'n_eligible_v2': 11470, 'n_eligible_v1': 11470, 'n_inner_join': 11470, 'prediction_allclose': True, 'prediction_max_abs_diff': 0.0, 'chosen_C_equal': True, 'fold_id_equal': True, 'tolerance': {'rtol': 1e-09, 'atol': 1e-09}}, 'n_folds_v2': 20, 'fold_ids_v2': ['2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q2', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4'], 'fold_ids_equal': True, 'chosen_C_v2': {'2019Q1': 0.01, '2019Q2': 0.01, '2019Q3': 0.01, '2019Q4': 0.01, '2020Q1': 0.01, '2020Q2': 0.01, '2020Q3': 0.01, '2020Q4': 0.01, '2021Q1': 0.01, '2021Q2': 0.01, '2021Q3': 0.01, '2021Q4': 0.01, '2022Q1': 0.01, '2022Q2': 0.01, '2022Q3': 0.01, '2022Q4': 0.01, '2023Q1': 0.01, '2023Q2': 0.01, '2023Q3': 0.01, '2023Q4': 0.01}, 'chosen_C_equal': True, 'walkforward_rerun': True, 'metric_log_loss_close': True, 'metric_brier_close': True, 'metric_auc_close': True, 'metric_rank_ic_close': True, 'selection_phase_0_equal': True, 'selection_phase_1_equal': True, 'selection_phase_2_equal': True, 'selection_equal': True, 'trade_ids_phase_0_equal': True, 'trade_count_phase_0_v2': 2374, 'trade_count_phase_0_v1': 2374, 'trade_ids_phase_1_equal': True, 'trade_count_phase_1_v2': 2364, 'trade_count_phase_1_v1': 2364, 'trade_ids_phase_2_equal': True, 'trade_count_phase_2_v2': 2376, 'trade_count_phase_2_v1': 2376, 'ledger_fast_equal': True, 'phase_trade_counts_v2': [1187, 1182, 1188], 'phase_trade_counts_v1': [1187, 1182, 1188], 'trade_identities_equal': True, 'economic_metrics_close': True, 'mean_sharpe_v2': 0.5827386645170073, 'mean_sharpe_v1': 0.5827386645170073, 'mean_cagr_v2': 0.07775255992272519, 'mean_cagr_v1': 0.07775255992272519}`

## C. V1 2019-2023 reconstruction comparison

Canonical paths: `{'oos_A': 'C:\\Users\\xsaad\\Documents\\jev-trading\\reports\\phase8_lineup\\oos_A.parquet', 'signals_A': 'C:\\Users\\xsaad\\Documents\\jev-trading\\reports\\phase8_lineup\\signals_A.parquet', 'comparison': 'C:\\Users\\xsaad\\Documents\\jev-trading\\reports\\phase8_lineup\\comparison_table.csv', 'phase3_metrics': 'C:\\Users\\xsaad\\Documents\\jev-trading\\reports\\phase3_baselines\\metrics.json', 'phase3_oos_A': 'C:\\Users\\xsaad\\Documents\\jev-trading\\reports\\phase3_baselines\\oos_A.parquet'}`

Canonical artifacts win over rounded chat sanity targets.

## D. Ledger vs fast-path equality

{'engine': 'run_fast delegates to run_ledger; EQUITY_TOL_REL=1e-9', 'checked_in_rehearsal': True, 'checked_in_2019_2023_equivalence': True}

## E. Random-null deterministic reproducibility

N=2000; base_seed=15224036858585574915; generator=numpy.random.PCG64

## F. Null uniformity/sanity fixture results

see tests/unit/test_v2_null_engine.py

## G. Acceptance-evaluator gate tests

{'p_threshold': 0.05, 'min_sharpe_advantage': 0.1, 'n': 2000}

## H. One-shot guard tests

see tests/unit/test_v2_sealed_guard.py

## I. Second-look invalidation tests

see tests/unit/test_v2_sealed_guard.py

## J. Atomic-path test

run_v2_final_test records steps without pause

## K. Training-cutoff test

T0=2024-01-02; cutoff=2023-12-22; fit_once=True

## L. No-sealed-access verification

sealed_access_count=0; dev_year_max=2023; V2_SEALED_PASS exists=False

## M. Pseudo-sealed development rehearsal status

label=`development_rehearsal_only`; ok=`True`

Labeled `development_rehearsal_only`. Not V2 evidence.

## N. Secret scan

Invoked via `pytest -m secret_scan`.

## O. Remaining known issues

- Model A 2019-2023 walk-forward equivalence PASSED.
- Rehearsal uses reduced N for software validation; frozen N remains 2000.
- V2_SEALED_PASS is not created from rehearsal.
- The real `jev v2 final-test` command is implemented but must not be executed against 2024-2025 in this phase (ForbiddenRealSealedSource).

integrity_digest is an integrity digest, not a cryptographic signature.
