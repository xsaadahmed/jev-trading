# Jev Trading Research Project

The authoritative specification for this repository is:

docs/MASTER_SPEC.md

Before making architectural decisions, read the relevant sections of MASTER_SPEC.md.

Important rules:

- Implement only the phase explicitly requested by the user.
- Never automatically continue to the next phase.
- Preserve completed functionality from prior phases.
- Never access sealed test data unless the requested phase explicitly authorizes it.
- Never enable live trading by default.
- Never place real orders unless the specification's live gates are satisfied and the user explicitly executes the live command.
- Never commit secrets or API keys.
- Do not add trading strategies, indicators, models, Jev questions, or assets that are not in MASTER_SPEC.md.
- Do not weaken tests to make them pass.
- If implementation behavior conflicts with MASTER_SPEC.md, follow MASTER_SPEC.md and report the conflict.
- At the end of every phase, run all required tests, linting, and type checks and produce the required phase report.
- Stop after completing the requested phase.