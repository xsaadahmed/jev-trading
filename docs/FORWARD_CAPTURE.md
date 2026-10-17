# Forward capture scheduling (Phase 13A)

Forward INPUT capture accumulates the **only clean evidence** against Jev training
contamination (§11.4, §16.3). It places **no orders**, imports **no** broker code,
and does **not** score models or join labels.

Evidence is valid only for the pinned Jev version recorded in `forward_registry`.
A new Jev version restarts the forward window.

## Command

```bash
# After the latest completed NYSE session, at or after 18:30 America/New_York:
jev run forward-capture

# Optional: explicit session / skip download (ops/debug)
jev run forward-capture --session 2026-10-07
jev run forward-capture --no-download
```

Exit codes: `0` for `complete`, `skipped_idempotent`, `skipped_holiday`, or `too_early`;
`1` for `incomplete` (Jev failure / model-id mismatch / missing inputs). Incomplete
days are never imputed and may be retried.

## What a successful run does

1. Resolves the target NYSE session (calendar-aware; holidays/weekends → skip).
2. Refuses to run before `decision_time` = session date **18:30 ET**.
3. Downloads bars + VIX with `available_at` and `retrieved_at` (Tiingo + FRED;
   Alpaca cross-check off; `jevlab.brokers` unused).
4. Builds features via `as_of(decision_time)`, builds `state_v1`, validates privacy.
5. Calls Jev with the pinned model id; aborts the Jev part on resolved-id mismatch.
6. Stores features, states, raw Jev responses, model id, and timestamps under
   `data/forward/YYYY-MM-DD/`.
7. On the **first** complete success, writes the immutable `forward_registry` row
   (`deployment_timestamp` UTC, pinned model id, qset hash, feature/state versions,
   git commit). `FORWARD_START_DATE` = first session whose decision time is after
   that timestamp.

Idempotent: a second run for an already-`complete` session is a no-op.

## cron (Linux)

Run shortly after the decision gate (e.g. 18:45 ET). Prefer an explicit timezone.

```cron
# crontab — America/New_York wall clock via cron's CRON_TZ (Vixie/cronie)
CRON_TZ=America/New_York
45 18 * * 1-5 cd /path/to/jev-trading && /path/to/uv run jev run forward-capture >>/var/log/jev-forward-capture.log 2>&1
```

Notes:

- `1-5` still fires on NYSE holidays; the command **skips** non-sessions
  (`skipped_holiday`) and is safe to leave enabled.
- Ensure `JEV_API_KEY`, `TIINGO_API_KEY`, and `FRED_API_KEY` are available to the
  job environment (never commit `.env`).
- Working directory must be the repo root so `config/` and `data/forward/` resolve.

## launchd (macOS)

`~/Library/LaunchAgents/ai.jevlab.forward-capture.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.jevlab.forward-capture</string>
  <key>WorkingDirectory</key>
  <string>/path/to/jev-trading</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/uv</string>
    <string>run</string>
    <string>jev</string>
    <string>run</string>
    <string>forward-capture</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>TZ</key>
    <string>America/New_York</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>45</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>45</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>45</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>45</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>45</integer></dict>
  </array>
  <key>StandardOutPath</key>
  <string>/tmp/jev-forward-capture.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/jev-forward-capture.err.log</string>
</dict>
</plist>
```

Load with:

```bash
launchctl load ~/Library/LaunchAgents/ai.jevlab.forward-capture.plist
```

Provide API keys via a root-owned env file or macOS Keychain wrapper — do not put
secrets in the plist if the file is backed up to a shared location.

## Windows Task Scheduler

Create a daily task (Mon–Fri) at 18:45 Eastern that runs:

```text
uv run jev run forward-capture
```

in the repository directory, with the same env vars as local development.

### V1 shutdown (2026-10-17)

The Windows scheduled task **`jevlab-forward-capture`** was set to **Disabled** when V1
closed (`docs/V1_ARCHIVE.md`). The task definition was retained for audit; it must not be
re-enabled for V1. Existing `data/forward/` rows and the immutable `forward_registry` entry
remain preserved. Do not reuse V1 forward observations as evidence for a future changed study.

## Acceptance (ops)

Phase 13A acceptance target: **10 consecutive clean sessions** captured with
`forward_registry` populated. Incomplete days break the consecutive streak and
must be investigated (API errors, model-id drift, data delays).
