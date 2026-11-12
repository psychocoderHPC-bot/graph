<!--
SPDX-FileCopyrightText: 2026 René Widera and contributors
SPDX-License-Identifier: MPL-2.0
-->

# HAL acceptance record

This file is the filled acceptance record produced by
`scripts/hal-acceptance.sh`. Placeholders are replaced from the artifact
directory of one run. A copy of this template lives in the repository so the
record layout stays stable; the committed `docs/hal-acceptance.md` is curated
from a run of this harness.

Generated at:

{{GENERATED_AT}}

## Verdict

{{VERDICT}}

## Environment

```text
{{ENVIRONMENT}}
```

## Gates

{{SUMMARY_TABLE}}

`PASS` means the gate ran and succeeded. `FAIL` means the gate ran and failed.
`PENDING` means the gate did not run for an environmental reason or was
skipped. `NOT-BUILT` means the example target does not exist yet and the gate
is not claimed as verified.

## Commands

`ARTIFACT` is the artifact directory of this run. Every command was invoked
with the hal environment scrub prefix documented in `docs/build.md`.

```sh
{{COMMANDS}}
```

## Details

{{DETAILS}}

## Example results

{{EXAMPLE}}

## Occupancy, geometry, and atomic transport evidence

Lines extracted from the JUnit `system-out` blocks and the configure logs. No
line is synthesized when the test output does not report the value.

```text
{{DIAGNOSTICS}}
```

## Artifacts

{{ARTIFACTS}}
