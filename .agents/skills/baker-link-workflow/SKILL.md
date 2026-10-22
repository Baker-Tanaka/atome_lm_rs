---
name: baker-link-workflow
description: "Build and debug embedded Rust firmware and examples with Baker Link Env. Use when changing RP2040 firmware, adding peripheral I/O, or investigating hardware behavior."
---

# Baker Link Workflow

## 1. Check

Use `bakerInspectProject` and read the relevant source, `Cargo.toml`, and
`.vscode/launch.json`. Confirm the intended behavior and board. For I/O changes,
ask about unknown wiring, pins, and voltage levels rather than guessing.

## 2. Implement and build

Make one small change using the project's existing HAL or Embassy patterns.
Use `bakerBuild` for the configured program and check its result.
For an example, use `bakerBuildArtifact` with its discovered name and required
features; retain the returned `artifact.artifactId`. Do not move example code
into `main.rs` or rewrite launch settings just to build it.

## 3. Debug

Check `bakerDebugStatus` first; do not replace an unrelated session.
The user must start Baker Link Env on the host with **Run**.
Explain that debugging can flash/reset the board and wait for confirmation.
Use `bakerStartDebugging` with `mode: "launch"` and the existing configuration.
For a built example, use `mode: "launchWithArtifact"` with that configuration
and the returned artifact ID.

## 4. Report and finish

Report the build result and what was actually observed. Session presence alone
does not prove the firmware works. Agent tools do not read RTT logs or target
variables; ask for VS Code output when needed.
Stop only a temporary session you started and remove only breakpoints you added.
If tools or the host are unavailable, report the blocker; do not silently
switch to terminal probe-rs commands.
