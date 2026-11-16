# Five-minute demonstration

1. Run `python3 -m kubeai demo --output results/release-evaluation`. It prints each policy/scenario result and writes the experiment report.
2. Open `results/release-evaluation/REPORT.md`. Compare the equal-speed and unequal-speed cases. The supplied run did not show a consistent KubeAI benefit in the equal-speed or mixed-model cases.
3. Start the two emulators and router using the three terminal commands in the main README, then open http://127.0.0.1:8080. The saved results appear automatically, including nested result directories.
4. Filter to `Emulator` and one scenario, then select a matching setup in `Recorded provenance` mode. The six charts compare policy means and individual-run dots; `Show detailed measurements` opens the tables. Inspect a run for hashes, outcomes and sampled signals. Different inputs, source/settings or topology stay in separate groups. Synthetic emulator runs have no physical GPU power/energy measurements.
5. Send a streaming chat request. Watch live reservations and then inspect `/metrics` or the structured router log for the selected endpoint and score terms.
6. Change `policy` in `config/local.json`, wait for a poll, and send another request. The dashboard shows the active policy. Stop an emulator to demonstrate health filtering; only the remaining healthy model-compatible endpoint receives requests. Restart it to demonstrate recovery.
7. Run both placement examples from the README. First-fit selects `gpu-large`; best-fit selects `gpu-small`, leaving the larger device available. Both satisfy the same capacity/compatibility filters. Remove `--at 1000` to show that this historical synthetic inventory is rejected as stale.

The emulator implements request timing and streaming, not a language model or a GPU. Its text, memory, KV-cache and token counts are synthetic. The demonstration establishes reproducible routing behavior and measurement plumbing; hardware performance requires the real vLLM path.

Use `docker compose down` for the regular Compose stack. The comparison command stops its own servers. Stop terminal-launched router/emulator processes with Ctrl+C. Existing original benchmark files are preserved.
