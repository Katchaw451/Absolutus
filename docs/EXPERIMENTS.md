# Experiments Roadmap

Each experiment is a self-contained C++ binary and a bash driver in
`experiments/`. Pass criterion: matches known physical result within
numerical tolerance.

| ID | Name              | Qubits (max on Acer) | Physics test                          |
|----|-------------------|----------------------|---------------------------------------|
| 01 | Unitarity         | any                  | U†U = I, ‖ψ‖ = 1                      |
| 02 | Bell / CHSH       | 2                    | S = 2√2 > 2                           |
| 03 | GHZ entanglement  | 10–20                | von Neumann entropy of subsystems     |
| 04 | QFT vs FFT        | 8–12                 | matches classical FFT                 |
| 05 | Grover            | 10–14                | quadratic speedup, P(success) → 1     |
| 06 | VQE for H2        | 2–4                  | ground-state energy vs. experiment    |
| 07 | QAOA              | 6–10                 | energy landscape                      |
| 08 | Parameter-shift   | 4–6                  | gradient vs. finite differences       |
| 09 | Barren plateaus   | 4–12                 | grad variance decays with n           |
| 10 | Decoherence       | 8–12 (density matrix)| Kraus / Lindblad, T1/T2               |
