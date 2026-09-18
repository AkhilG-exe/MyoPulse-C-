# Bioprosthetics Library — Project Log

Log of work on the C++20 sEMG intent-recognition library at
`C:\Users\gopua\Bioprosthetics libarary\prosthetic-intent-engine`.

---

## Goal of the project

Build a production-grade C++20 sEMG intent-recognition library with four modules:

| Module | File(s) | Purpose |
| ------ | ------- | ------- |
| IIR filtering | `src/emg_filter.cpp`, `include/intent_engine/emg_filter.hpp` | Biquad, RBJ notch, Butterworth bandpass (order 2/4/6/8), `EmgFilterChain` |
| Feature extraction | `src/feature_extractor.cpp`, `include/intent_engine/feature_extractor.hpp` | Sliding-window MAV / ZC / SSC / WL |
| Classification | `src/intent_classifier.cpp`, `include/intent_engine/intent_classifier.hpp` | LDA (generative, pooled covariance) + SVM (one-vs-one) |
| Proportional control | `src/proportional_control.cpp`, `include/intent_engine/proportional_control.hpp` | RMS to PWM with ramp, current latch, emergency stop |

Constraint: **standard library only, C++20, no code comments** anywhere.
Public headers in `include/intent_engine/`, implementations in `src/`, CMake build (`CMakeLists.txt`).

Runtime environment notes (Windows / Smart App Control):
- MinGW GCC 16.1.0 (`BrechtSanders.WinLibs.POSIX.UCRT`), g++ at
  `C:\Users\gopua\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\g++.exe`.
- Smart App Control only runs small/stripped unsigned binaries: build with `-static -s` to avoid
  runtime DLL errors (0xC0000135).
- Python 3.12.10 at `C:\Users\gopua\AppData\Local\Programs\Python\Python312\python.exe`; scikit-learn 1.9.1 for validation/ground truth.

Feature vector layout: per channel 4 features [MAV, ZC, SSC, WL] ⇒ `numFeatures = 4 × channels`.
Defaults: 100 ms window, 10 ms step. Sample rate 1000 Hz in tests ⇒ 16 features with 4 channels.

---

## Earlier session (summary)

- Full project scaffolded: 4 headers, 4 sources, `src/matrix_math.hpp`, `CMakeLists.txt`.
- Library builds clean with `g++ -std=c++20 -Wall -Wextra -pedantic -O2 -static -s` (no warnings).
- Smoke test (`Temp\opencode\smoke_main.cpp`): filter ok, 31 windows / 16 features,
  LDA 160/160, **SVM 40/160 (degenerate)**.
- First SMO bug fixed in takeStep: "changed enough" check ran before the L1/H1 re-tightening;
  reordered so the check happens after final clip. Also fixed a sloppy `labels.empty() ? 0 : 0`
  argument in `buildKernelMatrix`.
- Python reference `smo_ref2.py`: converges on all test sets, but `select_working_set` picked both
  indices from `I_up` → underperformed on overlapping data (0.775 vs sklearn 0.933, nSV=3) —
  wrong working-set selection was the root cause.

---

## Session: replace SVM training (SMO → libsvm-style solver)

The old Platt-style SMO diverged: bias blew up to ~1e16 and classification degraded to 40/160.
Replaced it with a stable, sklearn-validated libsvm (WSS2) binary QP solver.

### 1. Pulled gold-standard reference

Fetched official libsvm source → `Temp\opencode\svm.cpp` and read:
- `Solver::Solve` (lines 508–776): α-status initialized to LOWER for all at α=0; gradient init
  `G[i] = p[i]` where `p[i] = -1`; iteration loop; max iterations `max(10^7, 100·l)`; α update with
  *two* clamp pairs per branch (both `sum > C_i` and `sum > C_j` variants).
- `select_working_set` (790–887): second-order WSS2. `i` = argmax of `-y_i·G_i` over `I_up`;
  `j` = argmin of objective decrease `-(Gmax+G[j])² / quadCoef` (or `/TAU` when `quadCoef ≤ 0`)
  over `I_low` with `quadCoef = QD[i]+QD[j]−2·y[i]·Q_i[j]`. Stop when `Gmax + Gmax2 < eps` or no j.
- `calculate_rho` (970–1006): `rho = mean(yᵢGᵢ)` over free SVs; else `(ub+lb)/2` using the bound rules.

Key distinction from the old broken approach: WSS2 selects **i from I_up and j from I_low**
(opposite violating sets), and α-status uses **exact** comparisons (`α >= C`, `α <= 0`), not tolerances.

### 2. Python reference solver (`Temp\opencode\smo_libsvm.py`)

Line-faithful Python port of the libsvm binary solver (`BinarySolver`: `select_working_set`,
`update_step`, `rho`, `solve`), validated against scikit-learn before touching C++:
- Initial status fixed to all LOWER (matches libsvm; converges much faster: 4 iters vs 99 on
  separable blobs).
- Validation results (C=1.0, linear, tol 1e-3):

| Dataset | acc mine | acc sklearn | nSV mine | max\|diff\| | corr | iters |
| --- | --- | --- | --- | --- | --- | --- |
| separable-blobs | 1.000 | 1.000 | 3 | 3.46e-08 | 1.0000 | 4 |
| overlapping-blobs | 0.933 | 0.933 | 20 | 6.46e-03 | 1.0000 | 36 |
| tiny-2pt | 1.000 | 1.000 | 2 | 0 | 1.0000 | 1 |
| tiny-3pt | 1.000 | 1.000 | 2 | 0 | 1.0000 | 1 |
| synthetic-pair | 1.000 | 1.000 | 5 | 7.50e-04 | 1.0000 | 11 |
| highdim-gauss | 1.000 | 1.000 | 9 | 7.55e-04 | 1.0000 | 26 |

Decision values correlate 1.0000 with sklearn, support-vector counts identical, accuracies identical.

### 3. C++ port (`src/intent_classifier.cpp`)

- **Removed:** `clipTo`, `SmoState` (takeStep / examineExample), `SmoResult`, `trainBinarySmo`.
- **Added:**
  - Constants: `kSolverTau = 1e-12`, `kInfinity = 1e30`, α-status `kAlphaLower/Free/Upper`.
  - `BinarySvmResult { std::vector<double> alpha; double rho; }`.
  - `BinarySvmState`: stores `y`, y-folded Q: `Q[i·n+j] = yᵢ·yⱼ·K_ij`, `QD` (diagonal), `gradient`
    (init −1), `alpha` (init 0), `status` (init LOWER); methods `isUpper/isLower/updateStatus`,
    `selectWorkingSet`, `updateStep`, `solve`, `rho`.
  - `trainBinarySvm`: builds raw K, folds y in solver, runs solve with
    `maxIterations = max(10^7, 100·size)`.
- `trainPair` now stores `model.bias = -rho` (libsvm decision function = `Σ αᵢyᵢK − rho`, so
  `bias = −rho`). Coefficients remain `αᵢ·yᵢ` as before.
- `buildKernelMatrix` kept (raw K); y-folding happens inside the solver.

Two bugs in the first draft of the port, caught by code review before compiling:
1. Label signs were dropped from the state. They are essential: y decides the violating set
   (I_up/I_low), which α-update branch applies, and the ub/lb rules in rho. Fixed by storing
   `std::vector<int> y` alongside Q.
2. Leftover dead/scaffold code (`for (int label : status)` no-op loop, stray `if (i != j) {}`,
   `diff > c - c` placeholder) removed; structure now mirrors libsvm exactly.
3. Unused-variable warning (`size` in `trainBinarySvm`) removed.

### 4. Verification

- All 4 TUs compile with `-Wall -Wextra -pedantic -O2`: **no warnings**.
- Smoke test (`Temp\opencode\smoke_main.exe`):
  ```
  filter energy ok=1, windows=31, features=16
  lda correct=160/160, svm correct=160/160        (SVM was 40/160 before)
  gesture label for 2 -> Pinch | kernel for 1 -> rbf
  rms=0.9059 duty=0.0000 latched=1
  SMOKE PASS
  ```
- ML debug (`Temp\opencode\debug_ml.exe`): all 6 one-vs-one classifiers train, sane SV counts
  (nSV 4,5,4,5,17,12), finite biases (pair(1,3) bias −11.87 was the runaway one), **160/160 with
  zero confusion**.
- Rebuilt static archive `libintent_engine.a` from all 4 object files.
- Removed stale `debug_ml.exe`, `smoke_test.exe`, `verify_engine.exe` from the project directory
  (now only `include/`, `src/`, `CMakeLists.txt`).

---

## Current state

- **SVM:** stable libsvm-style WSS2 solver, sklearn-validated. Public API unchanged
  (`SvmParameters`, `SvmModel`, `SvmClassifier::fit/predict/decisionScores/load`).
  `maxPasses` is still validated as input but no longer drives the iteration count;
  iteration cap is libsvm's `max(10^7, 100·l)` (no shrinking).
- **LDA:** unchanged (generative, pooled covariance), 160/160.
- **Filter / features / control:** unchanged.

## Files

- `prosthetic-intent-engine/src/intent_classifier.cpp` — SVM solver replaced (this session).
- `Temp\opencode\smo_libsvm.py` — validated Python reference solver.
- `Temp\opencode\svm.cpp` — official libsvm reference source.
- `Temp\opencode\smoke_main.cpp` / `debug_ml.cpp` — verification programs.