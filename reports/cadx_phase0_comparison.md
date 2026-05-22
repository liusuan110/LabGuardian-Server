# CADx Phase 0 — Template Match Comparison Report

Registry: **6 templates** (inverting_amp_ua741_v1, summing_amp_ua741_v1, integrator_ua741_v1, common_emitter_v1, differential_pair_v1, rc_first_order_v1).


## Acceptance: **6 / 7 fixtures with expected topology pass** (top-1 matches expected & confidence ≥ 0.5)

## Per-fixture results

| Fixture | Expected | Top-1 (conf) | Top-2 (conf) | Top-3 (conf) | Pass |
|---|---|---|---|---|---|
| `bjt_diff_amp_correct_v1` | `differential_pair_v1` | `differential_pair_v1` (0.83) [shared_emitter_node] | `inverting_amp_ua741_v1` (0.00) | `summing_amp_ua741_v1` (0.00) | ✅ |
| `bjt_diff_amp_wrong_v1` | `differential_pair_v1` | `differential_pair_v1` (0.83) [shared_emitter_node] | `inverting_amp_ua741_v1` (0.00) | `summing_amp_ua741_v1` (0.00) | ✅ |
| `inverting_amp_correct_v1` | `inverting_amp_ua741_v1` | `inverting_amp_ua741_v1` (0.92) [no_bias_compensation] | `summing_amp_ua741_v1` (0.00) | `integrator_ua741_v1` (0.00) | ✅ |
| `opamp_inverting_lpf_correct_v1` | `integrator_ua741_v1` | `integrator_ua741_v1` (0.94) [with_leak_resistor] | `inverting_amp_ua741_v1` (0.88) [no_bias_compensation] | `summing_amp_ua741_v1` (0.00) | ✅ |
| `opamp_inverting_lpf_wrong_v1` | `inverting_amp_ua741_v1` | `inverting_amp_ua741_v1` (0.88) [no_bias_compensation] | `summing_amp_ua741_v1` (0.00) | `integrator_ua741_v1` (0.00) | ✅ |
| `opamp_summing_correct_v1` | `summing_amp_ua741_v1` | `summing_amp_ua741_v1` (0.87) [3_inputs] | `inverting_amp_ua741_v1` (0.83) [no_bias_compensation] | `integrator_ua741_v1` (0.00) | ✅ |
| `opamp_summing_wrong_v1` | `summing_amp_ua741_v1` | `inverting_amp_ua741_v1` (0.83) [no_bias_compensation] | `summing_amp_ua741_v1` (0.00) | `integrator_ua741_v1` (0.00) | ❌ |

## Top-1 role assignments (passing cases)

### `bjt_diff_amp_correct_v1` → `differential_pair_v1`
- `VT1` → **VT1**
- `Rc1` → **RC1**
- `Rc2` → **RC2**
- `VT2` → **VT2**

### `bjt_diff_amp_wrong_v1` → `differential_pair_v1`
- `VT1` → **VT1**
- `Rc1` → **RC1**
- `Rc2` → **RC2**
- `VT2` → **VT2**

### `inverting_amp_correct_v1` → `inverting_amp_ua741_v1`
- `IC1` → **opamp**
- `R1` → **R_f**
- `R3` → **R_g**

### `opamp_inverting_lpf_correct_v1` → `integrator_ua741_v1`
- `R1` → **R_in**
- `IC1` → **opamp**
- `C1` → **C_f**
- `Rf` → **R_leak_present**

### `opamp_inverting_lpf_wrong_v1` → `inverting_amp_ua741_v1`
- `IC1` → **opamp**
- `Rf` → **R_f**
- `R1` → **R_g**

### `opamp_summing_correct_v1` → `summing_amp_ua741_v1`
- `IC1` → **opamp**
- `Rf` → **R_f**
- `R1` → **R_in2**
- `R2` → **R_in1**

## Failures — needs investigation

### `opamp_summing_wrong_v1`
- Expected: `summing_amp_ua741_v1`
- Got top-1: `inverting_amp_ua741_v1` (0.829)
- Role assignments: `cur_compIC1`→opamp, `cur_compRf`→R_f, `cur_compR1`→R_g

## Hypothesis distribution (top-1 across all fixtures)

- `inverting_amp_ua741_v1`: 3
- `differential_pair_v1`: 2
- `integrator_ua741_v1`: 1
- `summing_amp_ua741_v1`: 1
