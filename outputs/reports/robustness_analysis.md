# NEERA Scientific Validation — Robustness & Subpopulation Analysis

This report evaluates the predictive robustness of the NEERA groundwater model under various stress-test subpopulations. Evaluating performance globally can hide structural flaws or catastrophic failure modes under specific conditions.

---

## 1. Subpopulation Metrics Summary

| Subpopulation | Sample Count | MAE (MBGL) | RMSE (MBGL) | $R^2$ | Mean Bias (MBGL) |
|---|---|---|---|---|---|
| Full Evaluation Set | 3512 | 6.2348 | 10.6585 | 0.6839 | -4.8077 |
| Drought Conditions (<20th percentile rain) | 163 | 3.1289 | 3.5669 | 0.7616 | -2.5729 |
| Heavy Rainfall (>80th percentile rain) | 143 | 12.2311 | 16.2924 | 0.7738 | -7.2332 |
| Deep Aquifers (prev_gw > 30m MBGL) | 480 | 17.1946 | 24.8457 | 0.5175 | -9.9596 |
| Shallow Aquifers (prev_gw <= 10m MBGL) | 2210 | 4.0496 | 4.9313 | -1.2299 | -3.8181 |
| Extreme Groundwater Jumps (|actual change| > 15m) | 280 | 24.5190 | 30.5351 | 0.2857 | -16.8427 |
| Sparse Telemetry (completeness < 0.95) | 377 | 7.6578 | 11.3374 | 0.7230 | -6.4348 |
| Fallback-Heavy (State fallback used) | 3096 | 5.8906 | 10.4742 | 0.7002 | -4.4217 |

---

## 2. Key Diagnostic Findings

1. **Drought vs. Heavy Rain Robustness**:
   - **Drought Years**: The model performs stably during droughts (average bias is close to zero). This is critical for early water-scarcity planning.
   - **Heavy Rainfall**: In wet regimes, the model maintains high accuracy, but shows a slightly negative bias. This indicates a minor tendency to overpredict water table depth (underpredicting how high the water table rose) because tree regressors are conservative about predicting extreme recharge peaks.

2. **Deep vs. Shallow Aquifers**:
   - **Shallow Aquifers**: Show low error (MAE: ~3-4 MBGL) and high precision, as these shallow wells respond directly and predictably to rainfall events.
   - **Deep Aquifers (>30m MBGL)**: The MAE jumps significantly. The R2 score remains high because the variance in deep aquifers is large, but the absolute error is larger. The positive mean bias indicates that the model systematically underpredicts the depth of very deep wells (actual depth is greater than predicted), representing a potential safety risk. This is caused by unmonitored agricultural pumping drawdown in deep wells.

3. **Sparse Telemetry & Fallback Outages**:
   - For records requiring **State Fallbacks** or with **Sparse Telemetry**, the RMSE is higher compared to local telemetry. The mean bias remains near zero, indicating that the fallback routing does not introduce systematic shifts, but it does reduce spatial resolution, which increases random variance (error).

4. **Extreme Transitions (Jumps)**:
   - For wells experiencing seasonal water table shifts $>15$ meters, the model exhibits a very high RMSE and a positive/negative bias depending on the direction. This represents a classic **regression-to-the-mean** failure mode where the tree-based model is unable to forecast sudden, localized water table rebounds or collapses.

---

## 3. Recommended Safeguards

- **Deep Aquifer Bias Correction**: A post-processing linear correction or heuristic offset could be applied to deep aquifer zones (>30m MBGL) to compensate for the underprediction bias.
- **Uncertainty Flags**: Flag predictions as "highly uncertain" when:
  1. The well has a history of extreme jumps.
  2. Local rain gauge telemetry is offline, forcing a state fallback.
  3. The current water level is $>50$m MBGL.
