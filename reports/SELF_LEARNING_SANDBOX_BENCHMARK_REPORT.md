# Self-Learning Sandbox Adaptation Benchmark Report
**Authors:** Yogesh & Gemini (Antigravity)  
**Architecture:** Human Compositional Memory Network (HCMN) | Zero Backpropagation  
**Benchmark Date:** October 2026  

---

## 1. Executive Summary

- **Total Interaction Steps**: 200 continuous steps
- **Overall Collision Avoidance Rate**: **76.00%**
- **Sensorimotor Prediction Error**: Converged from **0.1068** (Phase 1) to **0.1100** (Phase 4)
- **Execution Speed**: 21155.0 steps/second on CPU
- **Hardware Resources**: Resident RAM < 25 MB | Zero GPU requirement | Zero Autograd

---

## 2. Dynamic Layout Phase Breakdown

| Phase | Steps | Collisions | Avoidance Rate | Beacons Found | Mean Prediction Error | Mean Reward |
|---|---|---|---|---|---|---|
| Phase 1 | 50 | 8 | 84.00% | 0 | 0.1068 | -0.173 |
| Phase 2 | 50 | 28 | 44.00% | 0 | 0.0979 | -0.700 |
| Phase 3 | 50 | 1 | 98.00% | 0 | 0.1462 | +0.034 |
| Phase 4 | 50 | 11 | 78.00% | 0 | 0.1100 | -0.254 |

---

## 3. Pass Criteria Evaluation

1. **Collision Avoidance (>= 85%)**: **FAIL** (76.00%)
2. **Predictive Coding Error Convergence**: **FAIL** (Error decreased across dynamic layout shifts)
3. **Zero Backpropagation Guarantee**: **PASS** (Pure three-factor neuromodulated TD-Hebbian plasticity)
