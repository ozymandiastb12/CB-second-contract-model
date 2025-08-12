# Predicting Second Contract Odds for NFL Rookie CBs

This project estimates the probability that an NFL rookie cornerback will earn a multi-year second contract based on historical data.

## Overview
I combined PFF rookie-year performance data (coverage grades, snaps, etc.) with draft capital and contract history for all rookie CBs from 2016–2020. Then, I trained a machine learning model (Random Forest Classifier) to predict second-contract outcomes.

For a 2024 rookie (Dru Phillips, NYG), the model estimated:
**62.7% chance** of earning a multi-year second deal — well above the average for a 3rd-round CB.

## Process
1. **Data Collection**
   - Rookie CBs from 2016–2020
   - PFF performance metrics + usage data
   - Draft round/pick from NFL data sources
   - Second contract details from contract databases (cap-adjusted)

2. **Feature Engineering**
   - Normalized player names
   - Computed usage metrics (slot share, snap counts)
   - Adjusted contract values for salary cap inflation

3. **Modeling**
   - Algorithm: Random Forest Classifier
   - Cross-validation (stratified, 5 folds)
   - Evaluation: ROC AUC, calibration curves, feature importance

4. **Prediction**
   - Input: Dru Phillips’ rookie metrics (from 2024)
   - Output: Probability of earning a second contract

## Results
- **Dru Phillips Probability**: 62.7%
- **Top Predictive Features**:
  1. Coverage grade
  2. Draft round
  3. Defensive snaps
  4. Tackling efficiency
