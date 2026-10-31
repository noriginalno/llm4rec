# Actual-data validation analysis

These are temporal model-selection results, not returned Kaggle scores.

Common primary cohort: **1199 users**; all-template diagnostic: **1260 users**.

| Method | Primary MAP@5 | All-template MAP@5 | Alternate tie MAP@5 | Fit + score seconds* |
|---|---:|---:|---:|---:|
| popularity | 0.323150 | 0.312466 | 0.285402 | 0.025 |
| repeat | 0.386950 | 0.373776 | 0.355716 | 0.102 |
| cbf | 0.317320 | 0.307327 | 0.301076 | 1.083 |
| association | 0.278913 | 0.269989 | 0.256391 | 0.189 |
| mf | 0.201760 | 0.197171 | 0.175619 | 0.279 |
| ppr | 0.395179 | 0.381935 | 0.358960 | 10.652 |
| hybrid_balanced | 0.400395 | 0.386756 | 0.369652 | 18.694 |

*Hybrid includes every positively weighted constituent. Shared ranking/export/ETL and initialization are excluded; these are not online latencies.

Hybrid minus PPR: **+0.005215 MAP**, paired-user descriptive 95% bootstrap interval **[-0.002994, +0.013320]**; wins/losses/ties **458/420/321**.
The same validation period selected the final method. This interval does not remove selection bias or forecast-period uncertainty.

**714** primary users have a frequency tie across the fifth-label boundary. Alternate-ID sensitivity is diagnostic, not a reason to select the most flattering metric.

## Same two training-selected users

Labels describe program IDs, not exact episodes; 4,351 source IDs have multiple titles. Full representative source titles/categories are retained in the CSV.

### User 18417531283109304442
Training: 32 qualifying broadcasts, 30 distinct programs; largest share 6.25%.
| Method | AP@5 | Top-5 IDs |
|---|---:|---|
| popularity | 0.100000 | 2400480 240081 12001682 20088 2400508 |
| repeat | 0.200000 | 10001336 12002355 200338 400335 500281 |
| cbf | 0.200000 | 10001336 90054352 12002355 90073689 90048386 |
| association | 0.040000 | 12001682 12002856 12001734 12001732 10001261 |
| mf | 0.100000 | 12001682 10001336 10002514 12002856 12001732 |
| ppr | 0.146667 | 2400480 12001682 240081 20088 10001336 |
| hybrid_balanced | 0.200000 | 10001336 12002355 12001682 12001734 12001732 |

### User 5364378838289882469
Training: 29 qualifying broadcasts, 5 distinct programs; largest share 82.76%.
| Method | AP@5 | Top-5 IDs |
|---|---:|---|
| popularity | 0.000000 | 2400480 240081 12001682 20088 2400508 |
| repeat | 0.200000 | 10001336 1100411 200337 6700482 10003012 |
| cbf | 0.200000 | 10001336 10002827 10002884 90073884 1000446 |
| association | 0.000000 | 10001261 10001657 10002253 2400508 12001712 |
| mf | 0.180000 | 1100411 10001336 200337 10001261 5200303 |
| ppr | 0.200000 | 10001336 2400480 240081 12001682 1100411 |
| hybrid_balanced | 0.200000 | 10001336 200337 1100411 10003012 6700482 |
