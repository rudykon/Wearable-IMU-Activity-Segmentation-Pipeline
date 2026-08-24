# Dataset

**HLS-HAR** was designed for long-session sports recording rather than isolated
window classification. Huawei sports watches captured wrist acceleration and
angular velocity in real sports-health monitoring sessions. Research access is
coordinated by the Hainan University organizers.

<div class="metric-strip paper-metrics">
  <div class="metric"><strong>137</strong><span>recordings</span></div>
  <div class="metric"><strong>46.8M</strong><span>valid samples</span></div>
  <div class="metric"><strong>259.6 h</strong><span>sensor data</span></div>
  <div class="metric"><strong>5</strong><span>activities</span></div>
</div>

## Activities

The foreground vocabulary contains badminton, rope skipping, dumbbell fly,
running, and table tennis. Background motion is modeled internally but is not
reported as a workout record.

| Activity | External-test segments |
| --- | ---: |
| Badminton | 32 |
| Rope skipping | 20 |
| Dumbbell fly | 20 |
| Running | 20 |
| Table tennis | 22 |
| **Total** | **114** |

## Splits

The split is fixed at the recording level.

| Split | Recordings | Use |
| --- | ---: | --- |
| Training | 80 | model fitting and training-stage selection |
| Development / calibration | 20 | temporal-policy tuning and diagnostics |
| Independent external test | 37 | final scoring after the operating point is frozen |

The 100-record model-development pool contains 284 labeled activity segments,
about 34.8 million samples, and 130.5 hours of sensing. The 80-record training
split contributes about 28.2 million samples and 113.5 hours.

Development-set scores are **diagnostics**, not independent estimates. External
labels are used only for final scoring; they are not used to select checkpoints,
fusion rules, TRL parameters, or reported variants.

## Signals

The reported system uses six 100 Hz channels:

| Signal | Meaning |
| --- | --- |
| `ACC_X`, `ACC_Y`, `ACC_Z` | tri-axial wrist acceleration |
| `GYRO_X`, `GYRO_Y`, `GYRO_Z` | tri-axial wrist angular velocity |
| timestamp | millisecond reference for segment boundaries |

Three overlapping views are constructed with a one-second step:

| Window | Samples | Training windows |
| --- | ---: | ---: |
| 3 s | 300 | 281,871 |
| 5 s | 500 | 281,711 |
| 8 s | 800 | 281,471 |

## Labels

Each annotation is an `(activity, start, end)` segment. Evaluation therefore
measures complete records, including event count, boundaries, and duration.
Predictions are matched one-to-one with same-class labels at IoU > 0.5.

## Scope

Public HAR datasets only partly overlap with this setting. Some use phone
accelerometers, short scripted windows, daily activities, or no gyroscope;
others do not provide long-session segment records. Cross-paper window
accuracy is therefore not directly comparable with HLS-HAR segment F1.

HAR70+, WISDM-phone, PAMAP2, and OPPORTUNITY are used only for TRL portability
checks with dataset-specific models and parameters. They are not leaderboard
comparisons and do not establish transfer of the HLS-HAR result.

!!! important "Access and privacy"

    Participant recordings are not stored in the public repository. Before the
    planned PhysioNet release, research requests follow the
    [dataset access instructions](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/data/README.md).
    Storage layout and asset checks are documented in [Assets](../reference/assets.md).
