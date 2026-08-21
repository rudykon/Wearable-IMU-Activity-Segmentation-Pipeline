# Citation & license

## Cite the software

Until a project-specific archival DOI or paper citation is added, cite the
versioned repository:

~~~bibtex
@software{kong_2026_wearable_imu_segmentation,
  author  = {Kong, Minghao},
  title   = {Wearable IMU Activity Segmentation Pipeline},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline},
  license = {Apache-2.0}
}
~~~

For a reproducible report, replace or supplement `version` with the exact Git
commit used for training or inference.

## Suggested methods statement

> We used the Wearable IMU Activity Segmentation Pipeline (version 0.1.0,
> accessed at the cited commit) to process six-channel, 100 Hz accelerometer and
> gyroscope streams with aligned 3-, 5-, and 8-second classifiers and temporal
> segment decoding.

Adapt the statement to the actual configuration. Do not claim use of a scale,
checkpoint, decoder policy, Android path, or dataset that was not part of the
reported experiment.

## License

Repository-authored source code and the distributed Python and Android model
assets are licensed under the
[Apache License 2.0](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/LICENSE).
Scope-specific copies are retained beside model assets.

Apache-2.0 does not change:

- participant-data access restrictions;
- the terms of separately obtained datasets;
- third-party dependency licenses; or
- the need to cite external datasets and methods.

## Project links

- [Source repository](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline)
- [Issue tracker](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/issues)
- [Dataset access instructions](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/data/README.md)
- [Android model card](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)
