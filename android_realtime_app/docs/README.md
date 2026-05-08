# Documentation

This directory contains open-source-ready notes extracted from the local
WT9011DCL-BT50 hardware-development materials used before publishing this app.

## Files

- [WT9011DCL-BT50 Android integration notes](wt9011dcl-bt50-integration.zh-CN.md)
  - BLE UUIDs used by the Android app.
  - Default `55 61` IMU packet layout and unit conversion.
  - Return-rate, bandwidth, calibration, and register-read commands.
  - Practical checks for the 100 Hz recognition pipeline.
- [Source material selection notes](source-materials.zh-CN.md)
  - Which local materials were summarized.
  - Which vendor binaries, archives, PDFs, generated state, and duplicate code
    were intentionally not copied into this repository.

## Scope

The copied content is deliberately small and text-first. The original local
folder includes vendor manuals, Windows software, serial drivers, SDK archives,
screenshots, and temporary OMX state. Those files are useful during local
development, but most of them are too large, redundant, or license-unclear for
direct inclusion in an open-source app repository.
