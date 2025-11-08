# Extreme ET Detection (Egli et al. Reproduction + WRR/NCC Extensions)

This repository implements a reproducible and extensible framework for:

1. Reproducing the core ideas of:

   **Egli et al., 2025, "Detecting anthropogenically induced changes in extreme and seasonal evapotranspiration observations".**

2. Extending the framework toward your planned papers:
   - **WRR track**: Global extreme ET & PET attribution at ~0.1° (1950–2025), multi-metric definitions.
   - **NCC track**: Multi-resolution (0.1° / 100 m / basin-scale) extreme ET, urban vs natural, seasonality-aware thresholds, and future scenario projections.

Key capabilities:

- ETx7d (max 7-day cumulative ET) and seasonal ET indices.
- Flexible extreme definitions (block maxima, percentiles, local/seasonal thresholds).
- Ridge-regression based forced-response detection (model-ensemble constrained).
- SREX / basin / urban-region aggregation and detection metrics.
- Multi-resolution analysis: grid ↔ high-res ↔ basin scaling relationships.
- Record-year statistics for ET extremes.
- Clean Python package structure with bilingual (EN/中文) comments.
- Synthetic and semi-realistic complex examples ready to adapt to your own datasets.

> 数据部分需用户根据许可自行下载与配置（CMIP6, ERA5-Land, GLEAM, X-BASE, 高分辨率城市遥感数据等）。
