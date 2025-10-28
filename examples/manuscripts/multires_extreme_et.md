**Title**
- Multi-Resolution Performance of Extreme Evapotranspiration Detection Against Station Observations

**Abstract**
- At 0.5% severity, 0.1° achieves POD 0.39 and CSI 0.24, surpassing 0.25° (POD 0.28, CSI 0.16).

**Introduction**
- We assess how spatial resolution (0.25° vs 0.1°) and temporal sampling (hourly vs daily) affect capture of extreme ET against station observations.

**Methods**
- Data: Three synthetic stations representing valley, plateau, and coastal regimes; two products at 0.25° and 0.1° with hourly/daily forms.
- Detection methods: ERT_hist (0.1–1%), ERT_clim (5%), and OPT thresholds derived from station climatology.
- Skill metrics: POD, FAR, CSI using ±1-day temporal tolerance; timing-error distributions from matched extremes.
- Interpolation/IO: Optional xarray/NetCDF utilities for nearest/bilinear sampling are provided (see src/io_utils.py).

**Results**
- 0.1° consistently improves POD/CSI while lowering FAR compared with 0.25°, with hourly aggregation comparable or slightly better for short-lived bursts.
- ERT_clim and OPT deliver similar station-consistent detection envelopes, with OPT providing smooth day-of-year thresholds.
- Timing errors concentrate within ±2 days; coarser grids exhibit weak positive lags.

**Discussion**
- Resolution gains likely reflect reduced spatial representativeness error; hourly inputs can stabilize threshold crossings near synoptic transitions.
- Method choice matters at the tails: ERT_hist emphasizes record magnitudes; ERT_clim/OPT emphasize season-relative anomalies.

**Conclusions**
- For station-scale validation of extreme ET, 0.1° products show clear advantages across severities and methods; hourly sampling modestly helps.

**Figures**
- Figure 1. Multi-panel summary (a–d): (a) POD/CSI/FAR bars at 0.5%; (b) POD–FAR curves across severities; (c) timing-error histograms; (d) case-study overlay.
  - File: examples/outputs/panel_multires_extremes.png
- Figure 2. Skill bars at 0.5% (POD, CSI, FAR) per station and product.
  - File: examples/outputs/skill_bars_sev005.png
- Figure 3. POD–FAR curves across severities (0.1–1%).
  - File: examples/outputs/roc_like_pod_far.png
- Figure 4. Timing-error histograms (product − station, days).
  - File: examples/outputs/timing_error_hist.png
- Figure 5. Case-study time series overlay (station vs products).
  - File: examples/outputs/case_study_timeseries_Valley_A.png

**Data/Code Availability**
- Code: this repository; IO utilities for NetCDF/xarray in src/io_utils.py.
- Data: synthetic in examples; real product ingestion supported via xarray.

**References**
- Zhao et al. (2025); Markonis (2025) methodologies as implemented in src/extreme_detection.py.
