"""Synthetic Example: end-to-end pipeline.

使用合成数据演示：
- 生成多模式成员 ET
- 计算 ETx7d
- 拟合 ForcedResponseDetector
- 应用到“观测”成员
- 计算趋势与记录年
"""

import numpy as np
import xarray as xr

from extreme_et_detection.indices import compute_etx7d_annual_max
from extreme_et_detection.ridge_detection import ForcedResponseDetector
from extreme_et_detection.trends import linear_trend
from extreme_et_detection.records import record_years


def build_member(seed: int, years, lat, lon, trend_scale: float = 1.0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    n_years = len(years)
    n_lat = len(lat)
    n_lon = len(lon)
    pattern = 3.0 + 0.02 * np.cos(np.deg2rad(lat))[:, None]
    forced_pattern = 0.01 * (np.array(lat)[:, None] > 0)
    all_days = []
    for i in range(n_years):
        base = pattern
        forced = forced_pattern * (i / n_years) * trend_scale
        noise = rng.normal(0, 0.5, size=(n_lat, n_lon))
        daily = base + forced + noise + rng.normal(0, 0.3, size=(365, n_lat, n_lon))
        all_days.append(daily)
    et = np.concatenate(all_days, axis=0)
    da = xr.DataArray(
        et,
        coords={"time": np.arange(et.shape[0]), "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="et_mm_day",
    )
    return da


def main():
    years = np.arange(1900, 2000)
    lat = np.linspace(-60, 75, 8)
    lon = np.linspace(-180, 180, 12, endpoint=False)

    models = ["M1", "M2", "M3"]
    family = {"M1": "A", "M2": "A", "M3": "B"}
    n_members = {"M1": 3, "M2": 4, "M3": 2}

    X_all = []
    y_all = []
    g_all = []

    for m in models:
        members = []
        for k in range(n_members[m]):
            da = build_member(seed=10 + hash(m + str(k)) % 1000,
                              years=years, lat=lat, lon=lon)
            etx = compute_etx7d_annual_max(da)
            members.append(etx)

        ens = sum(members) / len(members)
        ens_gm = ens.mean(("lat", "lon")).values

        for k, etx in enumerate(members):
            X = etx.values.reshape(len(years), -1)
            X_all.append(X)
            y_all.append(ens_gm)
            g_all.append(np.array([family[m]] * len(years)))

    import numpy as np
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    g_all = np.concatenate(g_all)

    det = ForcedResponseDetector()
    det.fit(X_all, y_all, groups=g_all)

    # apply to synthetic "observation"
    obs = build_member(seed=999, years=years, lat=lat, lon=lon)
    obs_etx = compute_etx7d_annual_max(obs)
    X_obs = obs_etx.values.reshape(len(years), -1)
    forced_obs = det.predict_forced(X_obs)

    slope, p = linear_trend(forced_obs, years)
    print("Forced trend (synthetic obs):", slope, "per year, p=", p)

    rec = record_years(obs_etx)
    print("Record-year map:", rec.shape)


if __name__ == "__main__":
    main()
