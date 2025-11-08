"""Global configuration & default paths.

全局配置：请根据本地 / 集群环境修改为真实路径。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

# Raw data directories (user-provided)
RAW_CMIP6_DIR = DATA_DIR / "cmip6_raw"
RAW_ERA5_DIR = DATA_DIR / "era5_land"
RAW_GLEAM_DIR = DATA_DIR / "gleam"
RAW_XBASE_DIR = DATA_DIR / "xbase"
RAW_HIGHRES_DIR = DATA_DIR / "highres"  # 例如 30–100 m 城市遥感数据
BASIN_MASK_DIR = DATA_DIR / "basins"
STATIC_DIR = DATA_DIR / "static"

SREX_MASK_PATH = STATIC_DIR / "srex_mask.nc"

# Default variable names
ET_VAR_NAME = "evspsbl"  # CMIP6 ET (kg m-2 s-1)
PET_VAR_NAME = "pet"     # 用户可按数据源调整

# Default grid resolutions for multi-resolution analyses
COARSE_RES_DEG = 2.5     # Egli 等框架
MEDIUM_RES_DEG = 0.1     # WRR 计划：全球 / 大区域
HIGHRES_TARGET_M = 100   # NCC 计划：70–100 m 城市/流域
