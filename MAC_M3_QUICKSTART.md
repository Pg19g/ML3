# Mac M3 Pro Quick Start

## TL;DR - Get Running in 5 Minutes

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.11 and dependencies
brew install python@3.11 cmake libomp

# 3. Clone and setup
git clone https://github.com/Pg19g/ML3.git
cd ML3

# 4. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 5. Install packages (Mac M3 optimized)
pip install --upgrade pip
pip install -r requirements-mac-m3.txt

# 6. Configure
cp .env.example .env
# Edit .env and add your EODHD_API_KEY

# 7. Run!
make ingest
make build
make train
make dashboard
```

## Verified Compatible Packages

All packages in `requirements-mac-m3.txt` have been verified to work on Apple Silicon:

✅ **Data Processing**
- pandas 2.1.0+ (native ARM64)
- polars 0.20.0+ (native ARM64, excellent performance)
- duckdb 0.9.0+ (native ARM64)
- numpy 1.26.0+ (uses Accelerate framework)

✅ **Machine Learning**
- lightgbm 4.1.0+ (native ARM64 wheels)
- xgboost 2.0.0+ (native ARM64 wheels)
- scikit-learn 1.3.0+ (native ARM64)

✅ **Web Frameworks**
- streamlit 1.28.0+ (works perfectly)
- fastapi 0.104.0+ (works perfectly)

✅ **Model Export**
- onnxruntime 1.16.0+ (ARM64 support included)

## Common Issues & Quick Fixes

### Issue: "No module named 'lightgbm'"

```bash
brew install libomp
pip install lightgbm --no-cache-dir
```

### Issue: "Architecture not supported"

```bash
# Check you're using ARM64 Python
python -c "import platform; print(platform.machine())"
# Should output: arm64

# If x86_64, reinstall Python
brew reinstall python@3.11
```

### Issue: Slow performance

```bash
# Enable all cores
export OMP_NUM_THREADS=12
echo 'export OMP_NUM_THREADS=12' >> ~/.zshrc
```

## Performance on M3 Pro

Expected speeds (100 symbols, 3 years of data):

- **Data Ingestion**: ~3-4 minutes
- **PIT Processing**: ~15-20 seconds
- **Feature Engineering**: ~1-2 minutes  
- **Model Training (5-fold CV)**: ~3-5 minutes
- **Backtesting**: ~30-45 seconds

The M3 Pro's unified memory and efficiency cores make it excellent for this workload!

## Need More Details?

See [INSTALL_MAC_M3.md](INSTALL_MAC_M3.md) for comprehensive installation guide.

## Verify Installation

```bash
# Test all imports
python -c "
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import streamlit as st
import fastapi
import onnxruntime
print('✅ All packages imported successfully!')
print(f'Pandas: {pd.__version__}')
print(f'LightGBM: {lgb.__version__}')
print(f'XGBoost: {xgb.__version__}')
"

# Run tests
pytest tests/ -v
```

## Next Steps

1. ✅ Installation complete
2. 📝 Edit `config/universe.yaml` with your symbols
3. 🔑 Add EODHD_API_KEY to `.env`
4. 🚀 Run `make ingest && make build && make train`
5. 📊 Launch dashboard: `make dashboard`

Happy trading! 🎉
