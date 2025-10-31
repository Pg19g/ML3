# ML3 Installation Guide for Mac M3 Pro

This guide provides step-by-step instructions for installing ML3 on Mac M3 Pro (Apple Silicon).

## Prerequisites

### 1. Install Homebrew

If you don't have Homebrew installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, follow the instructions to add Homebrew to your PATH.

### 2. Install Python 3.11

**Option A: Using Homebrew (Recommended)**

```bash
# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
```

**Option B: Using pyenv (For multiple Python versions)**

```bash
# Install pyenv
brew install pyenv

# Add to shell configuration (~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Restart shell
source ~/.zshrc

# Install Python 3.11
pyenv install 3.11.7
pyenv global 3.11.7

# Verify
python --version
```

### 3. Install Required System Libraries

Some packages require system-level dependencies:

```bash
# Install essential build tools
brew install cmake libomp

# Install optional but recommended libraries
brew install openblas lapack
```

## Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/Pg19g/ML3.git
cd ML3
```

### Step 2: Create Virtual Environment

**Using venv (Recommended):**

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Using conda (Alternative):**

```bash
# Install miniforge (conda for Apple Silicon)
brew install --cask miniforge

# Create environment
conda create -n ml3 python=3.11
conda activate ml3
```

### Step 3: Install Dependencies

**For Mac M3 Pro, use the optimized requirements file:**

```bash
# Install from Mac-specific requirements
pip install -r requirements-mac-m3.txt
```

**If you encounter issues, try installing problematic packages separately:**

```bash
# Install numpy and scipy first (they're dependencies for many packages)
pip install numpy scipy

# Install ML libraries
pip install scikit-learn lightgbm xgboost

# Install remaining packages
pip install -r requirements-mac-m3.txt
```

### Step 4: Verify Installation

```bash
# Test Python imports
python -c "import pandas, numpy, lightgbm, xgboost, streamlit; print('All core packages imported successfully!')"

# Run tests
pytest tests/ -v
```

### Step 5: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your EODHD API key
nano .env  # or use your preferred editor
```

Add your API key:
```
EODHD_API_KEY=your_api_key_here
TZ=UTC
```

## Mac M3 Specific Considerations

### 1. LightGBM and XGBoost

Both LightGBM and XGBoost have native ARM64 wheels and work great on M3:

- **LightGBM**: Uses OpenMP for parallelization (installed via Homebrew)
- **XGBoost**: Fully optimized for Apple Silicon

If you encounter build issues:

```bash
# For LightGBM
brew install libomp
pip install lightgbm --no-cache-dir

# For XGBoost
pip install xgboost --no-cache-dir
```

### 2. ONNX Runtime

Use the standard `onnxruntime` package (not `onnxruntime-silicon`):

```bash
pip install onnxruntime>=1.16.0
```

The standard package now includes ARM64 support.

### 3. Polars

Polars has excellent ARM64 support. Use version 0.20.0 or later:

```bash
pip install polars>=0.20.0
```

### 4. NumPy and SciPy

These packages are optimized for Apple Silicon using the Accelerate framework:

```bash
# Install with optimized BLAS
pip install numpy scipy
```

For even better performance, you can use OpenBLAS:

```bash
brew install openblas
OPENBLAS="$(brew --prefix openblas)" pip install numpy scipy --no-binary :all:
```

### 5. Streamlit

Streamlit works perfectly on M3. If you encounter issues:

```bash
pip install streamlit --upgrade
```

### 6. Memory Management

M3 Pro has unified memory. For large datasets:

```python
# In your code, you can leverage this by setting appropriate chunk sizes
import pandas as pd
pd.set_option('mode.chained_assignment', None)

# For Polars, use lazy evaluation
import polars as pl
df = pl.scan_parquet('large_file.parquet').collect()
```

## Performance Optimization

### 1. Enable Multi-Threading

The M3 Pro has excellent multi-core performance. Ensure packages use all cores:

```python
# In your Python code or config
import os
os.environ['OMP_NUM_THREADS'] = '12'  # M3 Pro has 12 cores (6P + 6E)
```

Or set in your shell:

```bash
export OMP_NUM_THREADS=12
```

### 2. Use Polars for Large Datasets

Polars is optimized for Apple Silicon and can be faster than Pandas:

```python
# The codebase already uses Polars where appropriate
# You can switch more operations to Polars if needed
```

### 3. GPU Acceleration (Optional)

While ML3 doesn't require GPU, if you want to experiment:

```bash
# Install TensorFlow for Apple Silicon (optional)
pip install tensorflow-macos tensorflow-metal
```

## Troubleshooting

### Issue: "Architecture not supported" error

**Solution:**
```bash
# Check your Python architecture
python -c "import platform; print(platform.machine())"
# Should output: arm64

# If it shows x86_64, you're using Rosetta
# Reinstall Python using Homebrew or pyenv
```

### Issue: LightGBM build fails

**Solution:**
```bash
# Install OpenMP
brew install libomp

# Set environment variables
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Install LightGBM
pip install lightgbm --no-cache-dir
```

### Issue: XGBoost build fails

**Solution:**
```bash
# Use pre-built wheel
pip install xgboost --only-binary :all:
```

### Issue: Streamlit won't start

**Solution:**
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/

# Reinstall
pip install streamlit --upgrade --force-reinstall
```

### Issue: "Symbol not found" errors

**Solution:**
```bash
# Reinstall problematic package
pip uninstall <package-name>
pip install <package-name> --no-cache-dir
```

### Issue: Slow performance

**Solution:**
```bash
# Ensure you're not running under Rosetta
arch
# Should output: arm64

# Check Python architecture
python -c "import platform; print(platform.machine())"
# Should output: arm64

# If x86_64, reinstall Python natively
```

## Running ML3 on Mac M3

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run the pipeline
make ingest
make build
make train

# Launch dashboard
make dashboard
# Open browser to http://localhost:8501

# Launch API
make api
# API available at http://localhost:8000
```

### Performance Tips

1. **Use all cores**: M3 Pro has 12 cores (6 performance + 6 efficiency)
   ```bash
   export OMP_NUM_THREADS=12
   ```

2. **Monitor memory**: Use Activity Monitor to check memory usage
   - ML3 should use < 8GB for typical workloads

3. **SSD optimization**: Ensure data is on internal SSD for best I/O performance

4. **Background apps**: Close unnecessary apps during training

## Benchmarks on M3 Pro

Expected performance (approximate):

- **Data Ingestion**: 100 symbols in ~3-4 minutes
- **PIT Processing**: 1M rows in ~15-20 seconds
- **Feature Engineering**: 1M rows in ~1-2 minutes
- **Model Training**: 5-fold CV in ~3-5 minutes
- **Backtesting**: 1M predictions in ~30-45 seconds

The M3 Pro's unified memory architecture provides excellent performance for data-intensive operations.

## Additional Resources

- **Homebrew**: https://brew.sh/
- **Python on Mac**: https://docs.python.org/3/using/mac.html
- **Apple Silicon Guide**: https://github.com/apple/ml-stable-diffusion
- **LightGBM on Mac**: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html

## Getting Help

If you encounter issues specific to Mac M3:

1. Check this guide first
2. Search GitHub issues: https://github.com/Pg19g/ML3/issues
3. Create a new issue with:
   - Mac model and OS version
   - Python version (`python --version`)
   - Architecture (`arch` and `python -c "import platform; print(platform.machine())"`)
   - Error message and full traceback

## Next Steps

After successful installation:

1. **Configure Universe**: Edit `config/universe.yaml` with your symbols
2. **Set API Key**: Add EODHD_API_KEY to `.env`
3. **Read User Guide**: See `userGuide.md` for detailed usage
4. **Run Tests**: `pytest tests/` to verify everything works
5. **Start Pipeline**: `make ingest && make build && make train`

---

**Note**: This guide is specifically for Mac M3 Pro. For other systems, see the main README.md.
