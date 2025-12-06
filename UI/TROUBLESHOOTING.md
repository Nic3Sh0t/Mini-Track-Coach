# TrackCoach UI Troubleshooting Guide

## Issue: ERR_CONNECTION_REFUSED

If you encounter `ERR_CONNECTION_REFUSED` error, it means the Streamlit server did not start successfully. Please follow these steps to troubleshoot:

### Step 1: Launch in Debug Mode

Run `run_ui_debug.bat` instead of `run_ui.bat`, which will display detailed error messages.

### Step 2: Check Common Issues

#### Issue 1: Python Path Problem

**Symptoms**: Import errors, module not found

**Solution**:
```bash
# Ensure running in UI directory
cd UI
python -c "import sys; print(sys.path)"
```

#### Issue 2: trackcoach Module Not Found

**Symptoms**: `ModuleNotFoundError: No module named 'trackcoach'`

**Solution**:
1. Confirm that `trackcoach` directory exists in parent directory
2. Check if `trackcoach/__init__.py` file exists
3. Try manual import:
   ```python
   import sys
   sys.path.insert(0, r'path/to/parent/directory')
   from trackcoach import pipeline
   ```

#### Issue 3: Port Already in Use

**Symptoms**: Port 8501 is already in use

**Solution**:
```bash
# Check port usage
netstat -ano | findstr :8501

# Or use a different port
streamlit run app.py --server.port 8502
```

#### Issue 4: Firewall Blocking

**Symptoms**: Connection refused, but Streamlit shows it started

**Solution**:
1. Check Windows firewall settings
2. Allow Python through firewall
3. Try using `127.0.0.1:8501` instead of `localhost:8501`

### Step 3: Manual Testing

Run test script:
```bash
cd UI
python test_app.py
```

This will check:
- Whether all dependencies are installed
- Whether path settings are correct
- Whether app.py can be imported normally

### Step 4: View Streamlit Logs

Streamlit will display detailed logs in the console. Look for:
- `You can now view your Streamlit app in your browser.`
- If there are errors, they will be displayed in the logs

### Step 5: Check Streamlit Configuration

Create or edit `~/.streamlit/config.toml`:
```toml
[server]
headless = false
port = 8501
enableCORS = false
```

### Common Error Messages and Solutions

#### Error: `ImportError: cannot import name 'X' from 'Y'`

**Cause**: Module import failed

**Solution**: Check if related modules are correctly installed and paths are correct

#### Error: `FileNotFoundError: [Errno 2] No such file or directory`

**Cause**: Path setting error

**Solution**: Check path settings in `app.py`, ensure all directories exist

#### Error: `AttributeError: module 'X' has no attribute 'Y'`

**Cause**: Module version mismatch or import error

**Solution**: Reinstall related dependencies

### If Problem Persists

1. **View Complete Error Log**: Run `run_ui_debug.bat` and copy all output
2. **Check Python Version**: Ensure using Python 3.7+
   ```bash
   python --version
   ```
3. **Reinstall Dependencies**:
   ```bash
   pip install --upgrade streamlit
   pip install -r requirements_ui.txt
   ```
4. **Try Minimal Test**: Create a simple Streamlit app test
   ```python
   # test_simple.py
   import streamlit as st
   st.write("Hello World")
   ```
   Then run: `streamlit run test_simple.py`

### Contact Support

If none of the above methods solve the problem, please provide:
1. Complete error log
2. Python version
3. Operating system version
4. Output from `test_app.py`
