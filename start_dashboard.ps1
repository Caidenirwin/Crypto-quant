# Open the Crypto Quant dashboard (read-only) at http://localhost:8501
# Close this window (or press Ctrl+C) to stop the dashboard. The trading bot is unaffected.
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Start-Process "http://localhost:8501"
python -m streamlit run dashboard.py --server.port 8501
