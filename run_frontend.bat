@echo off
echo ==================================================
echo 🚀 Starting Karachi AQI Prediction Dashboard
echo ==================================================

REM Activate virtual environment
call venv\Scripts\activate

REM Navigate to streamlit app directory
cd streamlit_app

REM Launch Streamlit app
streamlit run app.py

REM Pause so window stays open after exit
pause