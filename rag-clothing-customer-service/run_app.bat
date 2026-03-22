@echo off
REM Windows 启动脚本：上传文件版 Streamlit 应用（最简单版本）

REM 切换到当前脚本所在目录（rag-clothing-customer-service）
cd /d "%~dp0"

REM 直接使用当前环境中的 python 启动 Streamlit 应用
python -m streamlit run app_file_uploader.py --server.address 0.0.0.0 --server.port 8501

pause

