#!/bin/bash
# AIn - Run Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}AIn${NC}"
echo "========================"

# Trust PyPI for corporate environments
export PIP_TRUSTED_HOST="pypi.org pypi.python.org files.pythonhosted.org"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate and install deps
echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

# Start backend in background
echo -e "${GREEN}Starting FastAPI backend on port 8000...${NC}"
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 2

# Start frontend
echo -e "${GREEN}Starting Streamlit frontend on port 8501...${NC}"
cd frontend
streamlit run app_unified.py --server.port 8501

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
