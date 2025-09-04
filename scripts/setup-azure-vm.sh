#!/bin/bash

# Azure VM Setup Script
# Run this script on your Azure VM to prepare it for deployment

set -e

echo "🚀 Setting up Azure VM for LLM Retrieval System deployment..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 if not available
echo "🐍 Installing Python 3.12..."
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip git curl

# Create application directory
APP_DIR="/home/$USER/llm-retrieval-system"
echo "📁 Creating application directory: $APP_DIR"
mkdir -p $APP_DIR
cd $APP_DIR

# Create logs directory
mkdir -p logs

# Install system dependencies for ML libraries
echo "📚 Installing system dependencies..."
sudo apt install -y build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev

# Create virtual environment
echo "🔧 Creating Python virtual environment..."
python3.12 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Create systemd service files for auto-start
echo "⚙️ Creating systemd service files..."

# FastAPI service
sudo tee /etc/systemd/system/llm-retrieval-api.service > /dev/null <<EOF
[Unit]
Description=LLM Retrieval System FastAPI Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
ExecStart=$APP_DIR/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/main.log
StandardError=append:$APP_DIR/logs/main.log

[Install]
WantedBy=multi-user.target
EOF

# Streamlit service
sudo tee /etc/systemd/system/llm-retrieval-streamlit.service > /dev/null <<EOF
[Unit]
Description=LLM Retrieval System Streamlit UI
After=network.target llm-retrieval-api.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
ExecStart=$APP_DIR/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/streamlit.log
StandardError=append:$APP_DIR/logs/streamlit.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Configure firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 8001/tcp # FastAPI
sudo ufw allow 8501/tcp # Streamlit
sudo ufw --force enable

# Create environment file template
echo "📄 Creating environment file template..."
cat > .env.template <<EOF
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# OpenSearch Configuration
OPENSEARCH_ENDPOINT=http://localhost:9200

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Debug Mode (set to true for development)
DEBUG_MODE=false

# Docker Environment Flags
SKIP_OPENSEARCH=false
SKIP_OLLAMA=false
EOF

echo "✅ Azure VM setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.template to .env and fill in your API keys"
echo "2. Set up your GitHub repository secrets:"
echo "   - AZURE_VM_HOST: $(curl -s ifconfig.me)"
echo "   - AZURE_VM_USERNAME: $USER"
echo "   - AZURE_VM_SSH_KEY: (your private SSH key)"
echo "   - AZURE_VM_PORT: 22"
echo "3. Push your code to trigger the first deployment"
echo ""
echo "🌐 After deployment, access your services at:"
echo "   - FastAPI: http://$(curl -s ifconfig.me):8001"
echo "   - Streamlit: http://$(curl -s ifconfig.me):8501"
