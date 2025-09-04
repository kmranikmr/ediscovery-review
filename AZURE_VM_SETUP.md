# Azure VM Deployment Guide

## Prerequisites

Since you already have an Azure VM running, ensure it meets these requirements:

### VM Requirements
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **Python**: 3.11+ (Python 3.12 recommended)
- **RAM**: Minimum 8GB (16GB+ recommended for BERT models)
- **Storage**: Minimum 20GB free space
- **Network**: Ports 8001 and 8501 accessible

### Required Software
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 and dependencies
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip git curl

# Install build tools for ML libraries
sudo apt install -y build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev
```

## Quick Setup on Your Azure VM

### 1. Connect to Your VM
```bash
# SSH into your Azure VM (replace with your VM details)
ssh your-username@your-vm-ip-address
```

### 2. Clone and Setup Application
```bash
# Create application directory
mkdir -p ~/llm-retrieval-system
cd ~/llm-retrieval-system

# Clone your repository (replace with your actual repo URL)
git clone https://github.com/your-username/your-repo-name .

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Create environment file
cp .env.template .env
# Edit .env file with your API keys and configuration
nano .env
```

### 3. Configure Environment Variables
Edit your `.env` file:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your-actual-openai-api-key

# OpenSearch Configuration (if using)
OPENSEARCH_ENDPOINT=http://localhost:9200

# Ollama Configuration (if using)
OLLAMA_BASE_URL=http://localhost:11434

# Debug Mode
DEBUG_MODE=false

# Azure VM specific settings
SKIP_OPENSEARCH=true
SKIP_OLLAMA=false
```

### 4. Configure Firewall
```bash
# Allow necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 8001/tcp # FastAPI
sudo ufw allow 8501/tcp # Streamlit

# Enable firewall if not already enabled
sudo ufw --force enable

# Check firewall status
sudo ufw status
```

### 5. Test Manual Startup
```bash
# Activate virtual environment
source venv/bin/activate

# Test FastAPI server
python main.py &
FASTAPI_PID=$!

# Wait a moment and test
sleep 5
curl http://localhost:8001/

# Test Streamlit (in another terminal or background)
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &
STREAMLIT_PID=$!

# Check if both are running
ps aux | grep python
```

### 6. Setup Systemd Services (Optional but Recommended)
```bash
# Create FastAPI service
sudo tee /etc/systemd/system/llm-retrieval-api.service > /dev/null <<EOF
[Unit]
Description=LLM Retrieval System FastAPI Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/llm-retrieval-system
Environment=PATH=$HOME/llm-retrieval-system/venv/bin
ExecStart=$HOME/llm-retrieval-system/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:$HOME/llm-retrieval-system/logs/main.log
StandardError=append:$HOME/llm-retrieval-system/logs/main.log

[Install]
WantedBy=multi-user.target
EOF

# Create Streamlit service
sudo tee /etc/systemd/system/llm-retrieval-streamlit.service > /dev/null <<EOF
[Unit]
Description=LLM Retrieval System Streamlit UI
After=network.target llm-retrieval-api.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/llm-retrieval-system
Environment=PATH=$HOME/llm-retrieval-system/venv/bin
ExecStart=$HOME/llm-retrieval-system/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10
StandardOutput=append:$HOME/llm-retrieval-system/logs/streamlit.log
StandardError=append:$HOME/llm-retrieval-system/logs/streamlit.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable llm-retrieval-api.service
sudo systemctl enable llm-retrieval-streamlit.service

# Start services
sudo systemctl start llm-retrieval-api.service
sudo systemctl start llm-retrieval-streamlit.service

# Check service status
sudo systemctl status llm-retrieval-api.service
sudo systemctl status llm-retrieval-streamlit.service
```

## GitHub Integration

### 1. Configure GitHub Secrets
In your GitHub repository, go to Settings > Secrets and variables > Actions, and add:

```
AZURE_VM_HOST=your-vm-ip-address
AZURE_VM_USERNAME=your-vm-username  
AZURE_VM_SSH_KEY=your-private-ssh-key-content
AZURE_VM_PORT=22
MIRROR_REPO_URL=git@github.com:mirror-account/repo-name.git
MIRROR_SSH_KEY=ssh-key-for-mirror-repo
```

### 2. Generate SSH Key for GitHub Actions
On your Azure VM:
```bash
# Generate SSH key for GitHub Actions (don't use passphrase)
ssh-keygen -t rsa -b 4096 -C "github-actions@your-domain.com" -f ~/.ssh/github_actions

# Add public key to authorized_keys
cat ~/.ssh/github_actions.pub >> ~/.ssh/authorized_keys

# Copy private key content for GitHub secret
cat ~/.ssh/github_actions
# Copy this entire output to AZURE_VM_SSH_KEY secret
```

### 3. Setup Mirror Repository
```bash
# Create SSH key for mirror repository
ssh-keygen -t rsa -b 4096 -C "mirror@your-domain.com" -f ~/.ssh/mirror_repo

# Add the public key to the mirror repository's deploy keys
cat ~/.ssh/mirror_repo.pub
# Add this to the mirror repository's Settings > Deploy keys

# Copy private key for GitHub secret
cat ~/.ssh/mirror_repo
# Copy this to MIRROR_SSH_KEY secret
```

## Deployment Commands

### Manual Deployment
```bash
# SSH into VM
ssh your-username@your-vm-ip

# Navigate to app directory
cd ~/llm-retrieval-system

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Restart services using systemd
sudo systemctl restart llm-retrieval-api.service
sudo systemctl restart llm-retrieval-streamlit.service

# Check logs
tail -f logs/main.log
tail -f logs/streamlit.log
```

### Automated Deployment via GitHub Actions
Once GitHub secrets are configured:
1. Push changes to `main` branch
2. GitHub Actions will automatically:
   - Deploy to your Azure VM
   - Restart services
   - Mirror to secondary repository

## Monitoring and Maintenance

### Check Service Status
```bash
# Check if services are running
sudo systemctl status llm-retrieval-api.service
sudo systemctl status llm-retrieval-streamlit.service

# Check processes
ps aux | grep python
ps aux | grep streamlit

# Check ports
sudo netstat -tlnp | grep :8001
sudo netstat -tlnp | grep :8501
```

### View Logs
```bash
# Real-time logs
tail -f logs/main.log
tail -f logs/streamlit.log

# Service logs via systemd
sudo journalctl -u llm-retrieval-api.service -f
sudo journalctl -u llm-retrieval-streamlit.service -f
```

### Restart Services
```bash
# Restart individual services
sudo systemctl restart llm-retrieval-api.service
sudo systemctl restart llm-retrieval-streamlit.service

# Or restart both
sudo systemctl restart llm-retrieval-api.service llm-retrieval-streamlit.service
```

## Accessing Your Application

After deployment, access your applications at:
- **FastAPI API**: `http://your-vm-ip:8001`
- **Streamlit UI**: `http://your-vm-ip:8501`
- **API Documentation**: `http://your-vm-ip:8001/docs`

## Troubleshooting

### Common Issues

1. **Port Access Issues**
   ```bash
   # Check if ports are open
   sudo ufw status
   # Make sure ports 8001 and 8501 are allowed
   ```

2. **Service Won't Start**
   ```bash
   # Check logs
   sudo journalctl -u llm-retrieval-api.service --no-pager
   
   # Check if virtual environment is correct
   source venv/bin/activate
   which python
   ```

3. **Permission Issues**
   ```bash
   # Fix ownership
   sudo chown -R $USER:$USER ~/llm-retrieval-system
   chmod +x scripts/*.sh
   ```

4. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   
   # If low on memory, consider reducing BERT model usage
   # Set SKIP_BERT=true in .env file
   ```

### Log Analysis
```bash
# Search for errors in logs
grep -i error logs/main.log
grep -i error logs/streamlit.log

# Check recent log entries
tail -n 100 logs/main.log
```

This setup will give you a robust deployment on your existing Azure VM with automatic GitHub integration!
