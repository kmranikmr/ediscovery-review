# Complete GitHub + Azure VM Setup Guide

## Overview
This guide sets up automatic deployment from GitHub to your existing Azure VM with repository mirroring to a secondary account.

## 🎯 What You'll Achieve
1. **GitHub Repository**: Code stored and versioned
2. **Automatic Deployment**: Push to main → deploys to Azure VM
3. **Repository Mirroring**: Automatic sync to secondary GitHub account
4. **Service Management**: FastAPI + Streamlit running on Azure VM
5. **Monitoring**: Logs and health checks

## 📋 Prerequisites Checklist

### Azure VM Requirements
- [ ] Ubuntu 20.04+ VM running
- [ ] SSH access configured
- [ ] Python 3.11+ installed
- [ ] 8GB+ RAM (16GB+ recommended)
- [ ] Ports 8001, 8501 accessible
- [ ] Git installed

### GitHub Requirements
- [ ] Primary GitHub repository created
- [ ] Secondary GitHub account/repository for mirroring
- [ ] SSH keys generated

## 🚀 Step-by-Step Setup

### 1. Prepare Your Local Repository

```bash
# In your project directory
cd /home/bbk/anil/from164/src/llm-retrieval-system/staging

# Initialize git (if not already done)
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: LLM Retrieval System with Family QA"

# Add your GitHub remote
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

# Push to GitHub
git push -u origin main
```

### 2. Setup Your Azure VM

SSH into your Azure VM and run:

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git llm-retrieval-system
cd llm-retrieval-system

# Run the setup script
chmod +x scripts/deploy-to-vm.sh
./scripts/deploy-to-vm.sh

# Edit environment variables
nano .env
# Add your OpenAI API key and other configuration
```

### 3. Configure GitHub Secrets

Go to your repository's Settings → Secrets and variables → Actions, and add:

| Secret Name | Value | Description |
|-------------|--------|-------------|
| `AZURE_VM_HOST` | `your.vm.ip.address` | Your Azure VM's public IP |
| `AZURE_VM_USERNAME` | `your-username` | Your VM username |
| `AZURE_VM_SSH_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` | Your private SSH key |
| `AZURE_VM_PORT` | `22` | SSH port (optional, defaults to 22) |
| `MIRROR_REPO_URL` | `git@github.com:mirror-account/repo.git` | Mirror repository URL |
| `MIRROR_SSH_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` | SSH key for mirror repo |

### 4. Generate SSH Keys

On your Azure VM:

```bash
# Generate SSH key for GitHub Actions
ssh-keygen -t rsa -b 4096 -C "github-actions" -f ~/.ssh/github_actions
cat ~/.ssh/github_actions.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/github_actions  # Copy this to AZURE_VM_SSH_KEY

# Generate SSH key for repository mirroring
ssh-keygen -t rsa -b 4096 -C "mirror-repo" -f ~/.ssh/mirror_repo
cat ~/.ssh/mirror_repo.pub  # Add to mirror repo's deploy keys
cat ~/.ssh/mirror_repo      # Copy this to MIRROR_SSH_KEY
```

### 5. Setup Repository Mirroring

#### Create Mirror Repository:
1. Create new repository in secondary GitHub account
2. Go to Settings → Deploy keys
3. Add the public key from step 4
4. Enable "Allow write access"

#### Test SSH Connection:
```bash
ssh -T git@github.com -i ~/.ssh/mirror_repo
```

### 6. Test the Complete Workflow

1. **Make a change** to your code
2. **Commit and push**:
   ```bash
   git add .
   git commit -m "Test automatic deployment"
   git push origin main
   ```
3. **Watch GitHub Actions** in your repository's Actions tab
4. **Verify deployment** on your Azure VM
5. **Check mirror repository** for automatic sync

## 🎛️ Usage Commands

### Local Development
```bash
# Start locally
./scripts/start-local.sh

# Stop services
./scripts/stop-services.sh
```

### Azure VM Management
```bash
# SSH into VM
ssh your-username@your-vm-ip

# Check services
sudo systemctl status llm-retrieval-api.service
sudo systemctl status llm-retrieval-streamlit.service

# View logs
tail -f logs/main.log
tail -f logs/streamlit.log

# Manual restart
sudo systemctl restart llm-retrieval-api.service
sudo systemctl restart llm-retrieval-streamlit.service
```

## 🌐 Access Your Applications

After successful deployment:
- **FastAPI API**: `http://your-vm-ip:8001`
- **Streamlit UI**: `http://your-vm-ip:8501`  
- **API Documentation**: `http://your-vm-ip:8001/docs`

## 🔄 Workflow Summary

```mermaid
graph LR
    A[Local Development] --> B[Git Push to Main]
    B --> C[GitHub Actions Triggered]
    C --> D[Deploy to Azure VM]
    C --> E[Mirror to Secondary Repo]
    D --> F[Restart Services]
    F --> G[Application Live]
```

## 🐛 Troubleshooting

### GitHub Actions Fails
- Check SSH key permissions
- Verify Azure VM is accessible
- Check GitHub secrets are correct

### Services Won't Start
```bash
# Check logs
tail -f logs/main.log
sudo journalctl -u llm-retrieval-api.service

# Check ports
sudo netstat -tlnp | grep :8001
sudo netstat -tlnp | grep :8501

# Manual restart
./scripts/deploy-to-vm.sh
```

### Repository Mirroring Fails
- Verify SSH key has write access to mirror repo
- Test SSH connection manually
- Check mirror repository permissions

## 📊 Monitoring

### Service Health
```bash
# Quick status check
curl http://your-vm-ip:8001/
curl http://your-vm-ip:8501/

# Process check
ps aux | grep python
```

### Logs
```bash
# Application logs
tail -f logs/main.log
tail -f logs/streamlit.log

# System service logs
sudo journalctl -u llm-retrieval-api.service -f
sudo journalctl -u llm-retrieval-streamlit.service -f
```

## 🔄 Continuous Iteration

After initial setup, your workflow becomes:

1. **Develop locally** with `./scripts/start-local.sh`
2. **Test your changes**
3. **Commit and push** to trigger automatic deployment
4. **Monitor deployment** in GitHub Actions
5. **Verify on Azure VM**
6. **Iterate** and repeat

The NER with BERT functionality and all other features are preserved and will work exactly as before, now with automatic deployment capabilities!

## 🎉 Next Steps

1. **Test all endpoints** to ensure everything works
2. **Setup monitoring/alerting** if needed
3. **Configure SSL certificates** for production
4. **Setup backup strategies** for your data
5. **Add integration tests** to GitHub Actions
6. **Scale as needed** with load balancers/multiple VMs
