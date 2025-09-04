# Simple Azure Deployment Setup

## 🎯 Direct Deployment to Azure VM

Your repository (kmranikmr/ediscovery-review) will directly deploy to your Azure VM when you push to the master branch.

## 🔑 Required Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

| Secret Name | Value | Description |
|-------------|--------|-------------|
| `AZURE_VM_HOST` | `your.vm.ip.address` | Your Azure VM's public IP address |
| `AZURE_VM_USERNAME` | `your-username` | Your Azure VM username |
| `AZURE_VM_SSH_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` | Your SSH private key for VM access |
| `AZURE_VM_PORT` | `22` | SSH port (optional, defaults to 22) |

## 🔐 Generate SSH Key for Azure VM (if needed)

If you don't have an SSH key for your Azure VM:

```bash
# Generate SSH key
ssh-keygen -t rsa -b 4096 -C "github-actions-azure" -f ~/.ssh/github_actions_azure

# Copy public key to Azure VM
ssh-copy-id -i ~/.ssh/github_actions_azure.pub your-username@your-vm-ip

# Get private key for GitHub secret
cat ~/.ssh/github_actions_azure
```

## 🚀 How to Deploy

1. **Make your changes** in your local repository
2. **Commit and push**:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin master
   ```
3. **Watch GitHub Actions** in your repository's Actions tab
4. **Access your applications**:
   - FastAPI: `http://your-vm-ip:8001`
   - Streamlit: `http://your-vm-ip:8501`

## ✅ What Happens During Deployment

1. **GitHub Actions triggers** on push to master
2. **Code is pulled** to your Azure VM
3. **Dependencies are installed** in a virtual environment
4. **Services are restarted** (FastAPI + Streamlit)
5. **Health checks** verify everything is running

## 🐛 If Something Goes Wrong

Check the GitHub Actions logs in your repository's Actions tab. The deployment script will show detailed logs for troubleshooting.

## 🎉 Ready to Go!

Once you add the 4 secrets to your GitHub repository, you can push your code and it will automatically deploy to your Azure VM!

No mirroring, no complexity - just simple, direct deployment.
