# Mirror-First Deployment Setup Guide

## 🎯 New Architecture Overview

**Primary Repository** (kmranikmr/ediscovery-review):
- Development and code storage
- Mirrors code to deployment repository
- NO Azure deployment

**Mirror Repository** (secondary account):
- Receives mirrored code automatically
- Runs GitHub Actions to deploy to Azure VM
- Handles all deployment logic

## 🚀 Setup Instructions

### 1. Setup the Mirror Repository

1. **Create a new repository** in the secondary GitHub account
2. **Clone this current repository** to get the deployment workflow
3. **Copy the deployment workflow** to the mirror repository

### 2. Configure Primary Repository (kmranikmr/ediscovery-review)

Replace the current workflow with the mirror-only workflow:

```bash
# In your current repository
rm .github/workflows/deploy.yml
mv .github/workflows/mirror-only.yml .github/workflows/deploy.yml
```

Add these secrets to the **primary repository**:

| Secret Name | Value | Description |
|-------------|--------|-------------|
| `DEPLOYMENT_REPO_URL` | `git@github.com:mirror-account/repo.git` | Mirror repository SSH URL |
| `DEPLOYMENT_SSH_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` | SSH key for mirror repo |
| `DEPLOYMENT_REPO_TOKEN` | `ghp_xxx...` | GitHub token for mirror repo |
| `DEPLOYMENT_REPO_NAME` | `mirror-account/repo-name` | Mirror repository name |

### 3. Configure Mirror Repository

Copy the deployment workflow to your mirror repository:

```bash
# In the mirror repository
mkdir -p .github/workflows
cp deploy-from-mirror.yml .github/workflows/deploy.yml
```

Add these secrets to the **mirror repository**:

| Secret Name | Value | Description |
|-------------|--------|-------------|
| `AZURE_VM_HOST` | `your.vm.ip.address` | Your Azure VM's public IP |
| `AZURE_VM_USERNAME` | `your-username` | Your VM username |
| `AZURE_VM_SSH_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` | SSH key for Azure VM |
| `AZURE_VM_PORT` | `22` | SSH port (optional) |

### 4. Generate Required SSH Keys

#### For Mirror Repository Access (Primary Repo → Mirror Repo):
```bash
ssh-keygen -t rsa -b 4096 -C "primary-to-mirror" -f ~/.ssh/primary_to_mirror
```
- Add **public key** to mirror repository's Deploy Keys (with write access)
- Add **private key** to primary repository's `DEPLOYMENT_SSH_KEY` secret

#### For Azure VM Access (Mirror Repo → Azure VM):
```bash
ssh-keygen -t rsa -b 4096 -C "mirror-to-azure" -f ~/.ssh/mirror_to_azure
```
- Add **public key** to Azure VM's `~/.ssh/authorized_keys`
- Add **private key** to mirror repository's `AZURE_VM_SSH_KEY` secret

#### For Repository Dispatch (Primary Repo → Mirror Repo):
1. Create a **Personal Access Token** in the mirror account with `repo` scope
2. Add token to primary repository's `DEPLOYMENT_REPO_TOKEN` secret

### 5. Test the Complete Flow

1. **Make a change** in your primary repository
2. **Commit and push**:
   ```bash
   git add .
   git commit -m "Test mirror deployment"
   git push origin master
   ```
3. **Watch the flow**:
   - Primary repo mirrors to secondary repo
   - Secondary repo automatically deploys to Azure VM

## 🔄 How It Works

```mermaid
graph LR
    A[Primary Repo<br/>kmranikmr/ediscovery-review] --> B[GitHub Actions<br/>Mirror Only]
    B --> C[Mirror Repository<br/>Secondary Account]
    C --> D[GitHub Actions<br/>Deploy to Azure]
    D --> E[Azure VM<br/>Running Services]
    
    A -.->|Repository Dispatch| C
    C -.->|Health Check| E
```

### Workflow Steps:

1. **Developer pushes** to primary repository
2. **Primary repo's GitHub Actions** mirror code to secondary repository
3. **Repository dispatch event** triggers deployment in mirror repository
4. **Mirror repo's GitHub Actions** deploy to Azure VM
5. **Services restart** on Azure VM automatically

## 🎛️ Usage

### Development Workflow:
```bash
# Work in primary repository as usual
git add .
git commit -m "Your changes"
git push origin master  # Triggers automatic deployment via mirror
```

### Monitor Deployments:
- **Primary repo**: Check mirroring status in Actions tab
- **Mirror repo**: Check deployment status in Actions tab
- **Azure VM**: Check services are running

### Manual Deployment (if needed):
```bash
# In mirror repository
git push origin main  # Triggers deployment
```

## 🐛 Troubleshooting

### Mirroring Fails:
- Check SSH key has write access to mirror repository
- Verify `DEPLOYMENT_REPO_URL` is correct SSH URL
- Test SSH connection manually

### Deployment Fails:
- Check Azure VM SSH access from mirror repository
- Verify Azure VM secrets in mirror repository
- Check Azure VM has required dependencies

### Repository Dispatch Fails:
- Verify GitHub token has `repo` scope
- Check `DEPLOYMENT_REPO_NAME` format is correct
- Ensure token belongs to mirror repository account

## ✅ Benefits of This Setup

1. **Separation of Concerns**: Development repo vs deployment repo
2. **Security**: Azure credentials only in deployment repository
3. **Flexibility**: Can have different deployment configurations
4. **Control**: Deployment repository can have additional security controls
5. **Clean History**: Development repo isn't cluttered with deployment commits

## 🔧 Next Steps

1. Test the complete flow end-to-end
2. Setup monitoring/alerting in mirror repository
3. Configure branch protection rules in mirror repository
4. Setup notifications for deployment status
5. Add integration tests to mirror repository if needed

The mirror repository becomes your **deployment control center** while keeping your main development clean and focused!
