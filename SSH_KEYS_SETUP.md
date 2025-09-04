# SSH Keys Setup Guide

## Your SSH Keys are Ready!

The SSH keys have been generated successfully. Here's exactly what to do:

## Step 1: Get the PUBLIC key for Mirror Repository

```bash
cat ~/.ssh/primary_to_mirror.pub
```

**What to do with this:**
1. Copy the entire output (starts with `ssh-rsa`)
2. Go to your **Mirror Repository** on GitHub
3. Go to Settings → Deploy keys → Add deploy key
4. Paste the public key and **check "Allow write access"**

## Step 2: Get the PRIVATE key for Primary Repository Secret

```bash
cat ~/.ssh/primary_to_mirror
```

**What to do with this:**
1. Copy the entire output (starts with `-----BEGIN OPENSSH PRIVATE KEY-----`)
2. Go to your **Primary Repository** (kmranikmr/ediscovery-review) on GitHub  
3. Go to Settings → Secrets and variables → Actions → New repository secret
4. Name: `DEPLOYMENT_SSH_KEY`
5. Value: Paste the entire private key

## Step 3: Other Secrets for Primary Repository

Add these to **kmranikmr/ediscovery-review** secrets:

| Secret Name | Value | Example |
|-------------|--------|---------|
| `DEPLOYMENT_REPO_URL` | `git@github.com:MIRROR-ACCOUNT/MIRROR-REPO.git` | `git@github.com:johnsmith/deployment-repo.git` |
| `DEPLOYMENT_SSH_KEY` | *Private key from Step 2* | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `DEPLOYMENT_REPO_TOKEN` | *GitHub token from mirror account* | `ghp_xxxxxxxxxxxx` |
| `DEPLOYMENT_REPO_NAME` | `MIRROR-ACCOUNT/MIRROR-REPO` | `johnsmith/deployment-repo` |

## Step 4: Replace Workflow in Primary Repository

```bash
# In your current directory
rm .github/workflows/deploy.yml
mv .github/workflows/mirror-only.yml .github/workflows/deploy.yml
```

## Ready to Test!

Once you have:
- ✅ Public key added to mirror repo Deploy Keys (with write access)
- ✅ Private key added to primary repo secrets as `DEPLOYMENT_SSH_KEY`
- ✅ Other secrets configured in primary repo
- ✅ Workflow replaced in primary repo

Then you can push to your primary repo and it will automatically mirror and deploy!

## Quick Commands to Get Your Keys:

```bash
# Public key (for mirror repo Deploy Keys)
echo "=== PUBLIC KEY FOR MIRROR REPO DEPLOY KEYS ==="
cat ~/.ssh/primary_to_mirror.pub
echo ""

# Private key (for primary repo DEPLOYMENT_SSH_KEY secret)
echo "=== PRIVATE KEY FOR PRIMARY REPO SECRET ==="
cat ~/.ssh/primary_to_mirror
echo ""
```
