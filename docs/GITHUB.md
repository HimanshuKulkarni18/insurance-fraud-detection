# Publishing to GitHub (securely)

## Do not share credentials in chat or commit them

- Never paste your **GitHub password**, **personal access tokens (PATs)**, or **SSH private keys** into messaging apps, email, or AI tools.  
- If someone asks for your GitHub password or token to “push for you,” **refuse** — use the steps below on **your own machine** only.

## Recommended: GitHub CLI

1. Install [GitHub CLI](https://cli.github.com/) (`gh`).
2. In the project folder:

   ```bash
   gh auth login
   ```

   Follow the prompts (HTTPS or SSH). This stores credentials in your OS keychain / credential helper — not in the repo.

3. Create the repository and push:

   ```bash
   cd insurance-fraud-detection
   git init
   git add .
   git commit -m "Initial commit: fraud detection prototype"
   gh repo create insurance-fraud-detection --private --source=. --push
   ```

   Adjust `--private` / `--public` as you like. If the repo already exists on GitHub:

   ```bash
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

## SSH keys (alternative)

1. Generate a key: `ssh-keygen -t ed25519 -C "your_email@example.com"`  
2. Add the **public** key (`*.pub`) in GitHub → **Settings → SSH and GPG keys**.  
3. Use the SSH remote URL: `git@github.com:YOUR_USER/YOUR_REPO.git`

## Personal access token (HTTPS)

If you use HTTPS with a PAT:

- Create a fine-scoped token in GitHub → **Settings → Developer settings**.  
- Use it only when Git prompts for a password, or configure the **Git credential manager** so the token is stored locally — **do not** add the token to `README`, code, or `.env` committed to git.

## What gets ignored

`.gitignore` excludes `data/raw/*.csv`, `artifacts/`, virtualenvs, and caches. Generated data and trained models stay local unless you choose to commit them (not recommended for large files — use [Git LFS](https://git-lfs.github.com/) or a release artifact instead).
