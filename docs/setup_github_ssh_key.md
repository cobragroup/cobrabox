# How to Set Up an SSH Key for GitHub

Simple Linux/macOS workflow:

## 1) Check if you already have an SSH key

```bash
ls -al ~/.ssh
```

If you already have `id_ed25519.pub`, you can reuse it.

## 2) Generate a new SSH key (if needed)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to use the default path.

## 3) Start ssh-agent and add your key

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## 4) Copy your public key

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the full output line.

## 5) Add key in GitHub

1. Open GitHub -> **Settings** -> **SSH and GPG keys**
2. Click **New SSH key**
3. Paste your public key
4. Click **Add SSH key**

## 6) Test the SSH connection

```bash
ssh -T git@github.com
```

You should see a success/authenticated message.

Official GitHub guide:
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
