# Your First Pull Request - Adding yourself as a contributor

This is a simple end-to-end guide for your first PR in CobraBox.

## 1) Set up your local repo

Follow:
`docs/setup_repo.md`

## 2) Create a new branch

Let's make sure we're on main (the checkout command is switching between branches, just like cd switches between folders)
```bash
git checkout main
```

Then let's ensure we're up do date. We'll pull (I.e. download the latest changes to the repo)
```bash
git pull
```

Then let's make a new branch. It's nice to have something that's informative.

```bash
git branch add_myname
```

Then let's check out your new branch
```bash
git checkout add_myname
```

### Note

In the future you can use this handy command to both create and checkout a branch:
```bash
git checkout -b add_myname
```

## 3) Add your name to contributors

Create a file with your name:
`contributors/yourname.txt`

(This is not how we'll actually represent contributors in the end it's just serves as an example of an edit for when making a pull request)

## 4) Commit your changes

```bash
git add .
git commit -m "add contributor [my name]"
```

## 5) Push your branch

```bash
git push -u origin add_myname 
```

## 6) Open the pull request

Use this simple guide:
`docs/how_to_make_a_pr.md`

Official GitHub guide:
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request

## 7) Respond to review feedback

Make updates locally, commit, and push again. The PR updates automatically.
