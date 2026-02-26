# How to Make a Pull Request

Simple step-by-step (GitHub website):

1. Push your branch to GitHub.
   ```bash
   git push -u origin <your-branch-name>
   ```
2. Open the repository on GitHub.
3. Click **Compare & pull request** (or go to the **Pull requests** tab and click **New pull request**).
4. Set:
   - **base** = `main`
   - **compare** = your feature branch
5. Add a clear title and short description of:
   - what changed
   - how you tested it (if relevant)
6. Click **Create pull request**.
7. Ask for review from the team.

Once you see some messages saying "Review required", "Merging is blocked" with some big scary traffic signs. Then you're done! Good job! Now it's up to the person reviewing the pull request to look at the code, and possibly give you feedback.


For the official detailed guide, see GitHub Docs:
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
