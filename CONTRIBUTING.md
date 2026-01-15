
# How to contribute to Optimum?

Optimum-rbln is an open source project, so all contributions and suggestions are welcome.

You can contribute in many different ways: giving ideas, answering questions, reporting bugs, proposing enhancements, improving the documentation, fixing bugs,...

Many thanks in advance to every contributor.

## How to work on an open Issue?

> If you want to ask a question, we assume that you have read the available [Documentation](https://docs.rbln.ai/software/optimum/optimum_rbln.html).

Before you ask a question, it is best to search for existing [Issues](/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/rbln-sw/optimum-rbln/issues/new/choose).
- Provide as much context as you can about what you're running into.

We will then take care of the issue as soon as possible.

## How to create a Pull Request?
1. Fork the [repository](https://github.com/rbln-sw/optimum-rbln) by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

	```bash
	git clone git@github.com:<your Github handle>/optimum-rbln.git
	cd optimum-rbln
	git remote add upstream https://github.com/rbln-sw/optimum-rbln.git
	```

3. Create a new branch to hold your development changes:

	```bash
	git checkout -b a-descriptive-name-for-my-changes
	```

	**do not** work on the `main` branch.

4. Set up a development environment by running the following command in a virtual environment:

	```bash
	pip install -e ".[dev]"
	```

   (If optimum-rbln was already installed in the virtual environment, remove
   it with `pip uninstall optimum-rbln` before reinstalling it in editable
   mode with the `-e` flag.)

5. Develop the features on your branch.

6. Format your code. Run ruff so that your newly added files look nice with the following command:

	```bash
	ruff format .
	ruff check . --fix
	```

7.  Once you're happy with your changes, add the changed files using `git add` and make a commit with `git commit` to record your changes locally:

	```bash
	git add modified_file.py
	git commit -m "Your commit message"
	```

	It is a good idea to sync your copy of the code with the original
	repository regularly. This way you can quickly account for changes:

	```bash
	git fetch upstream
	git rebase upstream/main
    ```

   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

8. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send your to the project maintainers for review.

## Code of conduct

This project adheres to the Rebellions' [code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

