# Contributing to afrr-remuneration

First off, thanks for you interest in contributing to this package.

We organize our work with [issues](https://guides.github.com/features/issues/) which then are resolved by merging pull 
requests.

## Creating an issue

Each planned piece of work is described in a separate issue - bug reports and feature requests.

### Bug reports

If you found a bug, we would love to hear about! Please tell

- what it is about
- how to reproduce (ideally, add a minimal example)
- details about your working environment

### Feature requests 

Please describe precisely what is the intention of the new feature and why it is important. If you already know how 
to implement, add a list of task that are required to implement.

## Submitting a pull requests

For fixing a bug or adding a new feature, please start from the tip of the `main` branch.
Thereof, create a new branch and use this naming pattern

- `fixes/#<issue-number>-really-condensed-title` for bug fixes
- `features/#<issue-number>-really-condensed-title` for feature requests

On GitHub, you can open a PR from any branch.

You can also fork the repository and create a PR from this fork. This is particularly helpful if you don't have the 
permissions to create new branches in the repository (yet!).

The pull request template helps you to write a good PR message.
Pull requests (PR) are merged after code reviews. At least one reviewer needs to accept the PR before merging.
By [mentioning the related issue](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#referencing-issues-and-pull-requests) 
the issue gets closed on merge automatically.

# Releasing new versions

A publishing workflow gets triggered when a [new release](https://github.com/energy2market/afrr-remuneration/releases) 
is created on GitHub. The version number is read from the git tag using `poetry-dynamic-versioning`.