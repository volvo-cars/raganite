# Contributing

This guide covers the complete contribution lifecycle: obtain the source, install
the development environment, make and validate a change, write release notes,
push a branch, open a pull request, and understand what happens after merge.

Follow the repository's [contribution policy](https://github.com/volvo-cars/raganite/blob/main/CONTRIBUTING.md)
and [code of conduct](https://github.com/volvo-cars/raganite/blob/main/CODE_OF_CONDUCT.md).
For a substantial feature or API change, discuss the approach in an issue before
starting. Report suspected security vulnerabilities privately to the maintainers;
do not include credentials or sensitive reproduction data in a public issue.

## The short version

Once Git, uv, and package-registry access are configured:

```sh
git clone git@github.com:volvo-cars/raganite.git
cd raganite
git switch -c feature/123-short-description
uv sync --group dev --python 3.13

# Make the change, add tests, and add a changelog fragment when appropriate.
uv run ruff format src benchmark tests
uv run ruff check src benchmark tests
uv run mypy
uv run ty check
uv run pytest

git status --short
git add path/to/changed-file.py changes/123.added.md
git diff --cached
git commit -s -m "Add a short description of the change"
git push -u origin feature/123-short-description
```

Replace the example branch, issue number, and file paths with your own. The push
example assumes you have repository write access. Without it, use a fork as
described below. Never push a feature directly to `main`.

## 1. Prepare your machine

You need:

- Git and a GitHub account with access to this repository, or permission to fork it.
- Python **3.11, 3.12, or 3.13**. CI tests all three. Python 3.10 and 3.14 are
  outside the current `requires-python` range.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which manages
  Python installations, the virtual environment, dependencies, and commands.
- Network access and, where required, credentials for the package registry used
  by `uv.lock`. Corporate network/VPN access may be necessary.


Configure the identity used for your commits. You can set it per repository
after cloning instead of globally:

```sh
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```



## 2. Install the development environment

From the repository root:

```sh
uv sync --group dev --python 3.13
uv run python --version
uv run raganite --help
uv run raganite-bench --help
```

This creates `.venv`, installs Raganite in editable mode, and installs the
development tools: pytest, pytest-asyncio, Ruff, mypy, and ty. Editable mode means
your source edits are used without reinstalling for each change.

You do not need to activate `.venv` when using `uv run`. In JetBrains or another
editor, select `.venv/bin/python` as the interpreter on macOS/Linux
(`.venv\Scripts\python.exe` on Windows).

`uv sync` and `uv run` may refresh `uv.lock` when dependency metadata or the
configured index changes. CI permits this refresh instead of rejecting an
out-of-date lockfile. Review resulting changes and commit `pyproject.toml` and
`uv.lock` together when intentionally changing dependencies.

### Model credentials are optional for development

Unit/regression tests use local stand-ins, in-memory SQLite, and mocked model and
Neo4j clients. You do **not** need OpenAI/Azure credentials or a running Neo4j
server to run the test suite. Do not use a live benchmark as a substitute for it.

For a manual exercise that actually calls a model:

```sh
cp .env.example .env
```

Edit `.env` locally and follow [Model providers](guides/providers.md). The default
CLI uses OpenAI and SQLite; Azure settings and deployment names are explicit
overrides. `init` and `config` do not call a model. `clear` does not call a model
either, but **deletes data** and requires confirmation.

Never commit `.env`, provider keys, database passwords, registry tokens, private
corpora, or generated benchmark results. Do not paste complete provider error
responses into a PR without reviewing them for sensitive data.

The application CLI loads `.env` from the working directory unless disabled or
an explicit `--env-file` is supplied. Configuration precedence is CLI options,
process environment, dotenv, config file, then defaults. Library constructors
do not automatically load dotenv files. See [Configuration](guides/configuration.md).

## 3. Start a focused branch
Use a descriptive branch name such as `fix/123-sqlite-reader` or
`docs/123-provider-setup`. These prefixes are recommendations, not enforced
automation. Keep unrelated refactors, dependency upgrades, and generated files
out of the change unless they are necessary for the issue.

## 4. Make the change and add coverage

### What to test

- Add a regression test that fails before a bug fix and passes afterwards.
- Cover both successful and invalid inputs for new behavior.
- Exercise SQLite and mocked Neo4j behavior when changing a shared backend
  contract; preserve namespace isolation, read-only behavior, and ownership of
  injected resources.
- Use mocked HTTP/model clients for provider behavior. Do not make default tests
  depend on paid APIs, local credentials, network downloads, or a live database.
- For retrieval changes, check evidence, ranking, filtering, and configuration
  behavior—not only that a nonempty answer is returned.
- Update help text and CLI tests when options or defaults change.


### Changing dependencies

Use uv rather than manually editing `uv.lock`:

### Updating documentation

Update the page relevant to the behavior you changed. Keep tutorials,
task-oriented guides, API/CLI reference, and conceptual explanations distinct.
Examples must use the current constructors and commands. Add new pages to
`mkdocs.yml` when they belong in the site navigation.

MkDocs Material is not currently in the development dependency group. To
preview documentation without modifying project dependencies:

```sh
uv run --with mkdocs-material mkdocs serve
```

For a documentation build:

```sh
uv run --with mkdocs-material mkdocs build --strict
```

## 7. Add a changelog fragment

### Normal feature and bug-fix PRs

The pull-request template asks for a Towncrier fragment for user-facing changes.
Create a small file under `changes/`; create the directory if it is absent:

```sh
mkdir -p changes
```

The filename is `changes/<issue-or-pr-number>.<type>.md`. Replace `123` with the
actual issue or PR number. If you have neither yet, open a draft PR, obtain its
number, then add the fragment and push another commit.

| Type | Use for |
| --- | --- |
| `added` | New user-visible functionality |
| `changed` | Changes to existing behavior, defaults, or public APIs |
| `deprecated` | Functionality still available but planned for removal |
| `removed` | Removed functionality or public names |
| `fixed` | Corrections to existing behavior |
| `security` | Security-related fixes, coordinated with maintainers |

For example, `changes/123.fixed.md` could contain:

```markdown
Preserve document provenance when retrieving passages from an existing Neo4j graph.
```

Write from the user's perspective: what changed, why it matters, and any action
needed. Use complete sentences. Do not add a release heading, date, author name,
or raw issue link inside the fragment; Towncrier handles grouping and issue
references. If one PR has distinct change types, use separate fragments.

For breaking changes, describe the migration in the fragment and the PR, and
update the affected guides/reference pages. An internal refactor is not
automatically user-invisible if it removes an import path.

### Changes that do not need a fragment

The PR template permits an exemption for internal-only, documentation-only,
test-only, and release changes. Select that checkbox and explain why it applies.
If users will observe a behavior change, add a fragment even when most of the
diff looks like a refactor.

### Should I edit `CHANGELOG.md` directly?

For routine PRs, add a fragment rather than having every branch edit the same
section of the root changelog. Maintainers consolidate the fragments into
`CHANGELOG.md` when preparing a release.

The root changelog already contains bootstrap notes under **Unreleased** for the
initial release. Preserve those notes. A release editor must reconcile them
with newly generated notes so the published version appears once, without
duplicated entries. The documentation's changelog page is not the place to add
per-PR release notes.

Towncrier is configured in `pyproject.toml` but is **not installed by the dev
group**, and CI currently has no fragment-presence check. The fragment requirement
comes from the PR/review process, not an automated job.

To optionally preview the next initial release without changing the changelog or
removing fragments:

```sh
uv run --with towncrier towncrier build --draft --version <latest-version>
```

Use the version agreed with maintainers if it is no longer the latest one. As with the
documentation tool, Towncrier is resolved separately for this invocation.

## 8. Run the pre-push checks

From the repository root, run:

```sh
uv sync --group dev
uv run ruff format src benchmark tests
uv run ruff check src
uv run ty check src
uv run pytest
git diff --check
```


## 9. Review, stage, and commit

Inspect your working tree and the exact staged content:

The PR template requires Developer Certificate of Origin sign-off.
`git commit -s` adds a `Signed-off-by` trailer certifying your right to submit
the contribution. It is not the same as cryptographically signing a commit.
If you forgot the trailer on your own latest commit:

```sh
git commit --amend --no-edit --signoff
```

Amending changes the commit ID. If it was already pushed, follow the safe
force-push guidance below. Do not add someone else's sign-off on their behalf.

## 10. Push and open the pull request

```sh
git push -u origin feature/123-short-description
```

Open the PR URL printed by Git, or use GitHub's **Compare & pull request** button.
Choose the official repository's `main` as the base and your feature branch as
the head. Use a draft PR for unfinished work or early design feedback.

Fill in the repository PR template:

1. **Summary:** describe the problem, the chosen solution, and why it is needed.
2. **Type of change:** select the relevant categories.
3. **Issue link:** use `Closes #123` only if merging should close that issue;
   otherwise use a normal reference such as `Related to #123`.
4. **Validation:** list exact commands, results, and any checks not run with a
   reason. State whether a manual benchmark used live services.
5. **Checklist:** confirm tests/docs, dependencies, sensitive-data review,
   changelog fragment or exemption, and DCO sign-off.
6. **Migration notes:** describe changes to imports, settings, stored data, or
   other public behavior. Write `None` if there is no migration.
7. **Reviewer notes:** call out risk areas and deliberate limitations.

For retrieval/performance changes, include a reproducible comparison when
available: dataset, sample size, model/deployment configuration without secrets,
baseline, failures, and metric definitions. Do not claim benchmark parity from a
different retrieval mode or by silently excluding failed questions.

## 11. Understand CI and respond to review

The `CI` workflow runs on pull requests, merge-queue events, and manual dispatch.
It is also reused by the post-merge packaging workflow.

| Job | What it does |
| --- | --- |
| Lint and types | Installs dev dependencies on Python 3.13; runs Ruff on `src`, mypy targeting 3.11, and ty |
| Tests (Python 3.11/3.12/3.13) | Runs the offline pytest suite independently on each supported version |
| Required checks | Fails unless both the quality job and the entire test matrix succeed |

JUnit XML reports are retained as workflow artifacts for **14 days** when
available. Publishing credentials and live model credentials are not supplied
to the test jobs. New pushes normally cancel older in-progress CI runs for the
same PR/ref; inspect the newest run. A quality step can stop later steps, so a
missing ty result after an earlier failure is not a successful ty check.

Branch protection is a GitHub repository setting, not something a workflow file
enables by itself. Maintainers should require **Required checks** and the agreed
review approvals before merge. Do not assume approval counts or a specific
merge strategy are enforced unless the repository rules say so.

### If CI fails

1. Open the failed job and find the **first** failing command.
2. Distinguish environment setup failures from code/test failures.
3. Reproduce the command locally with the same dependency settings and matching Python.
4. Fix the issue, rerun the relevant checks, commit, and push to the same branch.
5. Explain the update in the PR and resolve review conversations only after
   addressing them. Request another review when ready.

Do not delete an assertion, skip a test, suppress a type error, or disable a job
just to obtain a green check. Document intentional exclusions with a technical
reason and reviewer agreement.

Fork PRs normally cannot access repository secrets. Because the lockfile uses a
private registry, their dependency-install step may need maintainer assistance
or an approved runner/access arrangement. Do not solve that by exposing registry
credentials to untrusted code or switching to privileged PR event handling.

## 13. Release preparation — maintainers

Ordinary feature PRs should not bump the package version unless a maintainer
asks. Prepare a focused release PR when the scope/version is agreed:

1. Review the `changes/` fragments and any existing bootstrap **Unreleased** notes.
2. Select the release version; update `[project].version` in `pyproject.toml`.
   Do not reuse a version already published to the target registry.
3. Refresh `uv.lock` with `uv lock`, then `uv sync --group dev`.
4. Preview release notes before modifying files. For the initial release:

   ```sh
   uv run --with towncrier towncrier build --draft --version 0.1.0
   ```

5. Generate notes using the agreed version only after reviewing the preview:

   ```sh
   uv run --with towncrier towncrier build --version 0.1.0
   ```

   Unlike `--draft`, this updates the changelog and consumes the included
   fragments. Review any prompts and the resulting diff. Reconcile existing
   initial-release notes, preserve previous releases, and record the actual
   intended release date; do not describe an unpublished version as already
   released.
6. Rerun the checks, build the distribution, and validate metadata locally:

   ```sh
   uv build
   uvx twine@latest check --strict dist/*
   ```

   Use a clean build output directory so old distributions are not mistaken for
   the new release. Build dependencies and Twine also need registry access.
   The `@latest` selector requests the latest available Twine instead of reusing
   an older cached tool, keeping validation compatible with newer build metadata.
7. Commit the version, lockfile, generated notes, and consumed-fragment deletions
   in the release PR. Mark the PR as a release in the changelog exemption.
8. Merge through the normal checks/reviews and verify the resulting package
   artifacts. A normal `main` push builds packages but never publishes them.
   Create the release through one of the approved routes below. A tag push
   alone does not publish a package.

### Create a GitHub Release

After the release PR has merged, open **Actions → Package → Run workflow**:

1. Select the `main` branch.
2. Set `create_release` to true. Leaving it false runs CI/build only.
3. Run the workflow and review the CI/build results.
4. Approve the `github-release` environment deployment when prompted.

The release-creation job creates `v<project.version>` at the exact tested commit,
generates GitHub release notes, and attaches the validated distributions. It does
not bump versions, generate Towncrier notes, or open a release PR; those changes
must already be committed. Existing conflicting tags/releases are not overwritten.
Only stable versions in `X.Y.Z` form are supported by this workflow.

Alternatively, publish a non-draft, non-prerelease GitHub Release through the
GitHub UI or API. Its tag must be `v<project.version>`, and its commit must already
belong to `main`. Publishing that release triggers source validation, CI, and a
package build for the tagged commit before considering registry publication.

### Registry publication and approvals

Registry publication still requires all of the following:

- A published stable GitHub Release, or a successful explicitly requested
  release-creation job in the same workflow run.
- Successful source validation, CI, and package build.
- Repository variable `ENABLE_PACKAGE_PUBLISH` set to the literal string `true`.
- Approval for the `package-publish` environment, when its protection rules require it.
- An explicit `PACKAGE_PUBLISH_URL` upload endpoint and either a
  `PACKAGE_PUBLISH_TOKEN` secret or the `PACKAGE_PUBLISH_USERNAME` /
  `PACKAGE_PUBLISH_PASSWORD` secret pair.

**Before enabling publication**, administrators must configure required reviewers
for the `github-release` and `package-publish` environments under **Settings →
Environments**. Referencing an environment in YAML does not itself enforce human
approval; without protection rules a job can proceed without a review. Configure
branch/tag restrictions and disable self-approval where appropriate. Custom
runners must also provide Git, Bash, and the GitHub CLI (`gh`) for release creation.

Leave `ENABLE_PACKAGE_PUBLISH` unset or false to create GitHub Releases and build
artifacts without uploading to a package registry. Ordinary `main` pushes and
build-only manual runs cannot publish, even when that variable is true.

The workflow's default token permission remains `contents: read`. Only the
release-creation job receives `contents: write` for tags/releases; build, test,
and registry-publishing jobs do not. No job creates release PRs, so none receives
`pull-requests: write`. If release-PR automation is added later, grant that
permission only to its dedicated job, along with the contents access it needs.

Releases created using the workflow's `GITHUB_TOKEN` do not start another release
event workflow. The approved creation route therefore continues to the gated
publishing job in the same run. Registry uploads reuse that run's validated
distributions and deliberately fail on duplicate versions.
