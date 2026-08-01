# CI workflows, staged for a one-click move

These two files are finished and ready. They live here rather than in `.github/workflows/` for one
boring reason: pushing a file into `.github/workflows/` requires a GitHub token carrying the
`workflow` scope, and the token in this machine's keychain does not have it. Everything else in the
repo pushes fine.

## Activating them (browser only, about a minute, nothing to install)

For each of `ci.yml` and `trl-drift.yml`:

1. Open the file on github.com.
2. Click the pencil (Edit).
3. Click the filename box at the top and replace the whole path with
   `.github/workflows/ci.yml` (or `.github/workflows/trl-drift.yml`).
4. Commit.

The browser uses your logged-in session rather than the token, so the `workflow` restriction does
not apply. GitHub treats the rename as a move; the file content is unchanged.

## Or fix the token once, if you would rather

github.com -> Settings -> Developer settings -> Personal access tokens -> your token ->
tick **workflow** -> update. Then clear the cached credential so git asks for the new one:

    printf 'protocol=https\nhost=github.com\n\n' | git credential reject

The next push prompts for username and password; the password is the token. After that,
`git mv ci/workflows/*.yml .github/workflows/` pushes normally and stays working for future edits.

## What they do

`ci.yml`
  - lint: ruff + `mypy --strict` on the shipped package
  - unit: `pip install -e .` with **no extras** across Python 3.10-3.13 and macOS, then asserts
    that importing grpo_doctor and running a Monitor never pulls torch into `sys.modules`. That
    assertion is the point of the job -- the package is meant to be installed into someone's
    existing RL training image.
  - testbed: the research code, CPU torch only
  - guarantees: causality, blindness and constant memory as their own job, so a break there is its
    own red X rather than one line in a long test log

`trl-drift.yml`
  - weekly against `trl@main`, opens an issue on failure. It checks the three things this project
    asserts about TRL's internals: that `nanstd` still applies Bessel's correction (which sets the
    advantage bound quoted in the README), that `GRPOTrainer.log` still clears `_metrics` (the
    reason `logging_steps=1` is mandatory), and that `old_per_token_logps` still exists (why the
    clip signals are NaN rather than 0.0 at `num_iterations=1`).
