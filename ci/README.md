# CI notes

The workflows live in `.github/workflows/`. This file records what they are for and the one setup
detail that is easy to trip over.

Pushing anything into `.github/workflows/` requires a GitHub token carrying the **`workflow`**
scope. A token without it is rejected with *"refusing to allow a Personal Access Token to create or
update workflow ... without `workflow` scope"* -- while every other push in the repo succeeds, which
makes it look like a repository permission problem rather than a token one.

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
