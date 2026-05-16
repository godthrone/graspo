# Algorithm

GRASPO stands for Group Relative Adaptive Structured Policy Optimization. It is
best understood as a GRPO-style algorithm specialized for industrial Agent
structured-output tasks: it keeps group-relative advantages and the no-critic
resource profile, while emphasizing structured rewards, adaptive sampling, and
hard-sample reuse.

GRASPO is not a universal replacement for GRPO. It is a better fit when outputs
are verifiable, such as JSON extraction, structured classification, tool-call
argument generation, ticket parsing, and log parsing. For long-CoT math reasoning
or broad preference alignment, DAPO, GSPO, PPO, or other rollout methods may be
more appropriate.

## Core ideas

- Generate a group of completions for each prompt.
- Score each completion with a structured reward.
- Compute group-relative advantages.
- Skip prompts that are already solved on the first group.
- Retry difficult prompts to find useful reward variance.
- Drop invalid groups whose rewards are identical.
- Optimize with a clipped policy objective and no critic model.

## Motivation

Industrial Agents often need more than human-like answers. They need stable,
executable, parseable, and auditable structured outputs. Typical examples:

- strict JSON output
- tool-call argument generation
- field extraction from unstructured text
- ticket, log, and form parsing

These tasks usually have ground truth or field-level scoring. Format errors and
unnecessary explanations directly affect system reliability. GRASPO therefore
combines format, content, and conciseness rewards instead of only checking a
final answer.

## Reward

The default reward combines:

- marker rewards for required output structure
- JSON validity reward
- dictionary comparison reward
- exact match bonus
- anti-useless-text penalty

This makes GRASPO especially suitable for JSON extraction, classification,
tool-call learning, and other tasks with programmatically verifiable answers.

## Differences from GRPO

- GRPO is a more general group-relative policy optimization method, commonly
  used for math, code, and other verifiable reasoning tasks.
- GRASPO adds structure-aware rewards for JSON, markers, field matching, and
  anti-useless-text penalties.
- GRASPO can skip prompts solved on the first group, reducing updates on already
  mastered examples.
- GRASPO adaptively retries hard prompts to find groups with useful reward
  variance.
- GRASPO drops invalid groups whose rewards are all identical, reducing
  zero-advantage updates.

These changes make GRASPO practical for structured Agent workloads, but also
make it more dependent on clear ground truth and task-specific rules.

## Hard-Sample SFT

In practice, GRASPO can be paired with SFT on hard samples. Rollouts identify
prompts where the model often fails but can produce high-reward answers after
retry; those high-quality trajectories can then be converted into SFT data. This
is especially useful for small datasets because it consolidates trajectories
discovered by RL exploration.

## ARD Anti-Forgetting

Hard-sample-only SFT can make the model overfit to business formats and forget
general QA, coding, reasoning, and explanation skills. ARD mixes in a base-model
anchor bank during SFT and uses low-weight CE replay or optional KL distillation
to keep the model from drifting too far from its general capabilities.

ARD does not change the GRASPO rollout objective. It is a companion workflow for
hard-sample SFT. Generate the anchor bank offline once, then reuse it in every
`GRASPO -> hard sample mining -> ARD-SFT` iteration.
