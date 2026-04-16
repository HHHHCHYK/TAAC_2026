---
name: copilot-review-loop
description: Run an automated GitHub Copilot pull-request review loop. Use when Codex needs to trigger Copilot reviews on a PR, read the resulting comments, fix all issues locally, push, re-trigger, and repeat until the review produces zero new comments. Handles polling, comment parsing, fix-test-push cycles, and conversation lifecycle.
---

# Copilot Review Loop

## Overview

Use this skill to drive an automated fix-review cycle on a GitHub pull request.
The agent triggers a Copilot review, reads new comments, fixes every flagged issue locally, runs tests, commits, pushes, re-triggers, and repeats until the review body says **"generated no new comments"**.

Apply this skill when the user asks to "let Copilot review and fix everything", "iterate until clean", or similar requests that imply a closed-loop automated review cycle.

## Prerequisites

Before starting the loop, confirm:

- The working branch is pushed to origin and a PR exists.
- You know the repository owner, repo name, and PR number.
- You have permission to push commits (ask the user if unclear).
- The relevant test command is known (e.g. `uv run python -m pytest tests/test_foo.py -v --tb=short`).

Load the GitHub MCP tools before the first call:

- `mcp_github_request_copilot_review`
- `mcp_github_pull_request_read` (methods: `get_reviews`, `get_review_comments`)

## Loop procedure

### 1. Trigger a review

```
mcp_github_request_copilot_review(owner, repo, pullNumber)
```

### 2. Poll for the new review object

Call `mcp_github_pull_request_read(method="get_reviews")` and look for a **new review entry** whose `commit_id` matches the latest pushed commit.

**Critical**: Do NOT rely on the total number of reviews or the total comment count (`totalCount`) to detect a new review.
When Copilot finds nothing to comment on, it still creates a review object but the comment count stays unchanged.
Instead, check:

1. Whether a new review `id` has appeared that was not in the previous poll.
2. Whether the `commit_id` of the latest review matches the commit you just pushed.
3. Read the review `body` text — look for **"generated no new comments"** vs **"generated N comments"**.

Poll at reasonable intervals. If the review has not appeared after ~5 minutes, re-trigger with `mcp_github_request_copilot_review`.

### 3. Check for completion

If the latest review body contains **"generated no new comments"**, the loop is done — skip to step 7.

### 4. Read new comments

Call `mcp_github_pull_request_read(method="get_review_comments", perPage=30)`.
Filter for comments whose `created_at` timestamp is close to the new review's `submitted_at`.
Focus on threads where `is_resolved: false` and `is_outdated: false`.

Old threads that are `is_outdated: true` or `is_resolved: true` can be ignored.
Note: thread `node_id` values are not available through `get_review_comments`, so you cannot programmatically resolve old threads.

### 5. Fix, test, commit, push

For each new comment:

1. Read the relevant source file and understand the issue.
2. Implement the fix. Prefer minimal, targeted changes.
3. Update tests if signatures or behavior changed.
4. Run the test suite and confirm all tests pass.
5. Stage, commit with a descriptive message like `review(roundN): brief description`.
6. Push to origin.

### 6. Re-trigger and repeat

Go back to step 1.

### 7. Report completion

Summarize the full iteration history:

- Total rounds and total comments fixed.
- A table showing round number, comment count, and key fixes per round.
- Final test count and status.
- The commit hash of the final clean state.

## Conversation lifecycle

### Avoid context exhaustion

Each poll response consumes context window tokens.
Do not poll more than ~15 times before pausing to report status to the user.
If the context window is filling up, summarize progress and let the user continue in a new conversation turn.

### Session memory

At the start, create a session memory note (`/memories/session/review-loop-state.md`) tracking:

- PR number, branch, repo
- Current round number
- Commits pushed per round
- Outstanding old threads (for reference only)

Update this note after each round completes so the loop can resume if the conversation is interrupted.

### When to stop

Stop the loop when:

- The latest review says "generated no new comments".
- The user explicitly asks to stop.
- You have completed 15+ rounds without convergence (report this as an anomaly).

## Anti-patterns

| Anti-pattern | Correct approach |
|---|---|
| Counting `totalCount` of review comments to detect new review | Check for new review `id` with matching `commit_id` |
| Counting number of review objects to detect new review | A 0-comment review still adds a review object; check `commit_id` + `body` |
| Polling indefinitely in a tight loop | Limit polls, re-trigger after ~5 min, report to user if stalled |
| Fixing issues without running tests | Always run tests before committing |
| Pushing without confirming test pass | Gate push on green tests |
| Ignoring `is_outdated` threads | Only act on `is_outdated: false` + `is_resolved: false` threads |
| Making large refactors to address a single comment | Keep fixes minimal and targeted |

## Reference

### Useful MCP tool signatures

```
mcp_github_request_copilot_review(owner, repo, pullNumber)
mcp_github_pull_request_read(method="get_reviews", owner, repo, pullNumber)
mcp_github_pull_request_read(method="get_review_comments", owner, repo, pullNumber, perPage=30)
mcp_github_pull_request_read(method="get_diff", owner, repo, pullNumber)
```

### Typical review body patterns

- Has comments: `"generated N comments"` (where N > 0)
- Clean: `"generated no new comments"`
- Both appear inside the `body` field of the review object.
