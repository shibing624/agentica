[Continuing toward your standing goal]
Goal: {objective}
{subgoals_block}
Continue working toward this goal. Take the next concrete step. Do not stop merely because you made partial progress.

When you believe the goal is complete, DO NOT just say so — call the `verify_completion` tool to prove it:
- For code/tasks with a runnable check: `verify_completion(mode="test", verify_command="<cmd that exits 0 only when done, e.g. pytest ...>", final_answer="<the result>")`. Write the tests first.
- For non-code deliverables: `verify_completion(mode="criteria", acceptance_criteria="<checklist>", summary="<what you produced>", final_answer="<the result>")`.
Only a passing verification ends the loop. If it fails, fix the reported gap and verify again.

The user is likely away. Do not pause for ambiguity: pick the most reasonable option, keep going, and record the assumption in the final answer. Call `update_goal(status="paused", reason="...")` only when every remaining step needs something only the user can provide (credentials, access, approval for an irreversible step).
