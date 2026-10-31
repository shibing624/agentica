
Earlier work lists what the user asked for in earlier sessions and where it ended, newest first, as `<id>: [<project>] [<status>] <date> <ask> → <outcome>`. `this project` is the directory the user is working in now; any other project is named by its path.

Pick earlier work only when this message continues it, depends on its outcome, or repeats a problem it already worked through — in this project or another one. A shared generic word ("commit", "push", "fix", "test") is not a reason: a pending push in another project has nothing to do with "push" here. Picking nothing is the normal outcome.

Return at most {work_limit} earlier-work ids.
