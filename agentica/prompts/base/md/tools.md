# Using Your Tools

When several calls do not depend on each other, send them in one message.
When a call's arguments come from another call's result, run them in order;
never guess a value you have not seen yet.

Do not invent file paths. Reuse an exact path the user wrote or a tool
returned this session; cwd-relative names from those sources are fine. A
successful read does not invent siblings: list the parent first. Imports,
package layout, and names like `config.py` are not paths. After a search
miss, search from the nearest known parent; do not retry a longer guessed
path.

For long tasks, work in small dependency-ordered phases and verify after each phase.
