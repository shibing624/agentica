**user** — Anything *about the user themselves*: role, business profile, personal account state, owned assets, preferences, working style, constraints, recurring habits, and ongoing personal setup.
  Use this whenever the subject of the fact IS the user (their shop, their account, their product line, their pricing, their materials, their past actions on their own resources).
  Example: 'User runs a WeChat tea-utensil shop, sells single gaiwans and teapots across all price tiers, does not livestream.'

**project** — A concrete piece of *shared/collaborative work* with multiple actors, deliverables, or deadlines that is NOT derivable from code or git history.
  Use this only when the fact is about a named initiative, incident, or engagement involving more than just the user's own steady-state setup. Convert relative dates to absolute dates.
  Also: an experiment or investigation result that took real effort to establish and is not recorded in the repo — a dataset that lacks a needed label, an approach measured and rejected, with the numbers. Saving it keeps the next session from re-running it.
  Counter-example: a user describing their own shop's products or their own ad-account history is **user**, not project.

**reference** — Pointers to external resources, rules, or systems the user/team relies on: platform policies, issue trackers, dashboards, wikis, documentation sites, internal tools, SLAs, rebate/refund rules.
  Also the environment outside the repo: which machine/GPU runs which long-lived service on which port, shared model/data directories, env locations, restart scripts, processes that must not be touched, and measured limits of external APIs (rate limits, unsupported parameters).
  When to save: when you learn about an external system, environment fact, or rule that future sessions will need to look up — without waiting to be asked. A session notes file does not carry over to other sessions.
