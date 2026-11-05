/** Composer commands, shared by the `+` menu and the `/` menu so the two never
 *  drift (same ids, labels, order). Pure; ChatPage runs them. */

export type CommandId = "file" | "goal" | "queue" | "compact" | "permission" | "model" | "export";

export type CommandStrings = {
  cmdGroupAdd: string;
  cmdGroupCommands: string;
  cmdGroupSkills: string;
  cmdFile: string;
  cmdGoal: string;
  cmdQueue: string;
  cmdCompact: string;
  cmdPermission: string;
  cmdModel: string;
  cmdExport: string;
  slashGoal: string;
  slashQueue: string;
  slashCompact: string;
  cmdPermissionDesc: string;
  cmdModelDesc: string;
  cmdExportDesc: string;
};

export type MenuItem = {
  key: string;
  /** Present for built-ins; skills carry their slash command in `cmd`. */
  id?: CommandId;
  cmd: string;
  label: string;
  desc: string;
  kind: "command" | "skill";
};

export type MenuGroup = { title: string; items: MenuItem[] };

function command(id: CommandId, S: CommandStrings): MenuItem {
  const label: Record<CommandId, string> = {
    file: S.cmdFile, goal: S.cmdGoal, queue: S.cmdQueue, compact: S.cmdCompact,
    permission: S.cmdPermission, model: S.cmdModel, export: S.cmdExport,
  };
  const desc: Record<CommandId, string> = {
    file: "", goal: S.slashGoal, queue: S.slashQueue, compact: S.slashCompact,
    permission: S.cmdPermissionDesc, model: S.cmdModelDesc, export: S.cmdExportDesc,
  };
  return { key: "cmd:" + id, id, cmd: "/" + id, label: label[id], desc: desc[id], kind: "command" };
}

/** The `+` button: things to add to the message, then session commands. */
export function plusMenuGroups(S: CommandStrings): MenuGroup[] {
  return [
    { title: S.cmdGroupAdd, items: (["file", "goal"] as const).map((id) => command(id, S)) },
    { title: S.cmdGroupCommands, items: (["compact", "permission", "model", "export"] as const).map((id) => command(id, S)) },
  ];
}

/** Skills the user can invoke by slash command, as menu items. */
export function skillItems(skills: Array<Record<string, any>>): MenuItem[] {
  const out: MenuItem[] = [];
  for (const sk of skills || []) {
    if (sk.is_hidden || sk.user_invocable === false) continue;
    const cmd = String(sk.slash || sk.trigger || "").toLowerCase();
    if (!cmd.startsWith("/")) continue;
    out.push({ key: "skill:" + cmd, cmd, label: String(sk.name || cmd), desc: String(sk.description || ""), kind: "skill" });
  }
  return out;
}

/**
 * The `/` menu for what has been typed so far (``query`` starts with "/").
 * A command matches on its slash name or its label, a skill on its slash
 * command or its name; both by prefix. Empty groups are dropped.
 */
export function slashMenuGroups(query: string, skills: Array<Record<string, any>>, S: CommandStrings): MenuGroup[] {
  const q = query.replace(/^\//, "").toLowerCase();
  const hit = (it: MenuItem) => !q
    || it.cmd.slice(1).toLowerCase().startsWith(q)
    || it.label.toLowerCase().startsWith(q);
  const commands = (["goal", "queue", "compact", "permission", "model", "export"] as const)
    .map((id) => command(id, S)).filter(hit);
  const skill = skillItems(skills).filter(hit);
  return [
    { title: S.cmdGroupCommands, items: commands },
    { title: S.cmdGroupSkills, items: skill },
  ].filter((g) => g.items.length);
}

/** Groups flattened in display order: the keyboard cursor walks this. */
export function flatItems(groups: MenuGroup[]): MenuItem[] {
  return groups.flatMap((g) => g.items);
}
