"use strict";
/**
 * Desktop application menu builder and pure formatters for Diagnostics /
 * Resource Usage / Update version checks. Kept free of Electron imports so
 * menu layout and report formatting are unit-tested under plain `node --test`.
 */

const URLS = {
  docs: "https://shibing624.github.io/agentica",
  issues: "https://github.com/shibing624/agentica/issues",
  releases: "https://github.com/shibing624/agentica/releases",
  latestReleaseApi: "https://api.github.com/repos/shibing624/agentica/releases/latest",
};

const LABELS = {
  zh: {
    appName: "Agentica",
    about: "关于 Agentica",
    checkUpdates: "检查更新…",
    cliCommand: "管理 agentica 命令…",
    devTools: "开发者工具",
    hide: "隐藏 Agentica",
    hideOthers: "隐藏其他",
    showAll: "显示全部",
    quit: "退出 Agentica",
    file: "文件",
    newChat: "新建对话",
    closeWindow: "关闭窗口",
    edit: "编辑",
    view: "视图",
    reload: "重新加载",
    forceReload: "强制重新加载",
    resetZoom: "实际大小",
    zoomIn: "放大",
    zoomOut: "缩小",
    fullscreen: "切换全屏",
    window: "窗口",
    help: "帮助",
    docs: "使用文档",
    diagnostics: "诊断工具",
    resources: "资源占用",
    logsFolder: "打开日志文件夹",
    feedback: "意见反馈",
    testNotification: "发送测试通知",
  },
  en: {
    appName: "Agentica",
    about: "About Agentica",
    checkUpdates: "Check for Updates…",
    cliCommand: "Manage 'agentica' Command…",
    devTools: "Developer Tools",
    hide: "Hide Agentica",
    hideOthers: "Hide Others",
    showAll: "Show All",
    quit: "Quit Agentica",
    file: "File",
    newChat: "New Chat",
    closeWindow: "Close Window",
    edit: "Edit",
    view: "View",
    reload: "Reload",
    forceReload: "Force Reload",
    resetZoom: "Actual Size",
    zoomIn: "Zoom In",
    zoomOut: "Zoom Out",
    fullscreen: "Toggle Full Screen",
    window: "Window",
    help: "Help",
    docs: "Documentation",
    diagnostics: "Diagnostics",
    resources: "Resource Usage",
    logsFolder: "Open Logs Folder",
    feedback: "Report Issue",
    testNotification: "Send Test Notification",
  },
};

function isZh(lang) {
  return /^zh/i.test(String(lang || ""));
}

function menuLabels(lang) {
  return isZh(lang) ? LABELS.zh : LABELS.en;
}

/**
 * Build the Menu.buildFromTemplate array.
 * App menu follows DSH's layout; Help menu provides Docs / Diagnostics /
 * Resource Usage / Open Logs Folder / Report Issue + test notification.
 */
function buildAppMenuTemplate({
  platform = process.platform,
  lang = "zh",
  handlers = {},
} = {}) {
  const L = menuLabels(lang);
  const isMac = platform === "darwin";
  const call = (name) => () => {
    if (typeof handlers[name] === "function") handlers[name]();
  };

  const appMenu = isMac
    ? [{
        label: L.appName,
        submenu: [
          { label: L.about, role: "about" },
          { type: "separator" },
          { label: L.checkUpdates, click: call("onCheckUpdates") },
          { label: L.cliCommand, click: call("onManageCli") },
          { type: "separator" },
          { label: L.devTools, role: "toggleDevTools" },
          { type: "separator" },
          { label: L.hide, role: "hide" },
          { label: L.hideOthers, role: "hideOthers" },
          { label: L.showAll, role: "unhide" },
          { type: "separator" },
          { label: L.quit, role: "quit" },
        ],
      }]
    : [];

  const fileSubmenu = [
    { label: L.newChat, accelerator: "CmdOrCtrl+N", click: call("onNewChat") },
    { type: "separator" },
    ...(isMac ? [] : [
      { label: L.checkUpdates, click: call("onCheckUpdates") },
      { label: L.cliCommand, click: call("onManageCli") },
      { type: "separator" },
    ]),
    { label: L.closeWindow, role: "close" },
  ];

  const helpSubmenu = [
    { label: L.docs, click: call("onOpenDocs") },
    { label: L.diagnostics, click: call("onDiagnostics") },
    { label: L.resources, click: call("onResourceUsage") },
    { label: L.logsFolder, click: call("onOpenLogs") },
    { label: L.feedback, click: call("onFeedback") },
    { type: "separator" },
    { label: L.testNotification, click: call("onTestNotification") },
    ...(isMac ? [] : [
      { type: "separator" },
      { label: L.about, click: call("onAbout") },
    ]),
  ];

  return [
    ...appMenu,
    { label: L.file, submenu: fileSubmenu },
    { label: L.edit, role: "editMenu" },
    {
      label: L.view,
      submenu: [
        { label: L.reload, role: "reload" },
        { label: L.forceReload, role: "forceReload" },
        ...(isMac ? [] : [{ label: L.devTools, role: "toggleDevTools" }]),
        { type: "separator" },
        { label: L.resetZoom, role: "resetZoom" },
        { label: L.zoomIn, role: "zoomIn" },
        { label: L.zoomOut, role: "zoomOut" },
        { type: "separator" },
        { label: L.fullscreen, role: "togglefullscreen" },
      ],
    },
    { label: L.window, role: "windowMenu" },
    { label: L.help, role: "help", submenu: helpSubmenu },
  ];
}

/** Parse X.Y.Z semver core; returns -1 / 0 / 1 (a < b, a == b, a > b), or null if unparseable. */
function compareSemver(a, b) {
  const parse = (v) => {
    const m = /^v?(\d+)\.(\d+)\.(\d+)/i.exec(String(v || "").trim());
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
  };
  const pa = parse(a);
  const pb = parse(b);
  if (!pa || !pb) return null;
  for (let i = 0; i < 3; i += 1) {
    if (pa[i] !== pb[i]) return pa[i] > pb[i] ? 1 : -1;
  }
  return 0;
}

function formatDiagnosticsReport(info, lang = "zh") {
  const zh = isZh(lang);
  const yn = (val) => (val ? (zh ? "是" : "Yes") : (zh ? "否" : "No"));
  const lines = [
    `${zh ? "应用版本" : "App Version"}: ${info.appVersion || "-"} (${info.packaged ? (zh ? "打包版" : "Packaged") : (zh ? "开发模式" : "Dev")})`,
    `${zh ? "运行环境" : "Runtime"}: Electron ${info.electron || "-"} / Node ${info.node || "-"} / Chrome ${info.chrome || "-"}`,
    `${zh ? "系统平台" : "Platform"}: ${info.platform || "-"} (${info.arch || "-"}) ${info.osRelease || ""}`.trim(),
    `${zh ? "签名校验" : "Code Signature"}: ${yn(info.signed)}`,
    `${zh ? "Gateway 状态" : "Gateway"}: ${info.gatewayRunning ? (zh ? "运行中" : "Running") : (zh ? "未运行" : "Stopped")}`
      + (info.gatewayPort ? ` | port=${info.gatewayPort}` : "")
      + (info.gatewayPid ? ` | pid=${info.gatewayPid}` : "")
      + (info.gatewaySpawned !== undefined ? ` | ${info.gatewaySpawned ? (zh ? "桌面端拉起" : "spawned") : (zh ? "复用外部" : "attached")}` : ""),
    `${zh ? "健康检查" : "Health Check"}: ${info.healthOk ? `OK (${info.healthMs}ms)` : (info.healthError || "Unavailable")}`,
    `${zh ? "命令来源" : "Gateway Source"}: ${info.commandSource || "-"}`,
    `${zh ? "CLI 命令" : "CLI Command"}: ${info.cliTarget || "-"} (${info.cliInstalled ? (zh ? "已安装" : "installed") : (zh ? "未安装" : "not installed")})`,
    `${zh ? "数据目录" : "Agentica Home"}: ${info.agenticaHome || "-"}`,
    `${zh ? "日志目录" : "Logs Directory"}: ${info.logsDir || "-"}`,
    `${zh ? "桌面配置" : "User Data"}: ${info.userDataDir || "-"}`,
  ];
  return lines.join("\n");
}

function formatMb(kb) {
  const mb = Number(kb || 0) / 1024;
  return `${mb >= 100 ? mb.toFixed(0) : mb.toFixed(1)} MB`;
}

function formatResourceUsageReport(appMetrics = [], extraProcesses = [], lang = "zh") {
  const zh = isZh(lang);
  const rows = [];
  let totalKb = 0;
  let totalCpu = 0;

  for (const m of appMetrics) {
    if (!m || typeof m !== "object") continue;
    const kb = Number(m.memory && m.memory.workingSetSize ? m.memory.workingSetSize : 0);
    const cpu = Number(m.cpu && typeof m.cpu.percentCPUUsage === "number" ? m.cpu.percentCPUUsage : 0);
    totalKb += kb;
    totalCpu += cpu;
    const label = m.name ? `${m.type} (${m.name})` : String(m.type || "Process");
    rows.push(`PID ${m.pid}  ${label} — ${formatMb(kb)} | CPU ${cpu.toFixed(1)}%`);
  }

  for (const p of extraProcesses) {
    if (!p || typeof p !== "object") continue;
    const kb = Number(p.rssKb || 0);
    const cpu = Number(p.cpuPercent || 0);
    totalKb += kb;
    totalCpu += cpu;
    rows.push(`PID ${p.pid}  ${p.label || "Gateway"} — ${formatMb(kb)} | CPU ${cpu.toFixed(1)}%`);
  }

  const header = zh
    ? `总内存占用: ${formatMb(totalKb)}  |  总 CPU: ${totalCpu.toFixed(1)}%  |  进程数: ${rows.length}`
    : `Total Memory: ${formatMb(totalKb)}  |  Total CPU: ${totalCpu.toFixed(1)}%  |  Processes: ${rows.length}`;
  return rows.length ? `${header}\n\n${rows.join("\n")}` : header;
}

module.exports = {
  URLS,
  menuLabels,
  buildAppMenuTemplate,
  compareSemver,
  formatDiagnosticsReport,
  formatResourceUsageReport,
};
