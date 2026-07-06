// ============ ENTRY POINT ============
// Physically split from the former monolithic app.js into native ES modules
// (see /static/js/*.js). petite-vue (vendored under ./vendor/petite-vue.js)
// owns the templating/binding for the input box, all modals, and the sidebar
// project/session list — see the `v-scope` blocks in index.html. Everything
// else (message rendering in chat.js, the topbar, sidebar nav buttons) still
// uses plain inline `onclick=` attributes wired through `window`, since those
// areas weren't part of this pass.
import { createApp } from './vendor/petite-vue.js';
import { state } from './state.js';
import { getTheme, applyTheme, toggleTheme } from './theme.js';
import { setupDragDrop, onFilePick, removeFile } from './files.js';
import { fmtN, fmtTime, fmtFileSize, shortenPath, toggleSidebar } from './utils.js';
import {
  confirmOk, confirmCancel,
  openDirModal, closeDirModal, toggleDirHistory, selectDirHistory,
  saveDir, copyDir, openInFinder, openInTerminal,
  openAccountPanel, closeAccountPanel, currentUsage, archivedSessions, dirHistoryFiltered,
  openPluginsPanel, closePluginsPanel, saveGatewayToken,
} from './modals.js';
import {
  loadSessions, newSession, switchTo, archiveSession, unarchiveSession,
  forkSession, renameSession, commitSessionRename, renameKey,
} from './sessions.js';
import {
  sidebarTree, toggleProjectCollapsed,
  renameProject, commitProjectRename, projectRenameKey, removeProject,
} from './sidebar.js';
import {
  renderChat, copyMsg, editUserMsg, setMsgFeedback, forkFromMsg, retryMsg,
  toggleSec, toggleToolResult, toggleThinkBody, toggleToolGroup,
  scrollEnd, updateScrollBtn, isNearBottom,
  handleKey, autoResize, onAction,
} from './chat.js';
import {
  loadHealth, loadStatus, loadModels, loadProfiles, loadProviders,
  toggleModelDD, switchProfile, toggleCtxTip, tokenUsage,
  toggleQuickMenu, toggleApprovalMenu, setApprovalMode, triggerFilePicker,
  triggerFolderPicker, closeInputMenus,
} from './model-panel.js';
import {
  openCronModal, closeCronModal, cancelCronForm, showCronForm, editCronJob,
  saveCronForm, deleteCronJob, pauseCronJob, resumeCronJob, triggerCronJob, toggleCronRuns, fmtCronTime,
} from './cron-panel.js';
import {
  openSettingsModal, closeSettingsModal, cancelProfileForm, showProfileForm,
  addEnvRow, removeEnvRow, saveProfileForm, deleteProfile,
} from './settings-panel.js';

// ---- petite-vue app: shared `state` + getters/methods for every v-scope
// block in index.html (input box, modals, sidebar) ----
createApp({
  state,
  get modelLabel() {
    if (state.switchingLabel) return state.switchingLabel;
    return state.serverProfile ? state.serverProfile + ' · ' + state.serverModelName : (state.serverModelName || state.serverModel);
  },
  get usage() { return tokenUsage() },
  get accountUsage() { return currentUsage() },
  get archivedList() { return archivedSessions() },
  get dirHistoryFiltered() { return dirHistoryFiltered() },
  get sidebarTree() { return sidebarTree() },

  fmtN, fmtTime, fmtFileSize, fmtCronTime, shortenPath,

  onFilePick, removeFile,
  handleKey, autoResize, onAction,
  toggleCtxTip, toggleModelDD, switchProfile,
  toggleQuickMenu, toggleApprovalMenu, setApprovalMode, triggerFilePicker,
  triggerFolderPicker,

  confirmOk, confirmCancel,
  closeDirModal, saveDir, copyDir, openInFinder, openInTerminal, toggleDirHistory, selectDirHistory,
  closeAccountPanel, openSettingsModal, openCronModal, toggleTheme, switchTo, unarchiveSession,
  closePluginsPanel, saveGatewayToken,

  closeCronModal, showCronForm, editCronJob, pauseCronJob, resumeCronJob, triggerCronJob,
  deleteCronJob, toggleCronRuns, cancelCronForm, saveCronForm,

  closeSettingsModal, showProfileForm, deleteProfile, cancelProfileForm, saveProfileForm,
  addEnvRow, removeEnvRow,

  toggleProjectCollapsed, renameProject, commitProjectRename, projectRenameKey, removeProject,
  renameSession, commitSessionRename, renameKey, forkSession, archiveSession,
}).mount();

// ---- expose functions still referenced from inline onclick/onkeydown
// attributes outside the petite-vue-controlled regions (topbar, sidebar nav,
// welcome screen, and chat.js/tools-render.js's dynamically built message
// HTML) — template strings built at runtime can't "see" module-scoped
// bindings, only globals ----
Object.assign(window, {
  newSession, openCronModal, openPluginsPanel, openAccountPanel, toggleSidebar, openDirModal, scrollEnd,
  copyMsg, editUserMsg, setMsgFeedback, forkFromMsg, retryMsg,
  toggleSec, toggleToolResult, toggleThinkBody, toggleToolGroup,
});

// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(getTheme());
  loadHealth();
  loadSessions();
  loadStatus();
  loadProfiles();
  loadProviders();
  loadModels();
  setupDragDrop();
  document.getElementById('inputTa').focus();
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change', () => { if (getTheme() === 'auto') applyTheme('auto') });
  renderChat();
  document.addEventListener('click', e => {
    if (!document.getElementById('modelWrap').contains(e.target)) {
      state.modelDDOpen = false;
    }
    if (!document.getElementById('ctxWrap').contains(e.target)) {
      state.ctxTipOpen = false;
    }
    if (!document.getElementById('inputBox').contains(e.target)) {
      closeInputMenus();
    }
  });
  // scroll button visibility + track user scroll intent
  const chatArea = document.getElementById('chatArea');
  chatArea.addEventListener('scroll', () => {
    updateScrollBtn();
    // 用户滚动到接近底部时，解除锁定
    if (isNearBottom()) { state.userScrolledUp = false; state._scrollLock = false; }
    else if (state.streaming && !state._scrollLock) {
      // 流式期间，只要不是在底部就算上翻
      state.userScrolledUp = true;
    }
  });
  // Detect user-initiated scroll (wheel / touch) — any upward scroll immediately triggers
  chatArea.addEventListener('wheel', (e) => {
    if (!state.streaming) return;
    if (e.deltaY < 0 && isNearBottom()) {
      // 在底部附近上翻 → 立即锁定
      state._scrollLock = true;
      state.userScrolledUp = true;
      updateScrollBtn();
    } else if (e.deltaY > 0 && !isNearBottom()) {
      // 非底部区域下滚 → 不解锁（用户还在浏览中间内容）
      updateScrollBtn();
    } else if (e.deltaY > 0 && isNearBottom()) {
      state.userScrolledUp = false; state._scrollLock = false; updateScrollBtn();
    }
  }, { passive: true });
  chatArea.addEventListener('touchmove', () => {
    if (state.streaming && !isNearBottom()) {
      if (!state._scrollLock) { state.userScrolledUp = true; state._scrollLock = true; }
    }
    updateScrollBtn();
  }, { passive: true });
});
