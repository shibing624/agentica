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
import { getTheme, applyTheme, setTheme } from './theme.js';
import { setupDragDrop, onFilePick, removeFile } from './files.js';
import { fmtN, fmtTime, fmtFileSize, shortenPath, toggleSidebar, focusSidebarSearch } from './utils.js';
import {
  confirmOk, confirmCancel,
  openDirModal, closeDirModal, toggleDirHistory, selectDirHistory,
  saveDir, copyDir, copyCurrentDir, openInFinder, openInTerminal,
  openAccountPanel, closeAccountPanel, currentUsage, archivedSessions, dirHistoryFiltered,
  deleteArchivedSession, openDirModalForNewSession,
} from './modals.js';
import {
  loadSessions, newSession, switchTo, archiveSession, unarchiveSession,
  forkSession, renameSession, commitSessionRename, renameKey, createSessionInProject,
  toggleChatMenu, closeChatMenu, renameCurrentSession, forkCurrentSession, archiveCurrentSession,
  exportCurrentSessionMarkdown,
} from './sessions.js';
import {
  sidebarTree, toggleProjectCollapsed,
  renameProject, commitProjectRename, projectRenameKey, removeProject,
} from './sidebar.js';
import {
  renderChat, copyMsg, editUserMsg, setMsgFeedback, forkFromMsg, retryMsg,
  toggleSec, toggleToolResult, toggleThinkBody, toggleToolGroup,
  scrollEnd, updateScrollBtn, isNearBottom,
  handleKey, autoResize, onAction, onInput,
  removeQueueItem, editQueueItem, sendQueueItemNow, sessionQueue,
} from './chat.js';
import { jumpToChatMsg, updateChatNavActive, renderChatNav } from './chat-nav.js';
import {
  loadStatus, loadModels, loadProfiles, loadProviders,
  toggleModelDD, switchProfile, toggleCtxTip, tokenUsage,
  toggleQuickMenu, toggleApprovalMenu, setApprovalMode, triggerFilePicker,
  closeInputMenus, quickSkills,
  insertSkillRef, insertGoalPrefix,
  slashItems, selectSlash, updateSlash,
} from './model-panel.js';
import {
  cancelCronForm, showCronForm, editCronJob,
  saveCronForm, deleteCronJob, pauseCronJob, resumeCronJob, triggerCronJob, toggleCronRuns, fmtCronTime,
  polishCronPrompt,
} from './cron-panel.js';
import {
  openSettingsModal, closeSettingsModal, cancelProfileForm, showProfileForm,
  addEnvRow, removeEnvRow, saveProfileForm, deleteProfile,
  switchSettingsTab, copyConfigPath, openConfigFile,
} from './settings-panel.js';
import {
  openPluginsPanel, closePluginsPanel, filteredTools, filteredSkills,
  showSkillForm, cancelSkillForm, saveSkillForm, editSkill, deleteSkill,
  showMcpForm, cancelMcpForm, saveMcpForm, deleteMcpServer, addMcpEnvRow, removeMcpEnvRow,
} from './plugins-panel.js';

// ---- petite-vue app: shared `state` + getters/methods for every v-scope
// block in index.html (input box, modals, sidebar) ----
createApp({
  state,
  get modelLabel() {
    if (state.switchingLabel) return state.switchingLabel;
    return state.serverModelName || state.serverModel;
  },
  get usage() { return tokenUsage() },
  get accountUsage() { return currentUsage() },
  get archivedList() { return archivedSessions() },
  get dirHistoryFiltered() { return dirHistoryFiltered() },
  get sidebarTree() { return sidebarTree() },
  get filteredTools() { return filteredTools() },
  get filteredSkills() { return filteredSkills() },
  get quickSkills() { return quickSkills() },
  get slashItems() { return slashItems() },
  get sessionQueue() { return sessionQueue() },

  fmtN, fmtTime, fmtFileSize, fmtCronTime, shortenPath,

  toggleSidebar,
  onFilePick, removeFile,
  handleKey, autoResize, onAction,
  removeQueueItem, editQueueItem, sendQueueItemNow,
  toggleCtxTip, toggleModelDD, switchProfile,
  toggleQuickMenu, toggleApprovalMenu, setApprovalMode, triggerFilePicker,
  insertSkillRef, insertGoalPrefix,
  selectSlash, updateSlash, onInput,

  confirmOk, confirmCancel,
  closeDirModal, saveDir, copyDir, copyCurrentDir, openInFinder, openInTerminal, toggleDirHistory, selectDirHistory,
  toggleChatMenu, renameCurrentSession, forkCurrentSession, archiveCurrentSession, exportCurrentSessionMarkdown,
  closeAccountPanel, openSettingsModal, switchTo, unarchiveSession,
  closePluginsPanel, showSkillForm, cancelSkillForm, saveSkillForm, editSkill, deleteSkill,
  showMcpForm, cancelMcpForm, saveMcpForm, deleteMcpServer, addMcpEnvRow, removeMcpEnvRow,
  deleteArchivedSession,

  showCronForm, editCronJob, pauseCronJob, resumeCronJob, triggerCronJob,
  deleteCronJob, toggleCronRuns, cancelCronForm, saveCronForm, polishCronPrompt,

  closeSettingsModal, showProfileForm, deleteProfile, cancelProfileForm, saveProfileForm,
  addEnvRow, removeEnvRow, switchSettingsTab, copyConfigPath, openConfigFile, setTheme,

  toggleProjectCollapsed, renameProject, commitProjectRename, projectRenameKey, removeProject,
  renameSession, commitSessionRename, renameKey, forkSession, archiveSession, createSessionInProject,
}).mount();

// ---- expose functions still referenced from inline onclick/onkeydown
// attributes outside the petite-vue-controlled regions (topbar, sidebar nav,
// welcome screen, and chat.js/tools-render.js's dynamically built message
// HTML) — template strings built at runtime can't "see" module-scoped
// bindings, only globals ----
Object.assign(window, {
  newSession, openSettingsModal, openPluginsPanel, openAccountPanel, toggleSidebar, focusSidebarSearch, openDirModal, openDirModalForNewSession, scrollEnd,
  copyMsg, editUserMsg, setMsgFeedback, forkFromMsg, retryMsg,
  toggleSec, toggleToolResult, toggleThinkBody, toggleToolGroup,
  jumpToChatMsg,
});

// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(getTheme());
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
    if (!document.getElementById('chatMenuWrap')?.contains(e.target)) {
      closeChatMenu();
    }
    if (!document.getElementById('inputBox').contains(e.target)) {
      closeInputMenus();
    }
  });
  // scroll button visibility + track user scroll intent
  const chatArea = document.getElementById('chatArea');
  chatArea.addEventListener('scroll', () => {
    updateScrollBtn();
    updateChatNavActive();
    // unlock once the user scrolls back near the bottom
    if (isNearBottom()) { state.userScrolledUp = false; state._scrollLock = false; }
    else if (state.streaming && !state._scrollLock) {
      // during streaming, anything not at the bottom counts as scrolled up
      state.userScrolledUp = true;
    }
  });
  // Detect user-initiated scroll (wheel / touch) — any upward scroll immediately triggers
  chatArea.addEventListener('wheel', (e) => {
    if (!state.streaming) return;
    if (e.deltaY < 0 && isNearBottom()) {
      // scrolled up near the bottom → lock immediately
      state._scrollLock = true;
      state.userScrolledUp = true;
      updateScrollBtn();
    } else if (e.deltaY > 0 && !isNearBottom()) {
      // scrolling down away from the bottom → don't unlock (user still browsing mid-content)
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
  // Thinking/tool-call card ("已处理") has its own inner scroll while streaming.
  // Any upward scroll inside it should stop the auto-follow-to-bottom immediately —
  // no threshold, so the user can read older content without fighting new chunks.
  chatArea.addEventListener('wheel', (e) => {
    const body = e.target.closest && e.target.closest('.think-body');
    if (!body) return;
    if (e.deltaY < 0) {
      state._thinkScrollLock = true;
    } else if (e.deltaY > 0 && (body.scrollHeight - body.scrollTop - body.clientHeight) < 24) {
      state._thinkScrollLock = false;
    }
  }, { passive: true });
  chatArea.addEventListener('touchmove', (e) => {
    const body = e.target.closest && e.target.closest('.think-body');
    if (body) state._thinkScrollLock = true;
  }, { passive: true });
  // window resize reflows message heights, which the nav's tick positions
  // (computed from offsetTop) depend on — debounce so a drag-resize doesn't
  // thrash the DOM every frame.
  let navResizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(navResizeTimer);
    navResizeTimer = setTimeout(renderChatNav, 150);
  });
});
