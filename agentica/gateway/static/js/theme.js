// ============ THEME ============
import { state } from './state.js';

export function getTheme() { return localStorage.getItem('ag_theme') || 'auto' }

export function applyTheme(t) {
  state.theme = t;
  const d = document.documentElement;
  if (t === 'dark') d.setAttribute('data-theme', 'dark');
  else if (t === 'light') d.removeAttribute('data-theme');
  else { if (matchMedia('(prefers-color-scheme:dark)').matches) d.setAttribute('data-theme', 'dark'); else d.removeAttribute('data-theme') }
  const themeBtn = document.getElementById('themeBtn');
  if (themeBtn) themeBtn.innerHTML = d.hasAttribute('data-theme') ? '&#x2600;' : '&#x263E;';
  const isDark = d.hasAttribute('data-theme');
  const hljsLink = document.getElementById('hljs-theme');
  if (hljsLink) {
    hljsLink.href = isDark
      ? 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/atom-one-dark.min.css'
      : 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/atom-one-light.min.css';
  }
}

export function setTheme(t) {
  localStorage.setItem('ag_theme', t); applyTheme(t);
}
