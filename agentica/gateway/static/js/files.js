// ============ DRAG & DROP / FILE HANDLING ============
import { state } from './state.js';
import { uploadFileApi } from './api.js';

export function setupDragDrop() {
  const box = document.getElementById('inputBox');
  box.addEventListener('dragover', e => { e.preventDefault(); box.classList.add('dragover') });
  box.addEventListener('dragleave', () => box.classList.remove('dragover'));
  box.addEventListener('drop', e => {
    e.preventDefault(); box.classList.remove('dragover');
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
}

export function onFilePick(e) {
  if (e.target.files.length) addFiles(e.target.files);
  e.target.value = '';
}
export function addFiles(fileList) {
  for (const f of fileList) state.pendingFiles.push(f);
}
export function removeFile(i) {
  state.pendingFiles.splice(i, 1);
}

export async function uploadFiles(targetDir) {
  const uploaded = [];
  for (const f of state.pendingFiles) {
    const { ok, data } = await uploadFileApi(f, targetDir);
    if (ok && data) uploaded.push(data.path);
    else uploaded.push(`[upload failed: ${f.name}]`);
  }
  state.pendingFiles = [];
  return uploaded;
}
