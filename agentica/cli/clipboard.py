# -*- coding: utf-8 -*-
"""Cross-platform clipboard image extraction.

Provides ``save_clipboard_image(dest)`` that checks the system clipboard for
image data, saves it to *dest* as PNG, and returns True on success.

No external Python dependencies — uses only OS-level CLI tools:
  macOS   — osascript (always available), pngpaste (if installed)
  Windows — PowerShell via .NET System.Windows.Forms.Clipboard
  WSL2    — powershell.exe via .NET System.Windows.Forms.Clipboard
  Linux   — wl-paste (Wayland), xclip (X11)
"""

import base64
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_wsl_detected: bool | None = None


def save_clipboard_image(dest: Path) -> bool:
    """Extract an image from the system clipboard and save it as PNG.

    Returns True if an image was found and saved, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        return _macos_save(dest)
    if sys.platform == "win32":
        return _windows_save(dest)
    return _linux_save(dest)


def has_clipboard_image() -> bool:
    """Quick check: does the clipboard currently contain an image?"""
    if sys.platform == "darwin":
        return _macos_has_image()
    if sys.platform == "win32":
        return _windows_has_image()
    if _is_wsl():
        return _wsl_has_image()
    if os.environ.get("WAYLAND_DISPLAY"):
        return _wayland_has_image()
    return _xclip_has_image()


# ── macOS ────────────────────────────────────────────────────────────────

def _macos_save(dest: Path) -> bool:
    return _macos_pngpaste(dest) or _macos_osascript(dest)


def _macos_has_image() -> bool:
    try:
        info = subprocess.run(
            ["osascript", "-e", "clipboard info"],
            capture_output=True, text=True, timeout=3,
        )
        return "«class PNGf»" in info.stdout or "«class TIFF»" in info.stdout
    except Exception:
        return False


def _macos_pngpaste(dest: Path) -> bool:
    try:
        r = subprocess.run(["pngpaste", str(dest)], capture_output=True, timeout=3)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug("pngpaste failed: %s", e)
    return False


def _macos_osascript(dest: Path) -> bool:
    if not _macos_has_image():
        return False
    script = (
        'try\n'
        '  set imgData to the clipboard as «class PNGf»\n'
        f'  set f to open for access POSIX file "{dest}" with write permission\n'
        '  write imgData to f\n'
        '  close access f\n'
        'on error\n'
        '  return "fail"\n'
        'end try\n'
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "fail" not in r.stdout and dest.exists() and dest.stat().st_size > 0:
            return True
    except Exception as e:
        logger.debug("osascript clipboard extract failed: %s", e)
    return False


# ── PowerShell scripts (native Windows + WSL2) ──────────────────────────

_PS_CHECK_IMAGE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "[System.Windows.Forms.Clipboard]::ContainsImage()"
)

_PS_EXTRACT_IMAGE = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "Add-Type -AssemblyName System.Drawing;"
    "$img = [System.Windows.Forms.Clipboard]::GetImage();"
    "if ($null -eq $img) { exit 1 }"
    "$ms = New-Object System.IO.MemoryStream;"
    "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png);"
    "[System.Convert]::ToBase64String($ms.ToArray())"
)


# ── Native Windows ──────────────────────────────────────────────────────

_ps_exe: str | None | bool = False


def _find_powershell() -> str | None:
    for name in ("powershell", "pwsh"):
        try:
            r = subprocess.run(
                [name, "-NoProfile", "-NonInteractive", "-Command", "echo ok"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and "ok" in r.stdout:
                return name
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return None


def _get_ps_exe() -> str | None:
    global _ps_exe
    if _ps_exe is False:
        _ps_exe = _find_powershell()
    return _ps_exe if isinstance(_ps_exe, str) else None


def _windows_has_image() -> bool:
    ps = _get_ps_exe()
    if ps is None:
        return False
    try:
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command", _PS_CHECK_IMAGE],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and "True" in r.stdout
    except Exception as e:
        logger.debug("Windows clipboard image check failed: %s", e)
    return False


def _windows_save(dest: Path) -> bool:
    ps = _get_ps_exe()
    if ps is None:
        return False
    try:
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command", _PS_EXTRACT_IMAGE],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return False
        b64_data = r.stdout.strip()
        if not b64_data:
            return False
        dest.write_bytes(base64.b64decode(b64_data))
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        logger.debug("Windows clipboard image extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


# ── Linux ────────────────────────────────────────────────────────────────

def _is_wsl() -> bool:
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


def _linux_save(dest: Path) -> bool:
    if _is_wsl():
        if _wsl_save(dest):
            return True
    if os.environ.get("WAYLAND_DISPLAY"):
        if _wayland_save(dest):
            return True
    return _xclip_save(dest)


# ── WSL2 ──

def _wsl_has_image() -> bool:
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PS_CHECK_IMAGE],
            capture_output=True, text=True, timeout=8,
        )
        return r.returncode == 0 and "True" in r.stdout
    except FileNotFoundError:
        logger.debug("powershell.exe not found — WSL clipboard unavailable")
    except Exception as e:
        logger.debug("WSL clipboard check failed: %s", e)
    return False


def _wsl_save(dest: Path) -> bool:
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PS_EXTRACT_IMAGE],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return False
        b64_data = r.stdout.strip()
        if not b64_data:
            return False
        dest.write_bytes(base64.b64decode(b64_data))
        return dest.exists() and dest.stat().st_size > 0
    except FileNotFoundError:
        logger.debug("powershell.exe not found — WSL clipboard unavailable")
    except Exception as e:
        logger.debug("WSL clipboard extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


# ── Wayland ──

def _wayland_has_image() -> bool:
    try:
        r = subprocess.run(
            ["wl-paste", "--list-types"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and any(
            t.startswith("image/") for t in r.stdout.splitlines()
        )
    except FileNotFoundError:
        logger.debug("wl-paste not installed — Wayland clipboard unavailable")
    except Exception:
        pass
    return False


def _wayland_save(dest: Path) -> bool:
    try:
        types_r = subprocess.run(
            ["wl-paste", "--list-types"],
            capture_output=True, text=True, timeout=3,
        )
        if types_r.returncode != 0:
            return False
        types = types_r.stdout.splitlines()

        mime = None
        for preferred in ("image/png", "image/jpeg", "image/bmp", "image/gif", "image/webp"):
            if preferred in types:
                mime = preferred
                break
        if not mime:
            return False

        with open(dest, "wb") as f:
            subprocess.run(
                ["wl-paste", "--type", mime],
                stdout=f, stderr=subprocess.DEVNULL, timeout=5, check=True,
            )

        if not dest.exists() or dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            return False
        return True
    except FileNotFoundError:
        logger.debug("wl-paste not installed — Wayland clipboard unavailable")
    except Exception as e:
        logger.debug("wl-paste clipboard extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


# ── X11 ──

def _xclip_has_image() -> bool:
    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and "image/png" in r.stdout
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return False


def _xclip_save(dest: Path) -> bool:
    try:
        targets = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
            capture_output=True, text=True, timeout=3,
        )
        if "image/png" not in targets.stdout:
            return False
    except FileNotFoundError:
        logger.debug("xclip not installed — X11 clipboard image paste unavailable")
        return False
    except Exception:
        return False

    try:
        with open(dest, "wb") as f:
            subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                stdout=f, stderr=subprocess.DEVNULL, timeout=5, check=True,
            )
        if dest.exists() and dest.stat().st_size > 0:
            return True
    except Exception as e:
        logger.debug("xclip image extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False
