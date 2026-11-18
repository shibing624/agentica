# -*- coding: utf-8 -*-
"""Writing an uploaded file to disk without holding it in memory."""
import asyncio
import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

_CHUNK = 1024 * 1024


async def save_upload(file: UploadFile, dest: Path, max_mb: int) -> int:
    """Stream ``file`` into ``dest`` and return its size in bytes.

    The bytes go to a sibling temp file and replace ``dest`` only once the whole
    upload has arrived, so a refused or broken upload never leaves a truncated
    file (or clobbers an existing one). The size limit is enforced as chunks
    arrive: an oversized upload is cut off at the limit instead of being read
    to the end.
    """
    max_bytes = max_mb * 1024 * 1024
    part = dest.with_name(dest.name + ".part")
    total = 0
    try:
        with open(part, "wb") as out:
            while chunk := await file.read(_CHUNK):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File exceeds size limit of {max_mb}MB")
                await asyncio.to_thread(out.write, chunk)
        os.replace(part, dest)
    except HTTPException:
        part.unlink(missing_ok=True)
        raise
    except OSError as e:
        part.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to write file: {e}")
    return total
