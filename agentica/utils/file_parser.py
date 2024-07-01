# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import csv
import json
import ssl
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd

from agentica.utils.log import logger


def read_json_file(path: Path) -> str:
    try:
        logger.info(f"Reading: {path}")
        json_contents = json.loads(path.read_text("utf-8"))

        if isinstance(json_contents, dict):
            json_contents = [json_contents]

        data = [json.dumps(content, ensure_ascii=False) for content in json_contents]
        return "\n".join(data)
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_csv_file(path: Path, row_limit: Optional[int] = None) -> str:
    try:
        logger.info(f"Reading: {path}")
        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if row_limit is not None:
                csv_data = [row for row in reader][:row_limit]
            else:
                csv_data = [row for row in reader]
        return json.dumps(csv_data, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_txt_file(path: Path) -> str:
    try:
        logger.info(f"Reading: {path}")
        return path.read_text("utf-8")
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_pdf_file(path: Path) -> str:
    try:
        logger.info(f"Reading: {path}")
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required to read PDF files: `pip install pypdf`")
        with open(path, "rb") as fp:
            # Create a PDF object
            pdf = pypdf.PdfReader(fp)
            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)
            # This block returns a whole PDF as a single Document
            text = ""
            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                text += page_text
        return text
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_pdf_url(url: str) -> str:
    try:
        try:
            import httpx
        except ImportError:
            raise ImportError("`httpx` not installed")

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("`pypdf` not installed")
        logger.info(f"Reading: {url}")
        # Create a default context for HTTPS requests (not recommended for production)
        ssl._create_default_https_context = ssl._create_unverified_context
        response = httpx.get(url)
        doc_reader = PdfReader(BytesIO(response.content))
        content_list = [page.extract_text() for page in doc_reader.pages]
        return "\n".join(content_list)
    except Exception as e:
        logger.error(f"Error reading: {url}: {e}")
        return ""


def read_docx_file(path: Path) -> str:
    try:
        logger.info(f"Reading: {path}")
        try:
            import docx2txt
        except ImportError:
            raise ImportError("`docx2txt` not installed, please install using `pip install docx2txt`")
        doc_content = docx2txt.process(str(path))
        return doc_content
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_excel_file(path: Path) -> str:
    try:
        logger.info(f"Reading: {path}")
        data = pd.read_excel(path)
        data_dict = data.to_dict(orient="records")
        return json.dumps(data_dict, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""
