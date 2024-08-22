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
        from docx import Document
        from docx.oxml.ns import qn
    except ImportError:
        raise ImportError("`python-docx` not installed, please install using `pip install python-docx`")

    def get_hyperlink_target(doc, r_id):
        """获取超链接目标URL或者路径."""
        if r_id in doc.part.rels:
            return doc.part.rels[r_id].target
        return None

    try:
        logger.info(f"Reading: {path}")
        doc = Document(str(path))
        doc_content = ""
        for paragraph in doc.paragraphs:
            para_text = ""
            for run in paragraph.runs:
                run_text = run.text

                # 查找 Run 中的超链接
                hyperlink_elements = run._r.findall(".//w:hyperlink", namespaces=run._r.nsmap)
                for el in hyperlink_elements:
                    r_id = el.get(qn("r:id"))
                    if r_id:
                        hyperlink = get_hyperlink_target(doc, r_id)
                        text = "".join(node.text for node in el.findall(".//w:t", namespaces=run._r.nsmap))
                        if hyperlink:
                            run_text = run_text.replace(text, f'{text} ({hyperlink})')
                para_text += run_text

            # 查找段落中的超链接（如果有独立的情况）
            hyperlink_elements = paragraph._element.findall(".//w:hyperlink", namespaces=paragraph._element.nsmap)
            for el in hyperlink_elements:
                r_id = el.get(qn("r:id"))
                if r_id:
                    hyperlink = get_hyperlink_target(doc, r_id)
                    text = "".join(node.text for node in el.findall(".//w:t", namespaces=paragraph._element.nsmap))
                    if hyperlink:
                        para_text = para_text.replace(text, f'{text} ({hyperlink})')
            doc_content += para_text + "\n"
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
