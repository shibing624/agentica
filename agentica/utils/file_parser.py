# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import csv
import json
from pathlib import Path
from typing import Optional

from agentica.utils.log import logger


def read_json_file(path: Path) -> str:
    """Read JSON file."""
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                json_contents = [json.loads(line) for line in f]
        elif path.suffix == ".json":
            json_contents = json.loads(path.read_text("utf-8"))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        if isinstance(json_contents, dict):
            json_contents = [json_contents]
        data = [json.dumps(content, ensure_ascii=False) for content in json_contents]
        return "\n".join(data)
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_csv_file(path: Path, row_limit: Optional[int] = None) -> str:
    """Read CSV file."""
    try:
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
    """Read text file."""
    try:
        return path.read_text("utf-8")
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_pdf_file(path: Path) -> str:
    """Read PDF file."""
    try:
        from docling.document_converter import DocumentConverter
    except (ImportError, ModuleNotFoundError):
        raise ImportError("`docling` not installed, please install using `pip install docling`")
    text = ""
    try:
        logger.info(f"Reading: {path}")
        converter = DocumentConverter()
        res = converter.convert(path)
        text = res.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
    return text


def read_pdf_url(url: str) -> str:
    """Read PDF file from URL."""
    text = ""
    try:
        from docling.document_converter import DocumentConverter
    except (ImportError, ModuleNotFoundError):
        raise ImportError("`docling` not installed, please install using `pip install docling`")
    try:
        logger.info(f"Reading: {url}")
        converter = DocumentConverter()
        res = converter.convert(url)
        text = res.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Error reading: {url}: {e}")
    return text


def read_docx_file_by_docling(path: Path) -> str:
    """Read DOCX file using docling."""
    text = ""
    try:
        from docling.document_converter import DocumentConverter
    except (ImportError, ModuleNotFoundError):
        raise ImportError("`docling` not installed, please install using `pip install docling`")
    try:
        logger.info(f"Reading: {path}")
        converter = DocumentConverter()
        res = converter.convert(path)
        text = res.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
    return text


def read_docx_file(path: Path) -> str:
    """Read DOCX file."""
    try:
        from docx import Document as DocxDocument
        from docx.oxml.ns import qn
    except ImportError:
        raise ImportError("`python-docx` not installed, please install using `pip install python-docx`")

    def get_hyperlink_target(doc, r_id):
        """获取超链接目标URL或者路径."""
        if r_id in doc.part.rels:
            rel = doc.part.rels[r_id]
            if hasattr(rel, 'target_ref'):
                return rel.target_ref
        return None

    def is_image_url(url):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        _, ext = os.path.splitext(url)
        return ext.lower() in image_extensions

    def get_numbering(paragraph):
        """获取段落的编号信息."""
        numbering = paragraph._element.xpath('.//w:numPr')
        if (numbering and len(numbering) > 0 and
                numbering[0].xpath('.//w:numId') and numbering[0].xpath('.//w:ilvl')):
            num_id = numbering[0].xpath('.//w:numId')[0].get(qn('w:val'))
            ilvl = numbering[0].xpath('.//w:ilvl')[0].get(qn('w:val'))
            return num_id, ilvl
        return None, None

    try:
        doc = DocxDocument(str(path))
        doc_content = ""
        numbering_dict = {}

        for paragraph in doc.paragraphs:
            para_text = ""
            num_id, ilvl = get_numbering(paragraph)
            if num_id and ilvl:
                if num_id not in numbering_dict:
                    numbering_dict[num_id] = {}
                if ilvl not in numbering_dict[num_id]:
                    numbering_dict[num_id][ilvl] = 1
                else:
                    numbering_dict[num_id][ilvl] += 1
                para_text += f"- "

            for run in paragraph.runs:
                run_text = run.text

                # 查找 Run 中的超链接
                hyperlink_elements = run._r.findall(".//w:hyperlink", namespaces=run._r.nsmap)
                for el in hyperlink_elements:
                    r_id = el.get(qn("r:id"))
                    if r_id:
                        hyperlink = get_hyperlink_target(doc, r_id)
                        if hyperlink:
                            text = "".join(node.text for node in el.findall(".//w:t", namespaces=run._r.nsmap))
                            md_url = f"![{text}]({hyperlink})" if is_image_url(hyperlink) else f"[{text}]({hyperlink})"
                            if text in run_text:
                                run_text = run_text.replace(text, md_url)
                            else:
                                run_text += md_url
                para_text += run_text

            # 查找段落中的超链接
            hyperlink_elements = paragraph._element.findall(".//w:hyperlink", namespaces=paragraph._element.nsmap)
            for el in hyperlink_elements:
                r_id = el.get(qn("r:id"))
                if r_id:
                    hyperlink = get_hyperlink_target(doc, r_id)
                    if hyperlink:
                        text = "".join(node.text for node in el.findall(".//w:t", namespaces=paragraph._element.nsmap))
                        md_url = f"![{text}]({hyperlink})" if is_image_url(hyperlink) else f"[{text}]({hyperlink})"
                        if text in para_text:
                            para_text = para_text.replace(text, md_url)
                        else:
                            para_text += md_url
            doc_content += para_text + "\n"

        # 处理插入的图片
        # for rel in doc.part.rels.values():
        #     if "image" in rel.target_ref and is_image_url(rel.target_ref):
        #         img_id = rel.rId
        #         img_part = doc.part.related_parts[img_id]
        #         img_data = img_part.blob
        #         # 生成保存图片的文件名
        #         os.makedirs(DATA_DIR, exist_ok=True)
        #         img_filename = os.path.join(DATA_DIR, f'image_{img_id}.png')
        #         # 保存图片数据到本地
        #         with open(img_filename, 'wb') as img_file:
        #             img_file.write(img_data)
        #         logger.debug(f"Image found: {rel.target_ref}, size: {len(img_data)} bytes")
        #         img_info = f"![Image]({img_filename})"
        #         doc_content += img_info + "\n"

        return doc_content
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_excel_file(path: Path) -> str:
    """Read excel file."""
    text = ""
    try:
        from docling.document_converter import DocumentConverter
    except (ImportError, ModuleNotFoundError):
        raise ImportError("`docling` not installed, please install using `pip install docling`")
    try:
        logger.info(f"Reading: {path}")
        converter = DocumentConverter()
        res = converter.convert(path)
        text = res.document.export_to_markdown()
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
    return text


if __name__ == '__main__':
    url = "https://arxiv.org/pdf/2412.15166"
    result = read_pdf_url(url)
    print(result)
