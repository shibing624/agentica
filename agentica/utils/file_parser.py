# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import csv
import json
import ssl
import httpx
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import Optional

from agentica.utils.log import logger


def read_json_file(path: Path) -> str:
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
    try:
        return path.read_text("utf-8")
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


def read_pdf_file(path: Path, enable_image_ocr: bool = False) -> str:
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required to read PDF files: `pip install pypdf`")

    if enable_image_ocr:
        try:
            import imgocr
        except ImportError:
            raise ImportError("use `pip install imgocr`")

    text = ""
    try:
        # Read PDF text
        reader = pypdf.PdfReader(str(path))
        for page in reader.pages:
            text += page.extract_text() + "\n"

        if enable_image_ocr:
            # Read images and perform OCR
            ocr = imgocr.ImgOcr()
            for page in reader.pages:
                for image in page.images:
                    if image is None or not image:  # Check if the image data is valid
                        logger.warning(f"Invalid image data: {image}")
                        continue
                    img_array = np.array(image)
                    if img_array.ndim == 3 and img_array.shape[-1] == 4:  # Check if the image has an alpha channel
                        img_array = img_array[:, :, :3]  # Remove the alpha channel
                    elif img_array.ndim != 3 or img_array.shape[-1] not in [3, 4]:
                        logger.warning(f"Unexpected image shape: {img_array.shape}")
                        continue
                    ocr_result = ocr.ocr(img_array)
                    for result in ocr_result:
                        text += result["text"] + "\n"
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
    return text


def read_pdf_url(url: str, enable_image_ocr: bool = False) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("`pypdf` not installed, use `pip install pypdf`")
    if enable_image_ocr:
        try:
            import imgocr
        except ImportError:
            raise ImportError("use `pip install imgocr`")
    try:
        logger.info(f"Reading: {url}")
        # Create a default context for HTTPS requests (not recommended for production)
        ssl._create_default_https_context = ssl._create_unverified_context
        response = httpx.get(url)
        doc_reader = PdfReader(BytesIO(response.content))
        ocr_model = imgocr.ImgOcr() if enable_image_ocr else None
        text_content = ""
        for num, page in enumerate(doc_reader.pages):
            page_text = page.extract_text() or ""
            image_text_list = []
            if enable_image_ocr:
                for img_obj in page.images:
                    img_data = img_obj.data
                    # Perform OCR
                    ocr_result = ocr_model.ocr(img_data)
                    if ocr_result:
                        image_text_list += [i.get('text') for i in ocr_result]
            images_text = "\n".join(image_text_list)
            content = page_text + "\n" + images_text
            text_content += content
        return text_content
    except Exception as e:
        logger.error(f"Error reading: {url}: {e}")
        return ""


def read_docx_file(path: Path) -> str:
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
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise ImportError("`openpyxl` not installed, please install using `pip install openpyxl`")

    try:
        # 使用openpyxl读取Excel文件
        wb = load_workbook(str(path))
        sheet = wb.active
        logger.debug(f"Reading: {path}, load active Sheet: {sheet.title}")

        data = []
        for row in sheet.iter_rows(values_only=False):
            row_data = {}
            for idx, cell in enumerate(row):
                cell_name = sheet.cell(row=1, column=idx + 1).value
                if hasattr(cell, 'hyperlink') and cell.hyperlink:
                    row_val = f"[{cell.value}]({cell.hyperlink.target})"
                    row_data[cell_name] = row_val
                else:
                    row_data[cell_name] = cell.value
            data.append(row_data)

        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error reading: {path}: {e}")
        return ""


if __name__ == '__main__':
    url = "https://arxiv.org/pdf/2412.15166"
    result = read_pdf_url(url)
    print(result[:300])
