功能概述  
该技能提供一套端到端的 PDF 处理方案，覆盖读取、生成、合并、拆分、旋转、加密、表单填充、OCR 文本与表格抽取等全链路操作。它面向两类场景：一是批量文档自动化，如财报、发票、合同的信息抽取与归档；二是动态报告生成，如将数据库结果即时渲染为带页眉页脚、图表、水印的正式 PDF。通过统一接口，用户无需切换多款软件即可在 Python 脚本或命令行中完成从数据到版式再到安全分发的完整闭环，显著降低运维与授权成本。

使用方法  
1. 环境准备  
   pip install pypdf pdfplumber reportlab qpdf pdftotext pytesseract pdf2image  
   确保系统已安装 poppler-utils、qpdf、Tesseract 二进制并加入 PATH。  
2. 读取与解析  
   用 pdfplumber.open() 逐页提取文本或表格，可自动识别网格并返回 DataFrame；若遇扫描件，先用 pdf2image 转 PNG，再调 pytesseract 做 OCR。  
3. 合并与拆分  
   在 Python 中实例化 PdfWriter，循环 add_page() 后 write() 即可合并；若按页拆分，则每页新建 PdfWriter 并单独写出。命令行可一条 qpdf --empty --pages … -- merged.pdf 完成同等操作。  
4. 创建与排版  
   轻量级内容用 reportlab.canvas 直接 drawString/drawLine；多段落、分页、目录等复杂排版用 Platypus 流式对象，先构建 story 列表，再 doc.build(story) 生成。  
5. 安全与后处理  
   对敏感文件，调用 writer.encrypt(userpwd, ownerpwd) 设置打开与权限密码；用 PdfMerger 追加水印或页码；旋转、裁剪、元数据修改均通过页面对象原位操作，保存即生效。  

预期输出  
文本输出：字符串或按页拆分的 txt 文件，保留原始行级布局；表格输出：pandas.DataFrame 列表或合并后的 Excel，列名自动对齐表头；新生成 PDF：支持 letter/A4 及自定义尺寸，可嵌入 Type1/TrueType 字体、JPEG/PNG 图片、矢量线；合并或拆分结果：完整保留原书签、链接、注释，页码顺序与命令参数一致；加密文件：128-bit AES 加密，支持打印、复制权限位细粒度控制；OCR 结果：含页码标记的 UTF-8 文本，可直接喂给下游 NLP 流程。所有输出路径与文件名由用户指定，脚本返回 0 表示成功，非 0 时 stderr 输出详细错误定位，便于日志采集与 CI 集成。