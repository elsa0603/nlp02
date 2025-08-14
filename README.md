# 🤗 NLP Toolkit — Streamlit（含 URL 抓取）

支援 Hugging Face `transformers` 的多個 `pipeline`：
- Sentiment Analysis
- Question Answering
- Summarization
- Named Entity Recognition
- Zero-shot Classification

並可：
- 上傳 `.txt / .pdf / .docx` 作為內容來源
- 直接貼上文字
- **輸入網址（URL）自動抓取並抽取正文**（使用 `readability-lxml` + `BeautifulSoup`）

## 使用方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 注意事項
- 有些網站需要登入、動態渲染或反爬保護，可能無法直接抓取；此時建議手動複製貼上。
- 抓取僅供學習與個人測試，請留意網站的服務條款與版權。
- PDF/DOCX 解析使用 `PyPDF2` 與 `docx2txt`，掃描型 PDF 可能無法正確擷取文字。
- 摘要會將長文切塊後逐段摘要再合併。