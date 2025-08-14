# 中文純文字客服生成（RAG + 指令模型）

這是一個可直接上線的 Streamlit 範例：上傳 FAQ（question,answer）→ 建立向量索引 → 使用中文指令模型生成客服回覆。

## 快速開始

```bash
pip install -r requirements.txt
streamlit run app.py
```

**第一次載入模型會自動從 Hugging Face 下載權重**。建議先嘗試較小的中文指令模型（如 `Qwen/Qwen2.5-0.5B-Instruct`）。若顯示卡記憶體不足，可勾選側邊欄的「4-bit 量化」。

## FAQ 資料格式

CSV 需包含兩欄：

```csv
question,answer
你們的營業時間是？,我們的客服時間為週一至週五 09:00–18:00（國定假日除外）。
如何申請退貨？,請於到貨 7 天內透過訂單頁面點選『申請退貨』，系統將引導您完成流程。
```

## 架構說明

- **檢索 (RAG)**：`intfloat/multilingual-e5-small` 做 embedding，`faiss-cpu` 建立向量索引。
- **生成**：可選任何 Hugging Face 中文指令模型（Qwen/ChatGLM 等）。
- **提示詞**：將取回的 FAQ 以「知識庫」注入，模型在此範圍內作答以減少幻覺。
- **查詢重寫**：簡單規則（可換更小的 LLM 做語義重寫）。

## 下一步建議

1. 將 JDDC / CSDS 等客服資料做 **SFT 微調**（PEFT/LoRA）以強化語氣與步驟化回答。
2. 將公司 SOP/政策文檔嵌入到同一向量庫，統一檢索。
3. 加上 **敏感內容過濾**（關鍵字或小分類器）與 **人工升級**（無法回答時建立工單）。
4. 加上 **多輪意圖追問** 模組（缺少必要資訊時禮貌詢問）。