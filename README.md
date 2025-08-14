# 超輕量中文客服（純檢索版）

不上任何語言模型，僅用 TF‑IDF 檢索，把最相似的 FAQ 答案直接回傳。
適合**免費雲/超低配**環境。

## 使用
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV 格式
`question,answer` 兩欄，UTF‑8。