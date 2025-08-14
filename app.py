
import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="超輕量：中文客服檢索回覆", page_icon="💬", layout="wide")

st.title("💬 超輕量中文客服（僅檢索）")
st.caption("不上模型、純本地 TF‑IDF 檢索 → 直接回傳 FAQ 答案；適合超低資源/免費雲。")

# -----------------------
# 內建小型FAQ示範
# -----------------------
DEFAULT_FAQ = pd.DataFrame(
    [
        {"question":"你們的營業時間是？","answer":"我們的客服時間為週一至週五 09:00–18:00（國定假日除外）。"},
        {"question":"如何申請退貨？","answer":"請於到貨 7 天內透過訂單頁面點選『申請退貨』，系統將引導您完成流程。"},
        {"question":"運費如何計算？","answer":"單筆訂單滿 NT$ 1000 免運，未滿則酌收 NT$ 80。"},
        {"question":"可以開立發票嗎？","answer":"我們提供電子發票，請於結帳時填寫統一編號與抬頭。"},
    ]
)

if "faq_df" not in st.session_state:
    st.session_state.faq_df = DEFAULT_FAQ.copy()
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None

st.subheader("步驟1：上傳 FAQ / 知識庫 (CSV)")
st.write("需要兩欄：`question, answer`（UTF-8）。若未上傳，會使用示範資料。")
uploaded = st.file_uploader("上傳 CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        assert set(["question", "answer"]).issubset(df.columns), "CSV 需包含 question 與 answer 欄位"
        st.session_state.faq_df = df.dropna().reset_index(drop=True)
        st.success(f"已載入 {len(df)} 筆 FAQ。")
    except Exception as e:
        st.error(f"讀取CSV失敗：{e}")

with st.expander("查看目前 FAQ 資料", expanded=False):
    st.dataframe(st.session_state.faq_df, use_container_width=True)

st.subheader("步驟2：建立索引")
col_a, col_b = st.columns([1,2])
with col_a:
    do_index = st.button("建立 / 重建索引")
with col_b:
    st.info("使用中文斷詞（jieba）+ TF‑IDF。不上GPU、無需Transformers。")

def jieba_tokenize(text:str):
    return list(jieba.cut(text))

if do_index or (st.session_state.vectorizer is None):
    corpus = (st.session_state.faq_df["question"].astype(str) + " " + st.session_state.faq_df["answer"].astype(str)).tolist()
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize)
    tfidf = vectorizer.fit_transform(corpus)
    st.session_state.vectorizer = vectorizer
    st.session_state.tfidf = tfidf
    st.success("索引完成！")

st.subheader("步驟3：詢問客服")
q = st.text_input("輸入您的問題（中文）", placeholder="例如：想退貨要怎麼做？")
top_k = st.slider("取回前 K 筆", 1, 5, 3, 1)
threshold = st.slider("信心門檻（越高越嚴格）", 0.0, 1.0, 0.15, 0.01)

if st.button("送出") and q.strip():
    if (st.session_state.vectorizer is None) or (st.session_state.tfidf is None):
        st.warning("尚未建立索引，已自動建立。")
        corpus = (st.session_state.faq_df["question"].astype(str) + " " + st.session_state.faq_df["answer"].astype(str)).tolist()
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize)
        tfidf = vectorizer.fit_transform(corpus)
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf = tfidf

    vec = st.session_state.vectorizer.transform([q])
    sims = linear_kernel(vec, st.session_state.tfidf).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    rows = st.session_state.faq_df.iloc[idxs].copy()
    rows["score"] = sims[idxs]

    best_answer = None
    best_score = float(rows["score"].iloc[0]) if len(rows) else 0.0
    if len(rows) and best_score >= threshold:
        best_answer = rows["answer"].iloc[0]

    st.markdown("### 客服回覆")
    if best_answer:
        st.success(best_answer)
    else:
        st.warning("目前知識庫找不到足夠相似的答案。建議人工協助：請提供訂單編號與聯絡Email，我們將盡快回覆。")

    with st.expander("🔎 檢索依據（前K筆）", expanded=False):
        st.dataframe(rows[["question","answer","score"]], use_container_width=True)
    
st.markdown("---")
st.caption("備註：本App使用 jieba + scikit‑learn TF‑IDF 檢索，完全不需GPU。若要更流暢的自然語氣，可把本App當作後端，再串雲端LLM生成。")
