
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

st.set_page_config(page_title="中文客服生成 (RAG + 指令微調模型)", page_icon="💬", layout="wide")

# -----------------------
# Sidebar: 模型 & 參數
# -----------------------
with st.sidebar:
    st.markdown("## 模型設定")
    model_id = st.text_input(
        "Hugging Face 模型（建議：Qwen/Qwen2.5-0.5B-Instruct 或 1.5B）",
        value="Qwen/Qwen2.5-0.5B-Instruct"
    )
    use_4bit = st.checkbox("使用 4-bit 量化（需安裝 bitsandbytes；雲端不一定支援）", value=False)
    max_new_tokens = st.slider("回覆長度 (max_new_tokens)", 64, 1024, 256, 16)
    temperature = st.slider("溫度 (temperature)", 0.0, 1.5, 0.3, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    st.divider()
    st.markdown("### 檢索設定")
    top_k = st.slider("取回知識條目數 (top_k)", 1, 10, 3, 1)
    use_query_rewrite = st.checkbox("先重寫用戶問題（可提升檢索準確度）", value=True)

st.title("💬 中文客服生成 (RAG + 指令模型)")
st.caption("上傳 FAQ/知識庫 → 檢索 → 注入模型提示 → 產生客服回覆。預設支援中文。")

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

# -----------------------
# Session state
# -----------------------
if "faq_df" not in st.session_state:
    st.session_state.faq_df = DEFAULT_FAQ.copy()
if "index" not in st.session_state:
    st.session_state.index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role":"user/assistant", "content":str}

# -----------------------
# 上傳 FAQ/知識庫
# -----------------------
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

# -----------------------
# 建立向量索引
# -----------------------
st.subheader("步驟2：建立/更新 檢索索引")
if st.button("建立索引 / 重新索引"):
    with st.spinner("正在建立向量索引..."):
        # 多語模型，支援中文
        st.session_state.embedder = SentenceTransformer("intfloat/multilingual-e5-small")
        # e5 small 用法：需要在前面加上 "query: " / "passage: "
        corpus = ["passage: " + str(q) + " " + str(a) for q, a in zip(st.session_state.faq_df["question"], st.session_state.faq_df["answer"])]
        emb = st.session_state.embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)  # 內積=cosine (因已 normalize)
        index.add(emb)
        st.session_state.index = index
    st.success("索引完成！")

# -----------------------
# 載入文字生成模型
# -----------------------
st.subheader("步驟3：載入文字生成模型")
def load_model_tokenizer(model_id: str, use_4bit: bool=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb_config, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
    return tokenizer, model

if st.button("載入/更新 模型"):
    with st.spinner("正在載入模型與權重...（首次載入需較久）"):
        try:
            tok, mdl = load_model_tokenizer(model_id, use_4bit)
            st.session_state.tokenizer = tok
            st.session_state.model = mdl
            st.success(f"已載入：{model_id}")
        except Exception as e:
            st.error(f"載入失敗：{e}")

# -----------------------
# 檢索輔助：重寫查詢（選用）
# -----------------------
def rewrite_query(q: str) -> str:
    # 簡單啟發式重寫：移除贅詞、繁簡混合清理，可換成小模型生成（此處保持離線）
    q = q.strip()
    for t in ["請問", "麻煩", "一下", "可以", "能否", "想問", "謝謝", "謝了", "哈囉", "您好", "你好"]:
        q = q.replace(t, "")
    return q

# -----------------------
# 產生回覆（流式）
# -----------------------
def generate_stream(prompt, max_new_tokens=256, temperature=0.3, top_p=0.9):
    tok = st.session_state.tokenizer
    mdl = st.session_state.model
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        streamer=streamer
    )
    thread = Thread(target=mdl.generate, kwargs=gen_kwargs)
    thread.start()
    partial = ""
    for new_text in streamer:
        partial += new_text
        yield partial

# -----------------------
# Chat 區塊
# -----------------------
st.subheader("步驟4：開始對話")
user_input = st.text_input("輸入您的問題（中文）", placeholder="例如：想退貨要怎麼做？")

col1, col2 = st.columns([3,1])
with col1:
    send = st.button("送出", use_container_width=True)
with col2:
    clear = st.button("清除對話", use_container_width=True)

if clear:
    st.session_state.history = []
    st.experimental_rerun()

# 顯示對話歷史
for msg in st.session_state.history:
    with st.chat_message("user" if msg["role"]=="user" else "assistant"):
        st.markdown(msg["content"])

def build_prompt(query, retrieved):
    sys = (
        "你是專業、耐心的中文客服。"
        "請根據『知識庫』提供可靠且簡潔的回答，必要時提出可行步驟。"
        "若知識庫無答案，請坦誠說明並提供人工協助管道（例如客服信箱/工單流程的佔位描述）。"
        "禁止捏造訂單或個資；不要要求提供信用卡等敏感資訊。"
    )
    kb_text = "\\n\\n".join([f"[{i+1}] Q: {r['q']}\\nA: {r['a']}" for i, r in enumerate(retrieved)])
    chat_history = "\\n".join([f"{'客戶' if m['role']=='user' else '客服'}: {m['content']}" for m in st.session_state.history[-6:]])
    prompt = f"""<|system|>
{sys}
<|knowledge_base|>
{kb_text if kb_text else '（目前沒有可用的知識庫條目）'}
<|history|>
{chat_history}
<|user|>
{query}
<|assistant|>"""
    return prompt

if send and user_input:
    # 1) 查詢重寫
    q = rewrite_query(user_input) if use_query_rewrite else user_input

    # 2) 檢索
    retrieved = []
    if st.session_state.index is not None and st.session_state.embedder is not None:
        q_emb = st.session_state.embedder.encode(["query: " + q], convert_to_numpy=True, normalize_embeddings=True)
        D, I = st.session_state.index.search(q_emb, top_k)
        for idx in I[0]:
            if 0 <= idx < len(st.session_state.faq_df):
                row = st.session_state.faq_df.iloc[idx]
                retrieved.append({"q": str(row["question"]), "a": str(row["answer"])})
    else:
        # 若尚未索引，使用前三筆示例
        df = st.session_state.faq_df.head(top_k)
        for _, row in df.iterrows():
            retrieved.append({"q": str(row["question"]), "a": str(row["answer"])})

    # 3) 建立提示詞
    prompt = build_prompt(user_input, retrieved)

    st.session_state.history.append({"role":"user", "content": user_input})
    with st.chat_message("assistant"):
        if st.session_state.model is None or st.session_state.tokenizer is None:
            st.warning("尚未載入模型。請在側邊欄點擊『載入/更新 模型』。")
        else:
            # 流式輸出
            holder = st.empty()
            final_text = ""
            for partial in generate_stream(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p):
                final_text = partial
                holder.markdown(final_text)
            st.session_state.history.append({"role":"assistant", "content": final_text})

st.markdown("---")
st.caption("小貼士：")
st.caption("1) 先用 FAQ 做 RAG，可快速上線；2) 之後再用 JDDC/CSDS 等中文客服資料做 SFT 微調；3) 雲端部署可用 Streamlit Cloud / Hugging Face Spaces。")
