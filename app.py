# Streamlit + yfinance：輕量股價儀表板
# -------------------------------------------------------------
# 使用方式：
# 1) 安裝套件：pip install streamlit yfinance plotly pandas
# 2) 執行：streamlit run app.py
# -------------------------------------------------------------

import datetime as dt
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Dashboard (yfinance)", layout="wide")

# -----------------------------
# Sidebar：參數
# -----------------------------
st.sidebar.title("設定")
compare_mode = st.sidebar.checkbox("多標的比較 (標準化到同一起點)", value=False)

if compare_mode:
    tickers: List[str] = st.sidebar.text_input(
        "輸入代號 (逗號分隔)",
        value="AAPL, TSLA, NVDA",
        help="例如：AAPL, TSLA 或 ^TWII, 2330.TW (台積電)"
    ).replace(" ", "").split(",")
else:
    ticker = st.sidebar.text_input(
        "單一代號",
        value="AAPL",
        help="例如：AAPL、TSLA、NVDA、2330.TW、^GSPC"
    ).strip()
    tickers = [ticker]

# 日期區間
end_date = st.sidebar.date_input("結束日", dt.date.today())
start_default = end_date - dt.timedelta(days=365)
start_date = st.sidebar.date_input("開始日", start_default)

# K 線間隔
interval = st.sidebar.selectbox(
    "取樣頻率 (interval)",
    options=["1d", "1wk", "1mo"],
    index=0,
    help="日/週/月資料"
)

# 移動平均參數
ma_1 = st.sidebar.number_input("MA 窗口1", min_value=2, max_value=250, value=20)
ma_2 = st.sidebar.number_input("MA 窗口2", min_value=2, max_value=500, value=50)
show_volume = st.sidebar.checkbox("顯示成交量", value=True)

def _valid_range(s: dt.date, e: dt.date):
    if s >= e:
        st.sidebar.error("開始日需早於結束日")
        return False
    return True

# -----------------------------
# 下載資料（有 cache）
# -----------------------------
@st.cache_data(ttl=3600)
def load_prices(_tickers: List[str], start: dt.date, end: dt.date, interval: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t in _tickers:
        try:
            df = yf.download(
                t,
                start=start,
                end=end + dt.timedelta(days=1),  # 包含 end 當日
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                df["MA1"] = df["Close"].rolling(ma_1).mean()
                df["MA2"] = df["Close"].rolling(ma_2).mean()
                out[t] = df
        except Exception as ex:
            st.warning(f"下載 {t} 失敗：{ex}")
    return out

# -----------------------------
# 主區塊：標題 & 數據
# -----------------------------
st.title("📈 yfinance 股價儀表板")

if _valid_range(start_date, end_date):
    data = load_prices(tickers, start_date, end_date, interval)

    if not data:
        st.info("查無資料，請更換代號或時間區間。")
        st.stop()

    # 多標的比較：折線圖 (標準化到 100)
    if compare_mode:
        st.subheader("多標的走勢比較 (基準=100)")
        norm_df = pd.DataFrame()
        for t, df in data.items():
            if df.empty:
                continue
            base = df["Close"].iloc[0]
            series = (df["Close"] / base) * 100
            norm_df[t] = series
        st.line_chart(norm_df)

    # 單一標的：K 線 + MA + 成交量
    else:
        t = tickers[0]
        df = data.get(t)
        if df is None or df.empty:
            st.info("查無資料，請更換代號或時間區間。")
            st.stop()

        st.subheader(f"{t} 價格走勢（{interval}）")

        fig = go.Figure()
        # K 線
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="OHLC"
        ))
        # MA 線
        fig.add_trace(go.Scatter(x=df.index, y=df["MA1"], name=f"MA{ma_1}", mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA2"], name=f"MA{ma_2}", mode="lines"))

        fig.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="日期",
            yaxis_title="價格",
        )

        st.plotly_chart(fig, use_container_width=True)

        # 成交量 (獨立圖)
        if show_volume:
            vol = go.Figure()
            vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
            vol.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(vol, use_container_width=True)

        # 指標卡片
        # 指標卡片
        col1, col2, col3, col4 = st.columns(4)
        last_close = float(df["Close"].iloc[-1])
        ret = float(((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100)
        hi_ = float(df["High"].max())
        lo_ = float(df["Low"].min())
        vol_std = float(df["Close"].pct_change().std() * (252 ** 0.5) * 100)

        col1.metric("最新收盤", f"{last_close:,.2f}")
        col2.metric("區間報酬(%)", f"{ret:,.2f}")
        col3.metric("區間最高", f"{hi_:,.2f}")
        col4.metric("區間最低", f"{lo_:,.2f}")
        st.caption(f"年化波動率(近似)：{vol_std:.2f}%")

        # 原始資料表 & 下載
        with st.expander("查看原始資料表 / 下載 CSV"):
            st.dataframe(df)
            csv = df.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="下載 CSV",
                data=csv,
                file_name=f"{t}_{start_date}_{end_date}_{interval}.csv",
                mime="text/csv",
            )

    # 批次下載（多標的）
    with st.expander("批次下載（所有輸入的標的）"):
        # 以 Excel 打包多工作表
        if data:
            from io import BytesIO
            buff = BytesIO()
            with pd.ExcelWriter(buff, engine="xlsxwriter") as writer:
                for t, df in data.items():
                    if df.empty:
                        continue
                    df.to_excel(writer, sheet_name=t[:31])  # Excel sheet 名稱限制 31 字
            st.download_button(
                label="下載多標的 Excel",
                data=buff.getvalue(),
                file_name=f"prices_{start_date}_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# -----------------------------
# 底部說明
# -----------------------------
st.markdown(
    """
    **說明**  
    - 資料來源：`yfinance`（Yahoo Finance）。  
    - 多標的比較會將每個標的的收盤價標準化到同一起點（第一天=100），方便視覺比較表現。  
    - 單一標的視圖提供 K 線、雙 MA、成交量、關鍵指標。  
    """
)
