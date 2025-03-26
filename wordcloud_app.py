import streamlit as st
import pandas as pd
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 🎯 升級版：整合式 Section 分析器（文字雲、長條圖、雷達圖、圓餅圖）
st.set_page_config(page_title="分析神器", layout="wide", page_icon="📚")

font_path = "C:/Windows/Fonts/msjh.ttc"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name

st.title("📚 中文課多元資料分析工具")

uploaded_file = st.file_uploader("📂 上傳 CSV 檔案（需含 Section 雙層欄位）", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=[0, 1])

    # 建立欄位對應字典（Section ➜ [欄位名稱清單]）
    section_dict = {}
    for (section, question) in df.columns:
        if section == "基本資料":
            continue
        if section not in section_dict:
            section_dict[section] = []
        section_dict[section].append(question)

    selected_section = st.selectbox("請選擇主題區塊（Section）：", list(section_dict.keys()))
    section_cols = [(selected_section, q) for q in section_dict[selected_section]]

    # 過濾器（班級/學系/年級）
    for key in ["班級", "學系", "年級"]:
        df[("基本資料", key)] = df[("基本資料", key)].astype(str).str.strip()

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_class = st.multiselect("過濾班級：", sorted(df[("基本資料", "班級")].dropna().unique()))
    with col2:
        selected_dept = st.multiselect("過濾學系：", sorted(df[("基本資料", "學系")].dropna().unique()))
    with col3:
        selected_grade = st.multiselect("過濾年級：", sorted(df[("基本資料", "年級")].dropna().unique()))

    filtered_df = df.copy()
    if selected_class:
        filtered_df = filtered_df[filtered_df[("基本資料", "班級")].isin(selected_class)]
    if selected_dept:
        filtered_df = filtered_df[filtered_df[("基本資料", "學系")].isin(selected_dept)]
    if selected_grade:
        filtered_df = filtered_df[filtered_df[("基本資料", "年級")].isin(selected_grade)]

    numeric_cols = [col for col in section_cols if pd.api.types.is_numeric_dtype(filtered_df[col])]
    text_cols = [col for col in section_cols if not pd.api.types.is_numeric_dtype(filtered_df[col])]

    # 數值欄位 ➜ 雷達圖（平均值）
    if numeric_cols:
        st.markdown("### 📈 數值整合雷達圖")
        avg_scores = filtered_df[numeric_cols].mean().round(2)
        labels = [q[1] for q in avg_scores.index]
        values = avg_scores.values
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(f"{selected_section} 各面向平均分數雷達圖")
        st.pyplot(fig)

        st.dataframe(avg_scores.reset_index().rename(columns={0: "平均分數"}))

    # 類別欄位 ➜ 圓餅圖群組
    if text_cols:
        st.markdown("### 🧩 類別欄位圓餅統計圖")
        for col in text_cols:
            data = filtered_df[col].dropna().astype(str)
            if data.nunique() <= 10:
                fig, ax = plt.subplots()
                data.value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title(col[1])
                st.pyplot(fig)

        # 額外文字雲輔助（合併所有文字欄位）
        st.markdown("### ☁️ 質性補充文字雲")
        def get_word_freq(series_list):
            words = []
            for s in series_list:
                for text in s.dropna():
                    seg_list = jieba.lcut(str(text))
                    words.extend([w for w in seg_list if len(w) > 1])
            return Counter(words)

        word_freq = get_word_freq([filtered_df[col] for col in text_cols])
        if word_freq:
            wc = WordCloud(font_path=font_path, background_color='white', width=800, height=600)
            wc.generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            st.dataframe(pd.DataFrame(word_freq.most_common(20), columns=["詞語", "出現次數"]))
        else:
            st.info("⚠️ 沒有足夠文字資料可產出文字雲。")