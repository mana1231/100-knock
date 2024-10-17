import streamlit as st # type:ignore
import torch
import MeCab # type:ignore
from fairseq.models.transformer import TransformerModel # type:ignore
import re

@st.cache_resource
def load_model(model_path, checkpoint_file):
    model = TransformerModel.from_pretrained(
        model_path,
        checkpoint_file=checkpoint_file
    )
    model.eval()
    return model

def tokenize_ja(text):
    tagger = MeCab.Tagger()
    parsed = tagger.parse(text)
    
    tokens = []
    for line in parsed.split('\n'):
        if line == 'EOS' or line == '':
            continue
        surface, feature = line.split('\t')
        tokens.append(surface)
    
    return tokens

def translate(model, text):
    tokens = tokenize_ja(text)
    text_tok = ' '.join(tokens)
    output = model.translate(text_tok)
    output = re.sub(r'[@]', '', output)
    return output

def main():
    st.set_page_config(page_title="AI翻訳デモ", page_icon="🌐", layout="wide")

    st.title("🇯🇵 日英AI翻訳デモ 🇬🇧")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("入力 (日本語)")
        input_text = st.text_area("翻訳したい日本語を入力してください：", height=200)
    
    with col2:
        st.subheader("出力 (英語)")
        output_placeholder = st.empty()

    model_path = f"{main_path}/streamlit/"
    checkpoint_file = "model96/checkpoint_best.pt"
    model = load_model(model_path, checkpoint_file)

    if st.button("翻訳", key="translate_button", help="クリックして翻訳を開始"):
        if input_text:
            with st.spinner('翻訳中...'):
                translation = translate(model, input_text)
            output_placeholder.text_area("翻訳結果：", value=translation, height=200)
        else:
            st.warning("⚠️ テキストを入力してください。")

    st.markdown("---")
    st.markdown("### 📝 使い方")
    st.markdown("""
    1. 左側のテキストエリアに日本語を入力
    2. '翻訳'ボタンをクリック
    3. 右側のテキストエリアに英語の翻訳結果表示
    """)

if __name__ == "__main__":
    main()