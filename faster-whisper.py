import os
from faster_whisper import WhisperModel

def format_timestamp(seconds):
    """ 將秒數轉換為 MM:SS.sss 格式 """
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes:02}:{sec:06.3f}"

# **🔹 讀取術語.txt 來提供 initial_prompt**
def load_technical_terms(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            terms = [line.strip() for line in f.readlines() if line.strip()]
        return ", ".join(terms[:30])  # 只取前30個，避免 initial_prompt 太長
    except Exception as e:
        print(f"❌ 無法讀取術語文件: {e}")
        return ""

# **🔹 讀取術語並建立 initial_prompt**
terms_file = "電信網路術語_InitialPrompt.txt"  # 請確保 `術語.txt` 與程式在同一目錄
technical_terms = load_technical_terms(terms_file)
#此音訊檔主要語言為台灣國語，夾雜大量設備與電信術語，請輸出成繁體中文。本次會議主題為解釋如何導入無紙化登記制度。
#如遇到以下關鍵詞，請轉錄時維持其專業術語：{technical_terms}。

initial_prompt_text = f"""
本次轉錄請使用繁體中文及英文夾雜，並確保語句清晰，適當使用標點符號來斷句，以提高可讀性。請忽略冗餘的語助詞（如嗯、啊、喔），並確保技術術語保持原樣不被錯譯。

轉錄時，請根據語意適當分段，不要省略重要資訊，並避免不必要的換行。請確保數字、時間、單位（如 Mbps、kHz）及專有名詞（如 AI、ChatGPT、OpenAI）正確轉錄，不要誤譯或省略。

如果句子過長，請適當使用「，」或「。」來斷句，確保語句通順。例如：
錯誤示範：「這是一場有關 AI 的討論我們將探討深度學習語音辨識技術以及 ChatGPT 的應用」
正確示範：「這是一場有關 AI 的討論。我們將探討深度學習、語音辨識技術，以及 ChatGPT 的應用。」

如遇難以辨識的詞彙，請根據上下文合理判斷，嘗試轉錄，而非直接省略或拼音轉錄。

如遇到以下關鍵詞，請轉錄時維持其專業術語：{technical_terms}。
"""

# **🔹 讀取錯誤詞修正字典檔案**
def load_corrections(file_path):
    corrections = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    wrong, correct = parts
                    corrections[wrong] = correct
    except Exception as e:
        print(f"❌ 無法讀取錯誤詞修正檔案: {e}")
    return corrections

# **🔹 讀取 `corrections.txt`**
corrections = load_corrections("corrections.csv")

# **🔹 錯誤詞修正函數**
def correct_transcription(text):
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

# **🔹 避免 OpenMP 衝突**
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载模型
model = WhisperModel("medium", device="cpu", compute_type="int8")

# **🔹 轉錄函數**
def transcribe_audio(file_path):
    print(f"🎤 正在轉錄音檔: {file_path}")

    # 進行語音轉錄，設定 `initial_prompt`
    segments, info = model.transcribe(file_path, beam_size=6, initial_prompt=initial_prompt_text, vad_filter=True, temperature=0.2, word_timestamps=True)

    print(f"🌍 偵測語言：{info.language}")

    # **🔹 產生輸出檔案名稱**
    base_name = os.path.splitext(file_path)[0]
    output_txt = f"{base_name}.txt"

    # **🔹 存成 .txt**
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"🌍 偵測語言：{info.language}\n\n")
        for segment in segments:
            print(f"{segment.text}")
            corrected_text = correct_transcription(segment.text)  # **修正錯誤術語**
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)            
            f.write(f"[{start_time} --> {end_time}] {corrected_text}\n")

    print(f"✅ 轉錄完成，結果已儲存至: {output_txt}")

# **🔹 設定音檔名稱**
audio_file = f"20250616_4G轉5GS包裝修正DA效期議題.wav"  # **請修改為你的音檔名稱**

# **🔹 執行轉錄**
if __name__ == "__main__":
    transcribe_audio(audio_file)
