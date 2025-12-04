# AutoSubGen

**AutoSubGen** 是一個中英雙語字幕生成器,可支援多種格式影片 只需要將你要翻譯的影片丟入其中 等待一下 便可獲得該影片的中英文字幕.

## Features

-   **自動翻譯**: 藉助 [Whisper](https://github.com/openai/whisper) 自動生成影片字幕
-   **中英雙語**: 可生成英文和正體中文字幕
-   **可調整API**: 可選擇多種語言模型 API 接入

## 開始使用

### 環境支援

**Python**: 需要 Python 環境支援. Mac 用戶推薦使用 [homebrew](https://brew.sh/) 安裝
```bash
 brew install python3 
```

### 配置

**安裝依賴**
```bash
pip install openai google-generativeai anthropic pysubs2 openai-whisper tqdm psutil
```

### 開始使用

1. 在 [Release](https://github.com/7WayneLee/AutoSubGen/releases/tag/V1) 中下載 Source code 
2. 解壓縮 Source code
3. 完善 config.py
   
   ```python
    OPENAI_API_KEY = "" # 填入您的 Key
    GEMINI_API_KEY = ""
    CLAUDE_API_KEY = ""

    # --- 模型名稱設定 --- 可根據需求調整模型名稱
    MODELS = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
        "claude": "claude-3-5-haiku-20241022"
    }

    # --- Whisper 本地模型 --- 可根據需求調整 Whisper 模型名稱
    WHISPER_MODEL_SIZE = "turbo"

    # --- 檔案輸出設定 ---
    OUTPUT_SETTINGS = {
        "save_dir": "", #可以修改字幕保存地址 ""為保存在影片目錄下
        "suffix_en": "_en_refined.ass",
        "suffix_zh": "_zh.ass",
        "suffix_merge": ".ass"  
    }
   ```
4. run main.py
   ```bash
   python3 main.py
   ```



