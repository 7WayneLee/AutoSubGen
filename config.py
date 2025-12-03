# config.py
# -*- coding: utf-8 -*-

import os

# --- API Keys ---
OPENAI_API_KEY = "" # 填入您的 Key
GEMINI_API_KEY = ""
CLAUDE_API_KEY = ""

# --- 模型名稱設定 ---
MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-3-5-haiku-20241022"
}

# --- Whisper 本地模型 ---
WHISPER_MODEL_SIZE = "turbo"

# --- 檔案輸出設定 ---
OUTPUT_SETTINGS = {
    "save_dir": "",
    "suffix_en": "_en_refined.ass",
    "suffix_zh": "_zh.ass",
    "suffix_merge": ".ass"  # 最終成品
}

# --- ASS 核心設定 (復刻您的參考文件) ---
# 為了達到相同的視覺效果，必須使用相同的 PlayRes (解析度基準)
ASS_PARAMS = {
    "PlayResX": 384,
    "PlayResY": 288,
    "WrapStyle": 0,
    "ScaledBorderAndShadow": "no"
}

# --- ASS 樣式表 ---
# 嚴格參照您提供的 Bugonia.2025 文件
STYLE_CONFIG = {
    # 主樣式 (對應文件中的 Default，用於中文)
    "main": {
        "Name": "Default",
        "Fontname": "Microsoft YaHei", # 微軟雅黑
        "Fontsize": 19,                # 參考文件數值
        "PrimaryColour": "&H00DFDFDF", # 灰白色
        "SecondaryColour": "&H0000FFFF",
        "OutlineColour": "&H00000000", # 黑色邊框
        "BackColour": "&H00000000",
        "Bold": 1,
        "Italic": 0,
        "BorderStyle": 1,
        "Outline": 2,
        "Shadow": 1,
        "Alignment": 2, # 底部居中
        "MarginL": 5,
        "MarginR": 5,
        "MarginV": 5,
        "Encoding": 134
    },
    # 副樣式 (對應文件中的 Eng，用於英文)
    "second": {
        "Name": "Eng",
        "Fontname": "Microsoft YaHei",
        "Fontsize": 11,                # 參考文件數值 (較小)
        "PrimaryColour": "&H00027CCF", # 參考文件中的橙金色
        "SecondaryColour": "&H00000000",
        "OutlineColour": "&H00000000",
        "BackColour": "&H00000000",
        "Bold": 1,
        "Italic": 0,
        "BorderStyle": 1,
        "Outline": 2,
        "Shadow": 1,
        "Alignment": 2,
        "MarginL": 5,
        "MarginR": 5,
        "MarginV": 5,
        "Encoding": 1
    }
}