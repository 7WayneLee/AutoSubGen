# main.py
# -*- coding: utf-8 -*-

import os
import sys
import ssl
# 忽略 SSL 驗證 (macOS 必需)
ssl._create_default_https_context = ssl._create_unverified_context

from autosubgen import AutoSubGen

def get_user_input_path():
    while True:
        path_input = input("\n🎥 請輸入影片路徑 (直接拖入文件): ").strip()
        if (path_input.startswith('"') and path_input.endswith('"')) or \
           (path_input.startswith("'") and path_input.endswith("'")):
            path_input = path_input[1:-1]
        if os.path.exists(path_input) and os.path.isfile(path_input):
            return path_input
        print("❌ 檔案不存在，請重試。")

def select_provider():
    """讓用戶選擇 AI 提供商"""
    print("\n🧠 請選擇要使用的 AI 模型:")
    print("1. Chatgpt")
    print("2. Gemini")
    print("3. Claude")
    
    while True:
        choice = input("👉 請輸入編號 [1-3]: ").strip()
        if choice == "1": return "openai"
        if choice == "2": return "gemini"
        if choice == "3": return "claude"
        print("輸入錯誤，請輸入 1, 2 或 3。")

def main():
    print("=== AutoSubGen v2.0 (Multi-Provider) ===")
    
    # 1. 選擇影片
    video_file = get_user_input_path()
    
    # 2. 選擇 AI 提供商
    provider = select_provider()
    
    # 3. 初始化處理器 (傳入選擇的 provider)
    try:
        app = AutoSubGen(provider=provider)
        paths = app.generate_output_paths(video_file)
        
        print(f"\n✅ 影片: {os.path.basename(video_file)}")
        print(f"✅ AI 引擎: {provider.upper()}")
        print(f"📂 輸出目錄: {os.path.dirname(paths['merge'])}")
        print("-" * 40)

        # Step 1
        if not os.path.exists(paths['en']):
            print("\n[Step 1] 語音轉錄 (Whisper) & 潤飾...")
            app.transcribe_and_refine(video_file, paths['en'])
        else:
            print("\n[Step 1] 跳過 (檔案已存在)")

        # Step 2
        if not os.path.exists(paths['zh']):
            print("\n[Step 2] 翻譯 (Translation)...")
            app.translate_subtitles(paths['en'], paths['zh'])
        else:
            print("\n[Step 2] 跳過 (檔案已存在)")

        # Step 3
        print("\n[Step 3] 合併字幕...")
        app.merge_subtitles(paths['zh'], paths['en'], paths['merge'])

        print(f"\n🎉 全部完成！最終檔案: {paths['merge']}")

    except Exception as e:
        print(f"\n❌ 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()