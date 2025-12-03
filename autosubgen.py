# autosubgen.py
# -*- coding: utf-8 -*-

import os
import sys
import logging
import json
import time
import torch
import whisper
import pysubs2
import psutil
import threading
from tqdm import tqdm
from typing import List, Dict, Any
import config

# 引入不同廠商的庫
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import anthropic

# 設定 Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. 定義抽象介面與具體實作 (Backend Adapters)
# ==========================================

class LLMBackend:
    """所有 LLM 提供者的基類 (Interface)"""
    def check_health(self):
        raise NotImplementedError
    
    def process_batch(self, system_prompt: str, user_content: str) -> str:
        raise NotImplementedError

class OpenAIBackend(LLMBackend):
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY")
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.MODELS["openai"]

    def check_health(self):
        try:
            self.client.models.list() # 簡單驗證
        except Exception as e:
            raise ConnectionError(f"OpenAI 連線失敗: {e}")

    def process_batch(self, system_prompt: str, user_content: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

class GeminiBackend(LLMBackend):
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("缺少 GEMINI_API_KEY")
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model_name = config.MODELS["gemini"]
        # Gemini 的設定
        self.generation_config = {
            "temperature": 0.3,
            "response_mime_type": "application/json" # 強制 JSON
        }

    def check_health(self):
        try:
            model = genai.GenerativeModel(self.model_name)
            model.generate_content("Hi")
        except Exception as e:
            raise ConnectionError(f"Gemini 連線失敗: {e}")

    def process_batch(self, system_prompt: str, user_content: str) -> str:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=self.generation_config
        )
        response = model.generate_content(user_content)
        return response.text

class ClaudeBackend(LLMBackend):
    def __init__(self):
        if not config.CLAUDE_API_KEY:
            raise ValueError("缺少 CLAUDE_API_KEY")
        self.client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        self.model = config.MODELS["claude"]

    def check_health(self):
        try:
            self.client.messages.create(
                model=self.model, max_tokens=1, messages=[{"role": "user", "content": "Hi"}]
            )
        except Exception as e:
            raise ConnectionError(f"Claude 連線失敗: {e}")

    def process_batch(self, system_prompt: str, user_content: str) -> str:
        # Claude 沒有原生的 json_object 模式參數，但在 Prompt 中強調即可，或使用 prefill
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.3,
            system=system_prompt + " Output must be valid JSON.",
            messages=[{"role": "user", "content": user_content}]
        )
        return message.content[0].text

# ==========================================
# 2. 主程式邏輯 (AutoSubGen)
# ==========================================

class AutoSubGen:
    def __init__(self, provider: str = "openai"):
        """
        初始化：設定計算設備並載入指定的 LLM Provider
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"正在使用計算設備: {self.device}")
        
        # --- Factory Pattern: 根據選擇實例化不同的 Backend ---
        logger.info(f"正在初始化 LLM Provider: {provider.upper()}...")
        try:
            if provider == "openai":
                self.llm = OpenAIBackend()
            elif provider == "gemini":
                self.llm = GeminiBackend()
            elif provider == "claude":
                self.llm = ClaudeBackend()
            else:
                raise ValueError("不支援的 Provider")
            
            # 統一進行健康檢查
            self.llm.check_health()
            logger.info(f"✅ {provider.upper()} API 連線驗證成功！")
            
        except Exception as e:
            logger.critical(f"🛑 API 初始化失敗: {e}")
            logger.critical("請檢查 config.py 中的 Key 是否正確。")
            sys.exit(1)

        self.whisper_model = None
        self._stop_monitoring = False  # 控制監控執行緒的標誌

    # (generate_output_paths, _load_whisper_model, transcribe_and_refine, translate_subtitles 
    #  這些方法邏輯不變，除了調用 _process_text_with_gpt 時不需要改動)
    
    # 省略重複代碼，只列出與 LLM 互動相關的修改...
    # --- 新增：系統資源監控方法 ---
    def _monitor_resources(self):
        """後臺執行緒：每隔幾秒印出系統記憶體佔用"""
        while not self._stop_monitoring:
            # 獲取記憶體資訊
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            percent = mem.percent
            
            # 使用 \033 顏色代碼讓它顯眼一點 (青色)
            # 格式：[System] RAM: 8.5GB / 32.0GB (26.5%)
            print(f"\033[96m[System Monitor] RAM Usage: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent}%)\033[0m")
            
            # 每 3 秒更新一次
            time.sleep(3)
    
    # 必須完整保留 generate_output_paths
    def generate_output_paths(self, video_path: str) -> dict:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir_config = config.OUTPUT_SETTINGS.get("save_dir", "")
        if save_dir_config:
            output_dir = os.path.abspath(save_dir_config)
        else:
            output_dir = os.path.dirname(os.path.abspath(video_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return {
            "en": os.path.join(output_dir, f"{base_name}{config.OUTPUT_SETTINGS['suffix_en']}"),
            "zh": os.path.join(output_dir, f"{base_name}{config.OUTPUT_SETTINGS['suffix_zh']}"),
            "merge": os.path.join(output_dir, f"{base_name}{config.OUTPUT_SETTINGS['suffix_merge']}")
        }
    
    # 必須完整保留 _load_whisper_model
    def _load_whisper_model(self):
        if self.whisper_model is None:
            logger.info(f"正在加載 Whisper 模型 ({config.WHISPER_MODEL_SIZE})...")
            if os.path.exists(config.WHISPER_MODEL_SIZE):
                logger.info(f"檢測到本地模型文件: {config.WHISPER_MODEL_SIZE}")
            self.whisper_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=self.device)

    # 轉錄方法 (保留 verbose=True)
    def transcribe_and_refine(self, video_path: str, output_path: str):
        if not os.path.exists(video_path): raise FileNotFoundError(f"找不到: {video_path}")
        self._load_whisper_model()
        logger.info(f"開始轉錄: {video_path}")

        # --- 1. 啟動監控執行緒 ---
        self._stop_monitoring = False
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True 
        monitor_thread.start()
        
        # --- 2. 使用 try...finally 確保監控會停止 ---
        try:
            # 執行耗時的 Whisper 轉錄
            result = self.whisper_model.transcribe(
                video_path, 
                language="en", 
                fp16=(self.device=="cuda"), 
                verbose=True
            )
        finally:
            # --- 3. 無論成功或失敗，這裏都會執行，停止監控 ---
            self._stop_monitoring = True
            monitor_thread.join() # 等待執行緒乾淨地結束
            print("\n") # 印個換行，讓版面好看點

        #result = self.whisper_model.transcribe(video_path, language="en", fp16=(self.device=="cuda"), verbose=True)
        
        segments = result['segments']
        logger.info(f"\n✅ 轉錄完成，共 {len(segments)} 行。")
        
        subs = pysubs2.SSAFile()
        raw_texts = []
        for seg in segments:
            evt = pysubs2.SSAEvent(start=int(seg['start']*1000), end=int(seg['end']*1000), text=seg['text'].strip())
            subs.events.append(evt)
            raw_texts.append(evt.text)
            
        logger.info("正在進行潤飾...")
        refined = self._process_text_unified(raw_texts, "refine") # 改名調用統一方法
        
        for i, txt in enumerate(refined[:len(subs.events)]): subs.events[i].text = txt
        subs.save(output_path)
        logger.info(f"已保存: {output_path}")

    # 翻譯方法
    def translate_subtitles(self, input_path: str, output_path: str):
        if not os.path.exists(input_path): raise FileNotFoundError(f"找不到: {input_path}")
        subs = pysubs2.load(input_path)
        logger.info("正在進行翻譯...")
        translated = self._process_text_unified([e.text for e in subs.events], "translate")
        
        for i, txt in enumerate(translated[:len(subs.events)]): subs.events[i].text = txt
        subs.save(output_path)
        logger.info(f"已保存: {output_path}")

    # 合併方法 (保持不變)
    # 在 autosubgen.py 中替換/新增以下方法

    def merge_subtitles(self, zh_path: str, en_path: str, output_path: str):
        
        logger.info("開始合併雙語字幕 (樣式復刻模式)...")
        
        try:
            subs_zh = pysubs2.load(zh_path)
            subs_en = pysubs2.load(en_path)
        except Exception as e:
            logger.error(f"讀取字幕文件失敗: {e}")
            return

        # 1. 建立新的字幕檔，並設定 Header 參數 (PlayRes)
        merged_subs = pysubs2.SSAFile()
        merged_subs.info.update(config.ASS_PARAMS) # 寫入 384x288 解析度設定

        # 2. 載入並註冊樣式
        style_main_cfg = config.STYLE_CONFIG["main"]
        style_sec_cfg = config.STYLE_CONFIG["second"]

        merged_subs.styles[style_main_cfg["Name"]] = self._create_pysubs2_style(style_main_cfg)
        merged_subs.styles[style_sec_cfg["Name"]] = self._create_pysubs2_style(style_sec_cfg)

        # 3. 合併邏輯
        # 由於 Whisper 生成的時間軸非常精準，中英行數通常一致。
        # 為了保險，我們使用時間戳記來尋找對應的英文字幕，而不是假設行號對應。
        
        # 建立英文事件的索引加速查找
        # 簡單策略：假設行數大致對應，若不對應則尋找時間重疊最大的
        
        logger.info(f"正在合併 {len(subs_zh)} 行字幕...")
        
        # 為了處理兩者行數不一致的情況，我們遍歷中文，去英文裏找對應
        for z_event in subs_zh.events:
            z_start = z_event.start
            z_end = z_event.end
            z_text = z_event.text.strip()
            
            # 尋找時間重疊最多的英文句子
            best_match_en = ""
            max_overlap = 0
            
            for e_event in subs_en.events:
                # 計算重疊時間
                overlap_start = max(z_start, e_event.start)
                overlap_end = min(z_end, e_event.end)
                overlap = overlap_end - overlap_start
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match_en = e_event.text.strip()
                
                # 優化：如果你已經過了這段時間，就不用再往下找了 (假設是有序的)
                if e_event.start > z_end:
                    break
            
            # 4. 構建雙語內容
            # 格式：中文\N{\rEng}English
            # \N 是換行，{\rEng} 是強制重置該行剩餘部分的樣式為 "Eng"
            if best_match_en:
                final_text = f"{z_text}\\N{{\\r{style_sec_cfg['Name']}}}{best_match_en}"
            else:
                final_text = z_text # 沒找到英文就只放中文

            # 建立新事件，使用主樣式 (Default)
            new_event = pysubs2.SSAEvent(
                start=z_start,
                end=z_end,
                text=final_text,
                style=style_main_cfg["Name"]
            )
            merged_subs.events.append(new_event)

        merged_subs.save(output_path)
        logger.info(f"✅ 雙語字幕合併完成，已保存為: {output_path}")

    def _create_pysubs2_style(self, cfg: dict) -> pysubs2.SSAStyle:
        """
        將 config 字典轉換為 pysubs2.SSAStyle 物件
        """
        style = pysubs2.SSAStyle()
        style.fontname = cfg.get("Fontname", "Arial")
        style.fontsize = cfg.get("Fontsize", 20)
        style.primarycolor = pysubs2.Color(*self._parse_ass_color(cfg.get("PrimaryColour")))
        style.secondarycolor = pysubs2.Color(*self._parse_ass_color(cfg.get("SecondaryColour")))
        style.outlinecolor = pysubs2.Color(*self._parse_ass_color(cfg.get("OutlineColour")))
        style.backcolor = pysubs2.Color(*self._parse_ass_color(cfg.get("BackColour")))
        style.bold = cfg.get("Bold", 0)
        style.italic = cfg.get("Italic", 0)
        style.borderstyle = cfg.get("BorderStyle", 1)
        style.outline = cfg.get("Outline", 2)
        style.shadow = cfg.get("Shadow", 0)
        style.alignment = cfg.get("Alignment", 2)
        style.marginl = cfg.get("MarginL", 10)
        style.marginr = cfg.get("MarginR", 10)
        style.marginv = cfg.get("MarginV", 10)
        style.encoding = cfg.get("Encoding", 1)
        return style

    def _parse_ass_color(self, ass_hex: str):
        hex_str = ass_hex.replace("&H", "")
        if len(hex_str) != 8: return 255, 255, 255, 0
        return int(hex_str[6:8], 16), int(hex_str[4:6], 16), int(hex_str[2:4], 16), int(hex_str[0:2], 16)

    # ==========================================
    # 3. 統一的處理核心 (Unified Processor)
    # ==========================================
    
    def _process_text_unified(self, texts: List[str], mode: str) -> List[str]:
        """
        統一處理邏輯：負責 Batch 切分、重試循環、調用後端
        """
        processed_texts = []
        batch_size = 20
        max_retries = 3
        
        if mode == "refine":
            sys_prompt = "Correct grammar/punctuation. Return a JSON list of strings."
        else:
            sys_prompt = "Translate to Traditional Chinese (Taiwan). Return a JSON list of strings."

        for i in tqdm(range(0, len(texts), batch_size), desc=f"AI Processing ({mode})"):
            batch = texts[i : i + batch_size]
            user_content = json.dumps(batch, ensure_ascii=False)
            
            # 重試機制
            for attempt in range(max_retries):
                try:
                    # 調用多態的 llm.process_batch
                    response_text = self.llm.process_batch(sys_prompt, f"Process: {user_content}")
                    
                    # 嘗試解析 JSON
                    try:
                        # 清理可能存在的 Markdown code block (例如 ```json ... ```)
                        clean_text = response_text.replace("```json", "").replace("```", "").strip()
                        data = json.loads(clean_text)
                        
                        if isinstance(data, dict):
                            batch_result = list(data.values())[0] if data.values() else batch
                        elif isinstance(data, list):
                            batch_result = data
                        else:
                            batch_result = batch
                    except json.JSONDecodeError:
                        logger.warning(f"Batch {i} JSON 解析失敗，嘗試修復或放棄...")
                        batch_result = batch

                    if len(batch_result) != len(batch):
                        batch_result = batch
                        
                    processed_texts.extend(batch_result)
                    break # 成功則跳出重試

                except Exception as e:
                    # 統一捕捉各家 API 的錯誤
                    logger.warning(f"API Error (Attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Batch {i} 最終失敗，使用原文。")
                        processed_texts.extend(batch)
        
        return processed_texts