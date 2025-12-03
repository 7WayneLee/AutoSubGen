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
            self.client.models.list() 
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
        self.generation_config = {
            "temperature": 0.3,
            "response_mime_type": "application/json"
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
        """初始化：設定計算設備並載入指定的 LLM Provider"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"正在使用計算設備: {self.device}")
        
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
            
            self.llm.check_health()
            logger.info(f"✅ {provider.upper()} API 連線驗證成功！")
            
        except Exception as e:
            logger.critical(f"🛑 API 初始化失敗: {e}")
            logger.critical("請檢查 config.py 中的 Key 是否正確。")
            sys.exit(1)

        self.whisper_model = None
        self._stop_monitoring = False

    # --- 系統資源監控 ---
    def _monitor_resources(self):
        while not self._stop_monitoring:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            percent = mem.percent
            print(f"\033[96m[System Monitor] RAM Usage: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent}%)\033[0m")
            time.sleep(3)
    
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
    
    def _load_whisper_model(self):
        if self.whisper_model is None:
            logger.info(f"正在加載 Whisper 模型 ({config.WHISPER_MODEL_SIZE})...")
            if os.path.exists(config.WHISPER_MODEL_SIZE):
                logger.info(f"檢測到本地模型文件: {config.WHISPER_MODEL_SIZE}")
            self.whisper_model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=self.device)

    # --- 功能一：轉錄與潤飾 (修正監控邏輯) ---
    def transcribe_and_refine(self, video_path: str, output_path: str):
        if not os.path.exists(video_path): raise FileNotFoundError(f"找不到: {video_path}")
        
        self._load_whisper_model()
        logger.info(f"開始轉錄: {video_path}")

        # 1. 啟動監控
        self._stop_monitoring = False
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True 
        monitor_thread.start()
        
        result = None

        # 2. 執行轉錄 (含錯誤防護)
        try:
            result = self.whisper_model.transcribe(
                video_path, 
                language="en", 
                fp16=(self.device=="cuda"), 
                verbose=True
            )
        finally:
            # 3. 停止監控
            self._stop_monitoring = True
            monitor_thread.join()
            print("\n") 

        # 4. 後續處理
        if result:
            segments = result['segments']
            logger.info(f"✅ 轉錄完成，共 {len(segments)} 行。")
            
            subs = pysubs2.SSAFile()
            raw_texts = []
            for seg in segments:
                evt = pysubs2.SSAEvent(start=int(seg['start']*1000), end=int(seg['end']*1000), text=seg['text'].strip())
                subs.events.append(evt)
                raw_texts.append(evt.text)
            
            logger.info("正在進行潤飾...")
            # 潤飾不需要全局分析，直接調用
            refined = self._process_text_unified(raw_texts, "refine") 
            
            min_len = min(len(refined), len(subs.events))
            for i in range(min_len):
                subs.events[i].text = refined[i]
                
            subs.save(output_path)
            logger.info(f"已保存: {output_path}")

    # --- 功能二：翻譯 (兩階段流程：分析 -> 翻譯) ---
    def translate_subtitles(self, input_path: str, output_path: str):
        if not os.path.exists(input_path): raise FileNotFoundError(f"找不到: {input_path}")
        
        subs = pysubs2.load(input_path)
        original_texts = [event.text for event in subs.events]
        
        # --- Step 1: 執行全局分析 (Pass 1) ---
        global_context_str = self._analyze_global_context(original_texts)
        
        # --- Step 2: 執行翻譯 (Pass 2) ---
        logger.info("正在進行第二階段：逐段翻譯 (Pass 2)...")
        translated_texts = self._process_text_unified(
            original_texts, 
            mode="translate",
            global_context=global_context_str # 傳入聖經
        )
        
        min_len = min(len(translated_texts), len(subs.events))
        for i in range(min_len):
            subs.events[i].text = translated_texts[i]
            
        subs.save(output_path)
        logger.info(f"已保存: {output_path}")

    # --- Pass 1: 全局分析核心 ---
    def _analyze_global_context(self, all_texts: List[str]) -> str:
        """閱讀完整劇本，生成劇情大綱與角色關係表 (翻譯聖經)"""
        logger.info("正在進行第一階段：全局劇情與角色分析 (生成翻譯聖經)...")
        
        # 限制長度防呆
        full_script = "\n".join(all_texts)[:100000]
        
        analysis_prompt = (
            "You are a lead localization expert. Read the provided movie script/subtitles below.\n"
            "Create a concise 'Translation Bible' to guide the translators.\n"
            "Output strictly a JSON object (no markdown) with the following keys:\n"
            "1. 'summary': A 3-sentence plot summary.\n"
            "2. 'tone': The overall tone (e.g., Serious, Comedic, Formal, Slang-heavy).\n"
            "3. 'characters': A list of main characters with their GENDER (Male/Female) and RELATIONSHIPS (e.g., 'A is B's boss', 'C and D are lovers'). This is crucial for Chinese pronouns (他/她) and honorifics (你/您).\n"
            "4. 'key_terms': Key proper nouns or jargon that need consistent translation.\n"
        )

        try:
            response = self.llm.process_batch(analysis_prompt, f"Script Content:\n{full_script}")
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            logger.info("✅ 翻譯聖經生成完成。")
            return cleaned_response
        except Exception as e:
            logger.warning(f"⚠️ 全局分析失敗: {e}。將使用通用規則進行翻譯。")
            return ""

    # 修改 autosubgen.py 中的這個方法
    def _get_enhanced_system_prompt(self, mode: str, global_context: str = "") -> str:
        if mode == "refine":
            return (
                "You are a professional subtitle editor. "
                "Correct grammar, casing, and punctuation errors. "
                "Keep the subtitles concise. Do NOT change the meaning. "
                "Output strictly a JSON list of strings."
            )
        
        # Translate 模式 (加入第 7 點規則：反直譯與邏輯檢查)
        base_prompt = (
            "你是 Netflix 等級的資深字幕翻譯員，專精於將英文翻譯成「正體中文（台灣）」。\n"
            "你的任務是根據「劇情背景」與「上下文」，將輸入的字幕翻譯成流暢、自然的台灣口語。\n\n"
            "### 核心翻譯規則 (必須遵守) ###\n"
            "1. **在地化用語**：絕對避免中國大陸用語，必須使用台灣習慣用語。\n"
            "   - (例：視頻->影片, 質量->品質, 項目->專案, 軟件->軟體, 信息->資訊, 默認->預設, 網絡->網路)\n"
            "2. **語氣與敬語**：這由角色關係決定。對上級或陌生人使用「您」，對平輩或下屬使用「你」。\n"
            "3. **簡潔精準**：字幕不僅要準確，還要簡短有力，適合閱讀。\n"
            "4. **專有名詞**：若背景設定中有指定譯名，請嚴格遵守。\n"
            "5. **格式要求**：絕對不要輸出任何解釋或Markdown標記，**只輸出純 JSON 字串列表**。\n"
            "6. **標點與排版**：中文字幕內容**不可包含任何標點符號**（如：，。？！）。若句子中間需要停頓或斷句，請強制使用「空格」代替；句尾也不要加符號。\n"
            "7. **拒絕直譯 (關鍵)**：翻譯必須基於整句邏輯與語境。遇到英文慣用語 (Idioms)、倒裝句或強調句（如 'for the life of me', 'over my dead body'），**必須意譯其「言外之意」**，嚴禁逐字翻譯造成邏輯錯誤。\n"
        )

        if global_context:
            base_prompt += (
                "\n### 劇情背景與角色關係 (Translation Bible) ###\n"
                "請參考以下設定來決定對話的語氣（敬語/粗俗/正式）：\n"
                "------------------------------------------------\n"
                f"{global_context}\n"
                "------------------------------------------------\n"
            )
        
        base_prompt += (
            "\n### 動態輸入說明 ###\n"
            "你將收到一個 JSON 物件，包含：\n"
            "- 'previous_context': 上一段對話內容（僅供參考，用於連貫語氣）。\n"
            "- 'lines_to_process': 需要翻譯的英文句子列表。\n\n"
            "請參考 'previous_context' 的語境，僅翻譯 'lines_to_process' 部分。"
        )
        return base_prompt
    
    # --- Pass 2: 統一處理核心 (滑動窗口 + 注入 Prompt) ---
    def _process_text_unified(self, texts: List[str], mode: str, global_context: str = "") -> List[str]:
        processed_texts = []
        batch_size = 20
        max_retries = 3
        
        sys_prompt = self._get_enhanced_system_prompt(mode, global_context)
        previous_context = [] # 滑動窗口

        for i in tqdm(range(0, len(texts), batch_size), desc=f"AI Processing ({mode})"):
            batch = texts[i : i + batch_size]
            
            # 建構輸入資料：包含上下文 + 本次要翻譯的句子
            context_str = "\n".join(previous_context) if previous_context else "無 (對話開始)"
            user_content_obj = {
                "previous_context": context_str,
                "lines_to_process": batch
            }
            
            # Refine 模式使用簡單 JSON，Translate 模式使用帶上下文的 JSON
            if mode == "refine":
                user_content_str = json.dumps(batch, ensure_ascii=False)
            else:
                user_content_str = json.dumps(user_content_obj, ensure_ascii=False)

            for attempt in range(max_retries):
                try:
                    response_text = self.llm.process_batch(sys_prompt, f"Input Data:\n{user_content_str}")
                    
                    try:
                        clean_text = response_text.replace("```json", "").replace("```", "").strip()
                        data = json.loads(clean_text)
                        
                        if isinstance(data, dict):
                            # 嘗試找 list 類型的 value
                            values = [v for v in data.values() if isinstance(v, list)]
                            batch_result = values[0] if values else batch
                        elif isinstance(data, list):
                            batch_result = data
                        else:
                            batch_result = batch
                    except json.JSONDecodeError:
                        logger.warning(f"Batch {i} JSON 解析失敗，使用原文。")
                        batch_result = batch

                    if len(batch_result) != len(batch):
                        if len(batch_result) > len(batch):
                            batch_result = batch_result[:len(batch)]
                        else:
                            batch_result.extend(batch[len(batch_result):])

                    processed_texts.extend(batch_result)
                    
                    # 更新上下文：取最後 3 句翻譯結果
                    previous_context = batch_result[-3:]
                    break 

                except Exception as e:
                    logger.warning(f"API Error (Attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Batch {i} 失敗，使用原文。")
                        processed_texts.extend(batch)
                        previous_context = batch[-3:]
        
        return processed_texts

    # --- 字幕合併 (復刻樣式) ---
    def merge_subtitles(self, zh_path: str, en_path: str, output_path: str):
        logger.info("開始合併雙語字幕 (樣式復刻模式)...")
        try:
            subs_zh = pysubs2.load(zh_path)
            subs_en = pysubs2.load(en_path)
        except Exception as e:
            logger.error(f"讀取字幕文件失敗: {e}")
            return

        merged_subs = pysubs2.SSAFile()
        merged_subs.info.update(config.ASS_PARAMS)

        style_main_cfg = config.STYLE_CONFIG["main"]
        style_sec_cfg = config.STYLE_CONFIG["second"]

        merged_subs.styles[style_main_cfg["Name"]] = self._create_pysubs2_style(style_main_cfg)
        merged_subs.styles[style_sec_cfg["Name"]] = self._create_pysubs2_style(style_sec_cfg)

        logger.info(f"正在合併 {len(subs_zh)} 行字幕...")
        
        for z_event in subs_zh.events:
            z_start = z_event.start
            z_end = z_event.end
            z_text = z_event.text.strip()
            
            best_match_en = ""
            max_overlap = 0
            
            for e_event in subs_en.events:
                overlap_start = max(z_start, e_event.start)
                overlap_end = min(z_end, e_event.end)
                overlap = overlap_end - overlap_start
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match_en = e_event.text.strip()
                if e_event.start > z_end:
                    break
            
            if best_match_en:
                final_text = f"{z_text}\\N{{\\r{style_sec_cfg['Name']}}}{best_match_en}"
            else:
                final_text = z_text

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