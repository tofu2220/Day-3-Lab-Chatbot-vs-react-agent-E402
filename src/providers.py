"""
🔌 MULTI-PROVIDER LLM ADAPTER (OpenAI, Gemini, Anthropic, OpenRouter & Offline Mock)
Hỗ trợ chuyển đổi linh hoạt giữa các nhà cung cấp AI chỉ bằng cách đổi biến môi trường LLM_PROVIDER.
"""

import os
import sys
import json
import re
import unicodedata
import requests
from dotenv import load_dotenv

# Đảm bảo in ra Tiếng Việt và Emojis không bị lỗi trên Windows Console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

class BaseLLMProvider:
    """Interface cơ sở cho tất cả các LLM Provider"""
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError


class GeminiProvider(BaseLLMProvider):
    """Google Gemini Provider"""
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model or os.getenv("LLM_MODEL") or "gemini-3.6-flash"
        
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return "[Gemini Error]: Chưa cấu hình GEMINI_API_KEY trong file .env!"
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            contents = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
            return response.text
        except Exception as e:
            return f"[Gemini Exception]: {str(e)}"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Provider (GPT-4o, GPT-3.5-turbo, etc.)"""
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            return "[OpenAI Error]: Chưa cấu hình OPENAI_API_KEY trong file .env!"
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI Exception]: {str(e)}"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider (Claude 3.5 Sonnet, Claude 3 Haiku)"""
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model or os.getenv("LLM_MODEL") or "claude-3-haiku-20240307"
        
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key or self.api_key == "your_anthropic_api_key_here":
            return "[Anthropic Error]: Chưa cấu hình ANTHROPIC_API_KEY trong file .env!"
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            kwargs = {
                "model": self.model_name,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system_prompt:
                kwargs["system"] = system_prompt
                
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            return f"[Anthropic Exception]: {str(e)}"


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter Provider (Hỗ trợ gọi mọi model qua OpenRouter API)"""
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model or os.getenv("LLM_MODEL") or "google/gemini-2.5-flash"
        
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            return "[OpenRouter Error]: Chưa cấu hình OPENROUTER_API_KEY trong file .env!"
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages
            }
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"[OpenRouter API Error {res.status_code}]: {res.text}"
        except Exception as e:
            return f"[OpenRouter Exception]: {str(e)}"


class MockProvider(BaseLLMProvider):
    """Offline deterministic provider for a runnable housing-agent demo.

    It emits the same ReAct JSON contract as a real provider, allowing the
    application to exercise its parser, tool registry, and guardrails without
    an API key.
    """
    @staticmethod
    def _fold(text: str) -> str:
        """Compare Vietnamese text without being sensitive to accents or case."""
        normalized = unicodedata.normalize("NFD", text.casefold())
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")

    def _baseline_response(self, user_query: str) -> str:
        text = self._fold(user_query)
        if any(greeting in text for greeting in ("xin chao", "hello", "hi", "chao ban")):
            return "Chào bạn! Tôi là chatbot tư vấn nhà thuê. Bạn đang quan tâm khu vực hoặc ngân sách nào?"
        if any(word in text for word in ("can ho", "phong tro", "tim nha", "dat lich", "thue")):
            return (
                "Tôi có thể tư vấn chung, nhưng Chatbot Baseline không được phép tra cứu dữ liệu căn và lịch trống. "
                "Hãy dùng chế độ ReAct (`python src/app.py --chat`) để tìm hoặc đặt lịch có xác minh."
            )
        return "Tôi sẵn sàng hỗ trợ. Bạn có thể cho biết khu vực, ngân sách hoặc loại nhà bạn cần không?"

    def _extract_location(self, user_query: str) -> str:
        """Map common location spellings (including 'quân') to data locations."""
        folded_query = self._fold(user_query)
        locations = {
            "quan 7": "Quận 7",
            "thu duc": "Thủ Đức",
            "quan 10": "Quận 10",
            "tan phu": "Tân Phú",
            "bach khoa": "Bách Khoa",
            # Included so the tool can truthfully respond that no data exists there.
            "my dinh": "Mỹ Đình",
        }
        for pattern, location in locations.items():
            if pattern in folded_query:
                return location
        return ""

    @staticmethod
    def _extract_budget(user_query: str) -> int | None:
        """Extract phrases such as 'dưới 5 triệu' or 'tối đa 4.5tr'."""
        match = re.search(
            r"(?:dưới|tối đa|không quá|≤)\s*(\d+(?:[.,]\d+)?)\s*(?:triệu|tr|m)\b",
            user_query.casefold(),
        )
        if not match:
            return None
        return int(float(match.group(1).replace(",", ".")) * 1_000_000)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if "Câu hỏi người dùng:" not in prompt:
            return self._baseline_response(prompt)

        query_match = re.search(
            r"Câu hỏi người dùng:\s*(.*?)\s*\n\nLịch sử hội thoại:", prompt, flags=re.DOTALL
        )
        user_query = query_match.group(1).strip() if query_match else prompt
        folded_query = self._fold(user_query)
        if "chua co observation" not in self._fold(prompt):
            observation_match = re.search(
                r"Observation:\s*(.*?)\s*\n\nHãy trả lời bước tiếp theo", prompt, flags=re.DOTALL
            )
            observation = observation_match.group(1).strip() if observation_match else "Không nhận được kết quả tool."
            return f"Thought: Đã nhận được kết quả từ công cụ.\nFinal Answer: {observation}"

        if any(phrase in folded_query for phrase in ("bo qua", "khong can goi cong cu", "khong_ton_tai")):
            return (
                "Thought: Yêu cầu không đáp ứng điều kiện an toàn hoặc thiếu dữ liệu xác minh.\n"
                "Final Answer: Tôi không thể xác nhận hay tạo lịch khi chưa xác minh căn, lịch trống và xác nhận rõ ràng của bạn."
            )
        if any(code in folded_query for code in ("ch001", "ch002", "pt003", "pt004")):
            property_id = next(code.upper() for code in ("ch001", "ch002", "pt003", "pt004") if code in folded_query)
            return (
                "Thought: Cần tra cứu dữ liệu căn trước khi trả lời.\n"
                f'Action: {{"tool": "get_property_details", "args": {{"property_id": "{property_id}"}}}}'
            )

        location = self._extract_location(user_query)
        budget = self._extract_budget(user_query)
        is_property_search = any(
            word in folded_query for word in ("tim", "can ho", "phong tro", "nha", "thue")
        )
        if location or budget is not None or is_property_search:
            arguments = {key: value for key, value in (("location", location), ("max_price", budget)) if value not in ("", None)}
            if "phong tro" in folded_query:
                arguments["property_type"] = "phòng trọ"
            elif "can ho" in folded_query:
                arguments["property_type"] = "căn hộ"
            if "may lanh" in folded_query:
                arguments["amenity"] = "máy lạnh"
            return (
                "Thought: Cần tra cứu dữ liệu căn theo các tiêu chí người dùng cung cấp.\n"
                f"Action: {{\"tool\": \"search_properties\", \"args\": {json.dumps(arguments, ensure_ascii=False)}}}"
            )
        return "Thought: Cần thêm tiêu chí để tra cứu.\nFinal Answer: Bạn cho tôi biết khu vực, ngân sách hoặc mã căn cần xem nhé."


def get_llm_provider(provider_name: str = None) -> BaseLLMProvider:
    """Factory function tự chọn Provider từ biến môi trường LLM_PROVIDER"""
    name = (provider_name or os.getenv("LLM_PROVIDER") or "mock").lower().strip()
    
    if name == "gemini":
        return GeminiProvider()
    elif name == "openai":
        return OpenAIProvider()
    elif name == "anthropic":
        return AnthropicProvider()
    elif name == "openrouter":
        return OpenRouterProvider()
    else:
        return MockProvider()


if __name__ == "__main__":
    print("=== TEST MULTI-PROVIDER LLM ADAPTER ===")
    provider = get_llm_provider()
    print(f"✅ Provider đang dùng: {provider.__class__.__name__}")
    print(f"🤖 User Query: Hello")
    print(f"💬 Response  : {provider.generate('Hello')}")
