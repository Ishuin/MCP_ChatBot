import json
import os
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, Optional, Tuple
from contextlib import AsyncExitStack
import asyncio
import nest_asyncio
from abc import ABC, abstractmethod
from enum import Enum
import httpx
import re

nest_asyncio.apply()
load_dotenv()

class ServiceType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    OLLAMA = "ollama"
    GROK = "grok"
    MISTRAL = "mistral"
    PERPLEXITY = "perplexity"

class AIService(ABC):
    @abstractmethod
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict: pass
    @abstractmethod
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict: pass
    @abstractmethod
    def extract_message_content(self, response: Dict) -> Optional[str]: pass
    @abstractmethod
    def extract_tool_calls(self, response: Dict) -> List[Dict]: pass
    @abstractmethod
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict: pass
    @abstractmethod
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict: pass

# --- All Service Provider Classes ---
class OpenAIService(AIService):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        return {"type": "function","function": {"name": mcp_tool.name,"description": mcp_tool.description,"parameters": mcp_tool.inputSchema}}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        response = await self.client.chat.completions.create(model=self.model,messages=messages,tools=tools,max_tokens=kwargs.get('max_tokens', 4096))
        return response.model_dump()
    def extract_message_content(self, response) -> Optional[str]:
        if response and response.get('choices'):
            return response['choices'][0]['message'].get('content')
        return None
    def extract_tool_calls(self, response) -> List[Dict]:
        if not (response and response.get('choices')): return []
        tool_calls = response['choices'][0]['message'].get('tool_calls')
        if not tool_calls: return []
        return [{'id': tc['id'],'name': tc['function']['name'],'arguments': json.loads(tc['function']['arguments'])} for tc in tool_calls]
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        return {"role": "tool","tool_call_id": tool_call_id,"content": content}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        formatted_tool_calls = [{'id': tc['id'],'type': 'function','function': {'name': tc['name'],'arguments': json.dumps(tc['arguments'])}} for tc in tool_calls]
        return {'role': 'assistant','content': content,'tool_calls': formatted_tool_calls}

class AnthropicService(AIService):
    def __init__(self, model: str = "claude-3-5-sonnet-20240620", api_key: str = None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        return {"name": mcp_tool.name,"description": mcp_tool.description,"input_schema": mcp_tool.inputSchema}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        system_prompt = None
        if messages and messages[0]['role'] == 'system': system_prompt = messages.pop(0)['content']
        response = self.client.messages.create(model=self.model,system=system_prompt,messages=messages,tools=tools,max_tokens=kwargs.get('max_tokens', 4096))
        return response.model_dump()
    def extract_message_content(self, response) -> Optional[str]:
        for content_block in response.get('content', []):
            if content_block.get('type') == 'text': return content_block.get('text')
        return None
    def extract_tool_calls(self, response) -> List[Dict]:
        tool_calls = []
        for content_block in response.get('content', []):
            if content_block.get('type') == 'tool_use': tool_calls.append({'id': content_block['id'],'name': content_block['name'],'arguments': content_block['input']})
        return tool_calls
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        return {"role": "user","content": [{"type": "tool_result","tool_use_id": tool_call_id,"content": content}]}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        assistant_content = []
        if content: assistant_content.append({"type": "text", "text": content})
        for tool_call in tool_calls: assistant_content.append({"type": "tool_use","id": tool_call['id'],"name": tool_call['name'],"input": tool_call['arguments']})
        return {'role': 'assistant', 'content': assistant_content}

class GoogleService(AIService):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        return {"name": mcp_tool.name,"description": mcp_tool.description,"parameters": mcp_tool.inputSchema}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        google_contents = []
        for msg in messages:
            role = 'model' if msg['role'] == 'assistant' else msg['role']
            if role in ['user', 'model']:
                parts = []
                if msg.get('content'): parts.append({'text': msg['content']})
                if msg.get('tool_calls'):
                    for tc in msg['tool_calls']: parts.append({'functionCall': {'name': tc['name'], 'args': tc['arguments']}})
                google_contents.append({'role': role, 'parts': parts})
            elif role == 'tool':
                google_contents.append({'role': 'user','parts': [{'functionResponse': {'name': msg['name'],'response': {'content': msg['content']}}}]})
        payload = {"contents": google_contents,"tools": [{"function_declarations": tools}] if tools else None,"generationConfig": {"maxOutputTokens": kwargs.get('max_tokens', 8192)}}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",json=payload)
            response.raise_for_status()
            return response.json()
    def extract_message_content(self, response: Dict) -> Optional[str]:
        try:
            for part in response['candidates'][0]['content']['parts']:
                if 'text' in part: return part['text']
        except (KeyError, IndexError): return None
    def extract_tool_calls(self, response: Dict) -> List[Dict]:
        tool_calls = []
        try:
            for part in response['candidates'][0]['content']['parts']:
                if 'functionCall' in part:
                    fc = part['functionCall']
                    tool_calls.append({'id': fc['name'],'name': fc['name'],'arguments': fc.get('args', {})})
        except (KeyError, IndexError): return []
        return tool_calls
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        return {"role": "tool","name": tool_name,"content": content}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        return {'role': 'assistant','content': content,'tool_calls': tool_calls}

class CohereService(AIService):
    def __init__(self, model: str = "command-r-plus", api_key: str = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.base_url = "https://api.cohere.ai/v1"
        self.last_tool_calls = []
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        defs = {}
        if mcp_tool.inputSchema and 'properties' in mcp_tool.inputSchema:
            for name, prop in mcp_tool.inputSchema['properties'].items():
                defs[name] = {"description": prop.get("description", ""),"type": prop.get("type", "string"),"required": name in mcp_tool.inputSchema.get("required", [])}
        return {"name": mcp_tool.name,"description": mcp_tool.description,"parameter_definitions": defs}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        chat_history, tool_results = [], []
        if self.last_tool_calls:
            for msg in messages:
                if msg.get("role") == "tool_result":
                    call = next((c for c in self.last_tool_calls if c['name'] == msg['name']), None)
                    if call: tool_results.append({"call": call,"outputs": [msg['outputs']]})
                else: chat_history.append(msg)
        else: chat_history = messages
        for msg in chat_history:
            if msg['role'] == 'user': msg['role'] = 'USER'
            elif msg['role'] == 'assistant': msg['role'] = 'CHATBOT'
        last_message = chat_history.pop() if chat_history else {"content": ""}
        payload = {"model": self.model,"message": last_message.get('content', ''),"chat_history": chat_history,"tools": tools or None,"tool_results": tool_results or None}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/chat",headers={"Authorization": f"Bearer {self.api_key}"},json=payload)
            response.raise_for_status()
            return response.json()
    def extract_message_content(self, response: Dict) -> Optional[str]:
        return response.get('text', '')
    def extract_tool_calls(self, response: Dict) -> List[Dict]:
        tool_calls = response.get('tool_calls', [])
        if not tool_calls: return []
        self.last_tool_calls = tool_calls
        return [{'id': f"{tc['name']}_{int(time.time() * 1000)}",'name': tc['name'],'arguments': tc.get('parameters', {})} for tc in tool_calls]
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        try: parsed_content = json.loads(content)
        except (json.JSONDecodeError, TypeError): parsed_content = {"text": content}
        return {"role": "tool_result","name": tool_name,"outputs": parsed_content}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        return {'role': 'assistant','content': content}

class OllamaService(AIService):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model, self.base_url = model, base_url
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        return {"type": "function","function": {"name": mcp_tool.name,"description": mcp_tool.description,"parameters": mcp_tool.inputSchema}}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        payload = {"model": self.model,"messages": messages,"tools": tools,"stream": False}
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()
    def extract_message_content(self, response: Dict) -> Optional[str]:
        return response.get('message', {}).get('content', '')
    def extract_tool_calls(self, response: Dict) -> List[Dict]:
        tool_calls = []
        if 'tool_calls' in response.get('message', {}):
            for tc in response['message']['tool_calls']:
                tool_calls.append({'id': f"call_{int(time.time() * 1000)}",'name': tc['function']['name'],'arguments': json.loads(tc['function']['arguments'])})
        return tool_calls
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        return {"role": "tool","content": content}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        formatted_tool_calls = [{'function': {'name': tc['name'],'arguments': tc['arguments']}} for tc in tool_calls]
        return {'role': 'assistant','content': content,'tool_calls': formatted_tool_calls}

class GroqService(AIService):
    def __init__(self, model: str = "llama3-70b-8192", api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model, self.base_url = model, "https://api.groq.com/openai/v1"
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        return {"type": "function","function": {"name": mcp_tool.name,"description": mcp_tool.description,"parameters": mcp_tool.inputSchema}}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        payload = {"model": self.model,"messages": messages,"tools": tools,"max_tokens": kwargs.get('max_tokens', 8192)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/chat/completions",headers={"Authorization": f"Bearer {self.api_key}"},json=payload)
            response.raise_for_status()
            return response.json()
    def extract_message_content(self, response: Dict) -> Optional[str]:
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    def extract_tool_calls(self, response: Dict) -> List[Dict]:
        tool_calls = []
        if response.get('choices', []) and 'tool_calls' in response['choices'][0].get('message', {}) and response['choices'][0]['message']['tool_calls']:
            for tc in response['choices'][0]['message']['tool_calls']:
                tool_calls.append({'id': tc['id'],'name': tc['function']['name'],'arguments': json.loads(tc['function']['arguments'])})
        return tool_calls
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        return {"role": "tool","tool_call_id": tool_call_id,"content": content}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        formatted_tool_calls = [{'id': tc['id'],'type': 'function','function': {'name': tc['name'],'arguments': json.dumps(tc['arguments'])}} for tc in tool_calls]
        return {'role': 'assistant','content': content,'tool_calls': formatted_tool_calls}

class MistralService(AIService):
    def __init__(self, model: str = "mistral-large-latest", api_key: str = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model, self.base_url = model, "https://api.mistral.ai/v1"
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        return {"type": "function","function": {"name": mcp_tool.name,"description": mcp_tool.description,"parameters": mcp_tool.inputSchema}}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        payload = {"model": self.model,"messages": messages,"tools": tools,"max_tokens": kwargs.get('max_tokens', 4096)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/chat/completions",headers={"Authorization": f"Bearer {self.api_key}"},json=payload)
            response.raise_for_status()
            return response.json()
    def extract_message_content(self, response: Dict) -> Optional[str]:
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    def extract_tool_calls(self, response: Dict) -> List[Dict]:
        tool_calls = []
        if response.get('choices', []) and 'tool_calls' in response['choices'][0].get('message', {}) and response['choices'][0]['message']['tool_calls']:
            for tc in response['choices'][0]['message']['tool_calls']:
                tool_calls.append({'id': tc['id'],'name': tc['function']['name'],'arguments': json.loads(tc['function']['arguments'])})
        return tool_calls
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        return {"role": "tool","tool_call_id": tool_call_id,"content": content}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        formatted_tool_calls = [{'id': tc['id'],'type': 'function','function': {'name': tc['name'],'arguments': json.dumps(tc['arguments'])}} for tc in tool_calls]
        return {'role': 'assistant','content': content,'tool_calls': formatted_tool_calls}

class PerplexityService(AIService):
    def __init__(self, model: str = "llama-3.1-sonar-large-128k-online", api_key: str = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model, self.base_url = model, "https://api.perplexity.ai"
    def convert_mcp_to_service_schema(self, mcp_tool: types.Tool) -> Dict:
        return {"type": "function","function": {"name": mcp_tool.name,"description": mcp_tool.description,"parameters": mcp_tool.inputSchema}}
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        payload = {"model": self.model,"messages": messages,"max_tokens": kwargs.get('max_tokens', 4096)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/chat/completions",headers={"Authorization": f"Bearer {self.api_key}"},json=payload)
            response.raise_for_status()
            return response.json()
    def extract_message_content(self, response: Dict) -> Optional[str]:
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    def extract_tool_calls(self, response: Dict) -> List[Dict]: return []
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        return {"role": "tool","tool_call_id": tool_call_id,"content": content}
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        return {'role': 'assistant','content': content}

class ServiceFactory:
    @staticmethod
    def create_service(service_type: ServiceType, **kwargs) -> AIService:
        service_map = {
            ServiceType.OPENAI: OpenAIService,
            ServiceType.ANTHROPIC: AnthropicService,
            ServiceType.GOOGLE: GoogleService,
            ServiceType.COHERE: CohereService,
            ServiceType.OLLAMA: OllamaService,
            ServiceType.GROK: GroqService,
            ServiceType.MISTRAL: MistralService,
            ServiceType.PERPLEXITY: PerplexityService,
            # Add other services here if you expand the script
        }
        if service_type in service_map: return service_map[service_type](**kwargs)
        raise ValueError(f"Unsupported service type: {service_type}")

# --- MCP ChatBot with User-Friendly Command Handling ---
class MCP_ChatBot:
    def __init__(self, service_type: ServiceType = ServiceType.OPENAI, **service_kwargs):
        self.exit_stack = AsyncExitStack()
        self.service_type = service_type
        self.ai_service = ServiceFactory.create_service(service_type, **service_kwargs)
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_session: Dict[str, Tuple[ClientSession, types.Tool]] = {}
        self.available_prompts: Dict[str, types.Prompt] = {}
        self.static_resources: Dict[str, str] = {}
        self.dynamic_resources: List[str] = []
        self.resource_to_session: Dict[str, str] = {}
        self.available_tools: List[Dict] = []
        print(f"Initialized with {service_type.value} service")

    async def _rebuild_tool_schemas(self):
        self.available_tools = []
        for tool_name, (session, tool_object) in self.tool_to_session.items():
            service_schema = self.ai_service.convert_mcp_to_service_schema(tool_object)
            self.available_tools.append(service_schema)

    async def process_query(self, query: str, prompt_name: Optional[str] = None, resource_uri: Optional[str] = None):
        messages = []
        final_query = query

        if prompt_name:
            session_name = self.resource_to_session.get(prompt_name)
            if not session_name:
                print(f"Error: Could not find a session for prompt '{prompt_name}'.")
                return

            session = self.sessions[session_name]
            try:
                # 2. Ask the server to render the prompt with the query as the 'topic' argument.
                # This assumes the prompt's main argument is named 'topic'.
                prompt_response = await session.get_prompt(name=prompt_name, arguments={'topic': query})
                if prompt_response.messages:
                    prompt_content = prompt_response.messages[0].content.text
                    messages.append({'role': 'system', 'content': prompt_content})
                    print(f"Applied system prompt from '{prompt_name}'")
                else:
                    print(f"Warning: Prompt '{prompt_name}' did not return any content.")
            except Exception as e:
                print(f"Error getting prompt '{prompt_name}' from server: {e}")
                return

        if resource_uri:
            # Ensure the resource URI exists before proceeding
            if resource_uri not in self.resource_to_session:
                print(f"Error: The resource URI '{resource_uri}' is not associated with any active session.")
                return
                
            session_name = self.resource_to_session[resource_uri]
            session = self.sessions[session_name]
            try:
                # --- THIS IS THE FIX ---
                # Use the correct `read_resource` method with the full URI.
                response = await session.read_resource(uri=resource_uri)
                # --- END OF FIX ---

                content_obj = response.contents[0] if response.contents else None
                if not content_obj:
                    print(f"No content found for resource '{resource_uri}'.")
                    return
                resource_content = content_obj.text if hasattr(content_obj, 'text') else str(content_obj)
                user_friendly_name = resource_uri.split('//')[-1].replace('{', '<').replace('}', '>')
                final_query = f"Using context from resource '{user_friendly_name}':\n---\n{resource_content}\n---\n\nUser query: {query}"
                print(f"Injected context from resource: '{resource_uri}'")
            except Exception as e:
                print(f"Error fetching resource '{resource_uri}': {e}")
                return
        
        messages.append({'role': 'user', 'content': final_query})
        
        max_iterations = 10
        for _ in range(max_iterations):
            await self._rebuild_tool_schemas()
            completion_response = await self.ai_service.create_completion(messages=messages, tools=self.available_tools)
            content = self.ai_service.extract_message_content(completion_response)
            if content: print(f"\nAssistant: {content}")
            
            tool_calls = self.ai_service.extract_tool_calls(completion_response)
            if not tool_calls: break
            
            messages.append(self.ai_service.format_assistant_message(content, tool_calls))
            
            for tool_call in tool_calls:
                tool_id, tool_name, tool_args = tool_call['id'], tool_call['name'], tool_call['arguments']
                print(f"Calling tool `{tool_name}` with arguments: {tool_args}")
                result_content = await self.execute_mcp_tool(tool_name, tool_args)
                print(f"Tool `{tool_name}` result: {result_content}")
                messages.append(self.ai_service.format_tool_response(tool_id, tool_name, result_content))

    async def execute_mcp_tool(self, tool_name: str, tool_args: Dict) -> str:
        if tool_name not in self.tool_to_session:
            return f"Error: Tool '{tool_name}' not found."
        
        session, _ = self.tool_to_session[tool_name]
        try:
            result = await session.call_tool(tool_name, arguments=tool_args)
            if not result.content: return "Tool executed successfully."
            if isinstance(result.content, list):
                return "".join(str(item.text) if hasattr(item, 'text') else str(item) for item in result.content)
            return str(result.content)
        except Exception as e:
            return f"Error executing tool {tool_name}: {e}"

    def _get_help_message(self) -> str:
        help_text = "\n" + "="*50 + "\n"
        help_text += f"💬 MCP Chatbot ({self.service_type.value}) | Type 'quit' to exit.\n"
        help_text += "="*50 + "\n\n"
        help_text += "You can use the following commands and shortcuts:\n\n"
        if self.tool_to_session:
            help_text += f"🛠️  **Tools**: Just ask! The AI uses these automatically.\n    (Available: {', '.join(self.tool_to_session.keys())})\n\n"
        if self.available_prompts:
            help_text += f"📝 **Prompts**: Use `/prompt <name> <query>` to guide the AI.\n    (Available: {', '.join(self.available_prompts.keys())})\n\n"
        if self.static_resources or self.dynamic_resources:
            help_text += "📦 **Resources**: Use shortcuts to inject content into your query.\n"
            for shortcut in sorted(list(self.static_resources.keys())):
                help_text += f"    - Type `{shortcut}` or `@{shortcut}` to view the '{shortcut}' resource.\n"
            if self.dynamic_resources:
                # Make the example for dynamic resources clearer
                help_text += "    - Use `@<topic>` to view papers for a specific topic (e.g., `@algebra`).\n"
        help_text += "\n" + "-"*50
        return help_text

    async def chat_loop(self):
        print(self._get_help_message())
        
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                if not user_input: continue
                if user_input.lower() == 'quit': break
                
                parts = user_input.split()
                first_word = parts[0]
                prompt_name, resource_uri, query = None, None, user_input

                # --- FINAL, ROBUST PARSER ---
                is_command_handled = False

                # 1. Handle exclusive /prompt command
                if first_word == '/prompt' and len(parts) > 2:
                    prompt_name = parts[1]
                    if prompt_name not in self.available_prompts:
                        print(f"Error: Prompt '{prompt_name}' not found.")
                        continue
                    query = " ".join(parts[2:])
                    is_command_handled = True

                # 2. If not a prompt, check if it's a resource call
                if not is_command_handled:
                    target_shortcut = None
                    if first_word.startswith('@'):
                        target_shortcut = first_word[1:]
                    elif first_word in self.static_resources:
                        target_shortcut = first_word

                    if target_shortcut:
                        is_command_handled = True
                        query = " ".join(parts[1:]) if len(parts) > 1 else ""
                        
                        # Translate the shortcut into a full, technical URI
                        if target_shortcut in self.static_resources:
                            resource_uri = self.static_resources[target_shortcut]
                        else:
                            # --- THIS IS THE FIX ---
                            # It's not a static shortcut, so try to match it to a dynamic pattern.
                            found_pattern = False
                            for pattern in self.dynamic_resources:
                                if '{' in pattern:
                                    # Generate the specific URI (e.g., "papers://algebra")
                                    generated_uri = re.sub(r'{(.+?)}', target_shortcut, pattern)
                                    
                                    # Find the session that owns the original pattern
                                    session_for_pattern = self.resource_to_session.get(pattern)
                                    
                                    # If the pattern is valid, temporarily map the new, specific URI
                                    # to the correct session so process_query can find it.
                                    if session_for_pattern:
                                        self.resource_to_session[generated_uri] = session_for_pattern
                                        resource_uri = generated_uri
                                        found_pattern = True
                                        break
                            
                            if not found_pattern:
                                 print(f"Error: Resource shortcut '@{target_shortcut}' is not recognized.")
                                 continue
                
                # If a command was used but no query was provided, use a default.
                if is_command_handled and not query:
                    query = "Please describe this."

                await self.process_query(query, prompt_name=prompt_name, resource_uri=resource_uri)
                
            except Exception as e:
                print(f"\nAn error occurred: {e}")

    async def connect_to_server(self, server_name: str, server_config: dict):
        try:
            server_params = StdioServerParameters(**server_config)
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[server_name] = session
            print(f"Connected to '{server_name}'. Discovering assets...")

            # Graceful discovery for tools and prompts (no change here)
            try:
                resp = await session.list_tools()
                if resp.tools: print(f"  - Found Tools: {[t.name for t in resp.tools]}")
                for tool in resp.tools: self.tool_to_session[tool.name] = (session, tool)
            except Exception: pass
            
            try:
                resp = await session.list_prompts()
                if resp.prompts: print(f"  - Found Prompts: {[p.name for p in resp.prompts]}")
                for prompt in resp.prompts:
                    # Store the prompt object itself
                    self.available_prompts[prompt.name] = prompt 
                    # Store which session owns the prompt
                    self.resource_to_session[prompt.name] = server_name # Using resource_to_session for simplicity
            except Exception: pass

            # --- THIS IS THE DEFINITIVE FIX FOR RESOURCE DISCOVERY ---
            
            # 1. Discover STATIC resources
            try:
                resp = await session.list_resources()
                if resp.resources: print(f"  - Found Static Resources: {[r.name for r in resp.resources]}")
                for resource in resp.resources:
                    uri = resource.uri.encoded_string()
                    self.resource_to_session[uri] = server_name
                    user_shortcut = uri.split('//')[-1]
                    self.static_resources[user_shortcut] = uri
                    if resource.name and resource.name != user_shortcut:
                        self.static_resources[resource.name] = uri
            except Exception:
                # This is okay, some servers don't have static resources.
                pass

            # 2. Discover DYNAMIC resource templates
            try:
                # The response object for templates has a `.templates` attribute
                resp = await session.list_resource_templates()
                if resp.resourceTemplates: print(f"  - Found Dynamic Resource Templates: {[t.name for t in resp.resourceTemplates]}")
                for template in resp.resourceTemplates:
                    uri = template.uriTemplate
                    # The key for the session map MUST be the pattern itself.
                    self.resource_to_session[uri] = server_name
                    if uri not in self.dynamic_resources: 
                        self.dynamic_resources.append(uri)
            except Exception:
                # This is also okay, some servers don't have dynamic resources.
                pass
            
            # --- END OF FIX ---

        except Exception as e:
            print(f"Failed to connect to '{server_name}': {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                servers = json.load(file).get("mcpServers", {})
            if not servers:
                print("Warning: No MCP servers found in server_config.json.")
                return
            for name, config in servers.items():
                await self.connect_to_server(name, config)
        except FileNotFoundError:
            print("Error: server_config.json not found. Chatbot will run without tools.")
        except Exception as e:
            print(f"Error loading server configuration: {e}")

    async def cleanup(self):
        print("\nCleaning up and shutting down...")
        await self.exit_stack.aclose()

async def main():
    chatbot = MCP_ChatBot(service_type=ServiceType.OPENAI)
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")