import json
import os
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, Optional, Tuple, Any
from contextlib import AsyncExitStack
import asyncio
import nest_asyncio
from abc import ABC, abstractmethod
from enum import Enum
import httpx
import re

nest_asyncio.apply()
load_dotenv()

# --- All your AIService classes and the ServiceFactory go here ---
# (This entire section is unchanged from your existing, working script)
class ServiceType(Enum):
    OPENAI = "openai"; ANTHROPIC = "anthropic"; GOOGLE = "google"; COHERE = "cohere"
    OLLAMA = "ollama"; GROK = "grok"; MISTRAL = "mistral"; PERPLEXITY = "perplexity"
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
        if response and response.get('choices'): return response['choices'][0]['message'].get('content')
        return None
    def extract_tool_calls(self, response) -> List[Dict]:
        if not (response and response.get('choices')): return []
        tool_calls = response['choices'][0]['message'].get('tool_calls')
        if not tool_calls: return []
        return [{'id': tc['id'],'name': tc['function']['name'],'arguments': json.loads(tc['function']['arguments'])} for tc in tool_calls]
    def format_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> Dict:
        """
        Formats the tool response with extra context for the AI.
        """
        new_content = f"The tool '{tool_name}' returned the following output:\n---\n{content}\n---"
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": new_content
        }
    def format_assistant_message(self, content: str, tool_calls: List[Dict]) -> Dict:
        """
        Formats the assistant's message, ensuring content is never None.
        """
        final_content = content if content is not None else ""
        message = {'role': 'assistant', 'content': final_content}
        if tool_calls:
            formatted_tool_calls = [{'id': tc['id'],'type': 'function','function': {'name': tc['name'],'arguments': json.dumps(tc['arguments'])}} for tc in tool_calls]
            message['tool_calls'] = formatted_tool_calls
        return message

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
        }
        if service_type in service_map: return service_map[service_type](**kwargs)
        raise ValueError(f"Unsupported service type: {service_type}")

class MCP_ChatBot:
    # --- MODIFICATION 1: Updated __init__ to accept the model directly ---
    def __init__(self, service_type: ServiceType = ServiceType.OPENAI, model: str = "gpt-4o-mini"):
        self.exit_stack = AsyncExitStack()
        self.service_type = service_type
        # The factory now receives the specific model from the UI
        self.ai_service = ServiceFactory.create_service(service_type, model=model)
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_session: Dict[str, Tuple[ClientSession, types.Tool]] = {}
        self.available_prompts: Dict[str, types.Prompt] = {}
        self.static_resources: Dict[str, str] = {}
        self.dynamic_resources: List[str] = []
        self.resource_to_session: Dict[str, str] = {}
        self.prompt_to_session: Dict[str, str] = {}
        self.available_tools: List[Dict] = []
        print(f"Initialized Chatbot Engine with {self.service_type.value} ({model})")

    async def connect_to_servers(self):
        # This method is unchanged
        try:
            with open("server_config.json", "r") as file:
                servers = json.load(file).get("mcpServers", {})
            for name, config in servers.items():
                await self.connect_to_server(name, config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")

    async def connect_to_server(self, server_name: str, server_config: dict):
        # This method is unchanged
        try:
            server_params = StdioServerParameters(**server_config)
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[server_name] = session
            print(f"Connected to '{server_name}'.")
            try:
                resp = await session.list_tools()
                for tool in resp.tools: self.tool_to_session[tool.name] = (session, tool)
            except Exception: pass
            try:
                resp = await session.list_prompts()
                for prompt in resp.prompts: self.prompt_to_session[prompt.name] = server_name; self.available_prompts[prompt.name] = prompt
            except Exception: pass
            try:
                resp = await session.list_resources()
                for resource in resp.resources:
                    uri = resource.uri.encoded_string(); self.resource_to_session[uri] = server_name
                    if '{' in uri and '}' in uri:
                        if uri not in self.dynamic_resources: self.dynamic_resources.append(uri)
                    else:
                        user_shortcut = uri.split('//')[-1]; self.static_resources[user_shortcut] = uri
                        if resource.name and resource.name != user_shortcut: self.static_resources[resource.name] = uri
            except Exception: pass
        except Exception as e:
            print(f"Failed to connect to '{server_name}': {e}")
            
    async def _rebuild_tool_schemas(self):
        # This method is unchanged
        self.available_tools = []
        for _, (_, tool_object) in self.tool_to_session.items():
            self.available_tools.append(self.ai_service.convert_mcp_to_service_schema(tool_object))

    async def execute_mcp_tool(self, tool_name: str, tool_args: Dict) -> str:
        # This method is unchanged
        session, _ = self.tool_to_session[tool_name]
        result = await session.call_tool(tool_name, arguments=tool_args)
        if not result.content: return "Tool executed successfully."
        return "".join(str(item.text) for item in result.content) if isinstance(result.content, list) else str(result.content)

    # --- MODIFICATION 2: Removed chat_loop, _get_help_message, and process_query ---
    # --- MODIFICATION 3: Added the new 'invoke' method as the main entry point ---
    async def invoke(self, query: str, message_history: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        This is the main entry point for the Streamlit app.
        It orchestrates the entire query-response-tool-use cycle.
        """
        # The Streamlit app ALREADY adds the user's query to the history.
        # We just create a local copy. The line adding the user message is removed.
        messages = list(message_history)
        
        trace = [] # This will store the step-by-step execution for display

        max_iterations = 10
        for _ in range(max_iterations):
            await self._rebuild_tool_schemas()
            
            try:
                response = await self.ai_service.create_completion(messages=messages, tools=self.available_tools)
            except Exception as e:
                error_message = f"Error calling AI service: {e}"
                trace.append(f"❌ {error_message}")
                return "\n\n".join(trace), messages

            content = self.ai_service.extract_message_content(response)
            tool_calls = self.ai_service.extract_tool_calls(response)
            
            # The format_assistant_message fix prevents 'None' from being saved.
            assistant_message = self.ai_service.format_assistant_message(content, tool_calls)
            messages.append(assistant_message)
            
            if content:
                trace.append(content)

            if not tool_calls:
                break
            
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['arguments']
                trace.append(f"🛠️ **Tool Call:** `{tool_name}({json.dumps(tool_args)})`")
                
                try:
                    result_content = await self.execute_mcp_tool(tool_name, tool_args)
                    display_result = result_content[:1500] + "..." if len(result_content) > 1500 else result_content
                    trace.append(f"**Tool Result:**\n```\n{display_result}\n```")
                except Exception as e:
                    result_content = f"Error executing tool {tool_name}: {e}"
                    trace.append(f"❌ {result_content}")

                # The format_tool_response fix adds attribution.
                tool_response = self.ai_service.format_tool_response(tool_call['id'], tool_name, result_content)
                messages.append(tool_response)

        final_response = "\n\n".join(trace)
        return final_response, messages

    async def cleanup(self):
        # This method is unchanged
        await self.exit_stack.aclose()

# --- MODIFICATION 4: Removed the main() function and __main__ block ---
# This file is now a library to be imported by app.py, not an executable script.
