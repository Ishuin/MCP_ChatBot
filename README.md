# MCP Chatbot with Multiple AI Services

A comprehensive chatbot implementation that integrates with Model Context Protocol (MCP) servers and supports multiple AI service providers including OpenAI, Anthropic, Google, Cohere, Ollama, Grok, Mistral, and Perplexity.

## Features

- **Multiple AI Service Support**: Seamlessly switch between different AI providers
- **MCP Integration**: Connect to multiple MCP servers for extended functionality
- **Tool Management**: Automatic tool discovery and schema conversion
- **Interactive Chat**: Real-time conversation with tool calling capabilities
- **Service Switching**: Dynamic switching between AI services during conversations
- **Extensible Architecture**: Easy to add new AI service providers

## Supported AI Services

| Service | Model Examples | Function Calling | Status |
|---------|---------------|------------------|--------|
| OpenAI | gpt-4o-mini, gpt-4o, gpt-3.5-turbo | ✅ | Stable |
| Anthropic | claude-3-5-sonnet-20241022, claude-3-haiku | ✅ | Stable |
| Google | gemini-1.5-flash, gemini-1.5-pro | ✅ | Stable |
| Cohere | command-r-plus, command-r | ✅ | Stable |
| Ollama | llama3, mistral, codellama | ✅ | Stable |
| Grok | grok-beta | ✅ | Beta |
| Mistral | mistral-large, mistral-medium | ✅ | Stable |
| Perplexity | llama-3.1-sonar-large-128k-online | ❌ | Limited |

## Installation

### Prerequisites

```bash
pip install openai anthropic mcp httpx asyncio nest-asyncio python-dotenv requests
```

### Environment Variables

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
COHERE_API_KEY=...
GROK_API_KEY=xai-...
MISTRAL_API_KEY=...
PERPLEXITY_API_KEY=...
```

### Server Configuration

Create a `server_config.json` file to define your MCP servers:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "/path/to/database.db"],
      "env": {}
    }
  }
}
```

## Usage

### Basic Usage

```python
from mcp_chatbot import MCP_ChatBot, ServiceType

# Initialize with OpenAI
chatbot = MCP_ChatBot(
    service_type=ServiceType.OPENAI,
    model="gpt-4o-mini"
)

# Connect to MCP servers and start chat
await chatbot.connect_to_servers()
await chatbot.chat_loop()
```

### Service-Specific Initialization

#### OpenAI
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.OPENAI,
    model="gpt-4o-mini",
    api_key="your-api-key"  # Optional if set in .env
)
```

#### Anthropic
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.ANTHROPIC,
    model="claude-3-5-sonnet-20241022"
)
```

#### Google
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.GOOGLE,
    model="gemini-1.5-flash"
)
```

#### Cohere
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.COHERE,
    model="command-r-plus"
)
```

#### Ollama
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.OLLAMA,
    model="llama3",
    base_url="http://localhost:11434"
)
```

#### Grok
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.GROK,
    model="grok-beta"
)
```

#### Mistral
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.MISTRAL,
    model="mistral-large"
)
```

#### Perplexity
```python
chatbot = MCP_ChatBot(
    service_type=ServiceType.PERPLEXITY,
    model="llama-3.1-sonar-large-128k-online"
)
```

### Interactive Commands

During the chat session, you can use these commands:

- `quit` - Exit the chatbot
- `switch <service>` - Switch to a different AI service
- `list services` - Show all available AI services

Example:
```
Query: switch anthropic
Switched to anthropic service

Query: list services
Available services:
  - openai
  - anthropic
  - google
  - cohere
  - ollama
  - grok
  - mistral
  - perplexity
```

## Testing and Development

### Testing Commands

For testing the current implementation, install the MCP server fetch package:

```bash
pip install mcp-server-fetch
```

### MCP Inspector

To run an MCP inspector for debugging and testing MCP server connections:

```bash
npx @modelcontextprotocol/inspector python3 archive_mcp.py
```

The MCP inspector provides a visual interface to:
- Test MCP server connections
- Inspect available tools and resources
- Debug tool schemas and responses
- Validate server configurations

## Architecture

### Class Hierarchy

```
AIService (Abstract Base Class)
├── OpenAIService
├── AnthropicService
├── GoogleService
├── CohereService
├── OllamaService
├── GrokService
├── MistralService
└── PerplexityService
```

### Key Components

1. **ServiceFactory**: Creates AI service instances based on service type
2. **MCP_ChatBot**: Main orchestrator that manages MCP connections and chat flow
3. **AIService**: Abstract base class defining the interface for all AI services
4. **Service Classes**: Concrete implementations for each AI provider

### Tool Schema Conversion

Each AI service implements its own tool schema conversion from MCP format:

- **OpenAI**: Uses `tools` array with `function` objects
- **Anthropic**: Uses `tools` array with `input_schema` objects
- **Google**: Uses `function_declarations` format
- **Cohere**: Uses `parameter_definitions` format
- **Others**: Follow OpenAI-compatible format

## Examples

### Example 1: File System Operations

```python
# server_config.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"],
      "env": {}
    }
  }
}

# Usage
Query: List all Python files in the current directory
# The chatbot will use the filesystem tool to list files
```

### Example 2: GitHub Integration

```python
# server_config.json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}

# Usage
Query: Show me the latest issues in my repository
# The chatbot will use GitHub API through MCP
```

### Example 3: Database Queries

```python
# server_config.json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "/path/to/database.db"],
      "env": {}
    }
  }
}

# Usage
Query: What are the top 10 customers by sales?
# The chatbot will query the SQLite database
```

## Adding New AI Services

To add a new AI service provider:

1. **Create Service Class**: Inherit from `AIService`
2. **Implement Abstract Methods**: All required methods must be implemented
3. **Add to ServiceType Enum**: Add the new service type
4. **Update ServiceFactory**: Add creation logic for the new service
5. **Test Integration**: Verify tool calling and message formatting

### Template for New Service

```python
class NewAIService(AIService):
    def __init__(self, model: str = "default-model", api_key: str = None):
        self.api_key = api_key or os.getenv("NEW_AI_API_KEY")
        self.model = model
        self.base_url = "https://api.newai.com/v1"
    
    def convert_mcp_to_service_schema(self, mcp_tool) -> Dict:
        # Convert MCP tool schema to service-specific format
        return {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema
        }
    
    async def create_completion(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict:
        # Make API call to the service
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "max_tokens": kwargs.get('max_tokens', 2024)
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            )
            return response.json()
    
    def extract_message_content(self, response: Dict) -> Optional[str]:
        # Extract text content from response
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    def extract_tool_calls(self, response: Dict) -> List[Dict]:
        # Extract tool calls from response
        tool_calls = []
        # Implementation depends on service response format
        return tool_calls
    
    def format_tool_response(self, tool_call_id: str, content: str) -> Dict:
        # Format tool response for the service
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        }
    
    def format_assistant_message(self, content: str, tool_calls) -> Dict:
        # Format assistant message for the service
        return {
            'role': 'assistant',
            'content': content,
            'tool_calls': tool_calls
        }
```

### Steps to Add New Service

1. **Add to ServiceType enum**:
```python
class ServiceType(Enum):
    # ... existing services
    NEW_AI = "new_ai"
```

2. **Update ServiceFactory**:
```python
@staticmethod
def create_service(service_type: ServiceType, **kwargs) -> AIService:
    # ... existing services
    elif service_type == ServiceType.NEW_AI:
        return NewAIService(**kwargs)
```

3. **Add environment variable**: Add `NEW_AI_API_KEY` to your `.env` file

4. **Test the integration**: Run the chatbot and switch to the new service

## Error Handling

The chatbot includes comprehensive error handling:

- **API Errors**: Graceful handling of API failures with user-friendly messages
- **Tool Execution Errors**: Robust error reporting when MCP tools fail
- **Service Switch Errors**: Safe fallback when switching between services
- **Connection Errors**: Retry logic for network issues

## Performance Considerations

- **Async Operations**: All API calls are asynchronous for better performance
- **Connection Pooling**: HTTP clients use connection pooling
- **Tool Caching**: Tool schemas are cached to avoid repeated conversions
- **Rate Limiting**: Built-in respect for API rate limits

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure all required API keys are set in `.env`
2. **MCP Server Connection Failed**: Check MCP server configuration and paths
3. **Tool Not Found**: Verify MCP server is running and tools are available
4. **Service Switch Failed**: Check if the target service is properly configured

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export DEBUG=1
```

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new services
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### v2.0.0
- Added support for Google, Cohere, Ollama, Grok, Mistral, and Perplexity
- Improved service switching mechanism
- Enhanced error handling and debugging
- Added comprehensive documentation

### v1.0.0
- Initial release with OpenAI and Anthropic support
- Basic MCP integration
- Interactive chat functionality