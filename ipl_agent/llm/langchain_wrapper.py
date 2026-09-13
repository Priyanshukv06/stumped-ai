"""LangChain-compatible ChatModel backed by the multi-provider router.

This is the bridge between LangChain's agent framework (create_react_agent,
which needs a BaseChatModel with tool-calling support) and the async
RoutedLLMBackend (which speaks OpenAI-compatible API across 5 providers).

The wrapper:
  1. Converts LangChain BaseMessage objects → OpenAI dict format
  2. Forwards tools (from bind_tools) through to providers
  3. Runs the async router synchronously (via nest_asyncio)
  4. Converts the OpenAI response back to LangChain AIMessage with tool_calls
"""

import asyncio
import json
import logging
from typing import Any, List, Optional

import nest_asyncio
from pydantic import PrivateAttr

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.outputs import ChatGeneration, ChatResult

from .router import RoutedLLMBackend, strip_reasoning

# Allow nested event loops (needed for Streamlit + sync LangChain)
nest_asyncio.apply()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message format converters
# ---------------------------------------------------------------------------

def _langchain_to_openai(messages: List[BaseMessage]) -> list[dict]:
    """Convert LangChain message objects to OpenAI-format dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})

        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})

        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            # Handle tool calls in the message history
            if msg.tool_calls:
                entry["content"] = msg.content if msg.content else None
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            else:
                entry["content"] = msg.content or ""
            result.append(entry)

        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                "tool_call_id": msg.tool_call_id
            })

        else:
            # Fallback for any unknown message type
            result.append({"role": "user", "content": str(msg.content)})

    return result


def _openai_msg_to_langchain(msg) -> AIMessage:
    """Convert an OpenAI ChatCompletionMessage to a LangChain AIMessage."""
    content = msg.content or ""
    # Strip <think> blocks from reasoning models
    content = strip_reasoning(content)

    tool_calls = []
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append({
                "name": tc.function.name,
                "args": args,
                "id": tc.id,
                "type": "tool_call",
            })

    return AIMessage(content=content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# LangChain wrapper
# ---------------------------------------------------------------------------

class RoutedChatModel(BaseChatModel):
    """LangChain BaseChatModel that routes through 5 LLM providers.

    Usage:
        llm = RoutedChatModel()
        # Use like any LangChain chat model:
        response = llm.invoke([HumanMessage(content="Hello")])
        # With tool calling:
        agent = create_react_agent(llm, tools=[...], prompt="...")
    """

    temperature: float = 0.3
    max_tokens: int = 4096

    # Private: the async router instance
    _router: RoutedLLMBackend = PrivateAttr()

    def __init__(self, temperature: float = 0.3, max_tokens: int = 4096, **kwargs):
        super().__init__(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self._router = RoutedLLMBackend()

        # Log initialization summary
        configured = self._router.configured_providers()
        chain = self._router.chain()
        logger.info(
            "RoutedChatModel initialized — %d providers: %s",
            len(configured), ", ".join(configured) if configured else "(none)"
        )
        if chain:
            logger.info(
                "Model chain (%d models): %s",
                len(chain),
                " → ".join(m.name for m in chain[:5]) + (" ..." if len(chain) > 5 else "")
            )

    @property
    def _llm_type(self) -> str:
        return "routed-multi-provider"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Bind tools for use with create_react_agent.

        Converts LangChain tool objects to OpenAI-format tool dicts
        and returns a bound runnable that passes them through to _generate.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        bind_kwargs = {"tools": formatted_tools, **kwargs}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Core generation — converts messages, calls router, returns ChatResult."""

        # 1. Convert LangChain messages to OpenAI format
        openai_messages = _langchain_to_openai(messages)

        # 2. Extract tools if bound via bind_tools()
        tools = kwargs.get("tools")

        # 3. Run the async router synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response_msg = loop.run_until_complete(
            self._router.route_chat(
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
            )
        )

        # 4. Convert OpenAI response to LangChain AIMessage
        ai_message = _openai_msg_to_langchain(response_msg)

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)]
        )
