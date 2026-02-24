import os
import re
import json
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import StructuredTool

from db_tools import create_db_tool

load_dotenv()

# ========== LLM Setup ==========
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    temperature=0,
    api_key=groq_api_key
)

# ========== DB Tools ==========
institutions_tool = create_db_tool(
    db_path="databases/institutions.db",
    table_name="institutions",
    description="Useful for questions about universities, colleges, government institutions in Bangladesh.",
    llm=llm
)

hospitals_tool = create_db_tool(
    db_path="databases/hospitals.db",
    table_name="hospitals",
    description="Useful for questions about hospitals, bed capacity, doctors, facilities in Bangladesh.",
    llm=llm
)

restaurants_tool = create_db_tool(
    db_path="databases/restaurants.db",
    table_name="restaurants",
    description="Useful for questions about restaurants, cuisine, ratings, locations in Bangladesh.",
    llm=llm
)

# ========== Web Search Tool with Error Handling ==========
def web_search_func(query: str) -> str:
    try:
        return DuckDuckGoSearchRun().run(query)
    except ImportError:
        return "Web search is unavailable: missing dependency 'ddgs'. Please install it with `pip install -U ddgs`."
    except Exception as e:
        return f"Web search failed: {e}"

web_search = StructuredTool.from_function(
    func=web_search_func,
    name="web_search",
    description="Search the web for general knowledge about Bangladesh. Use this for questions not answered by the local databases."
)

tools = [institutions_tool, hospitals_tool, restaurants_tool, web_search]
tool_map = {t.name: t for t in tools}

print("TOOLS:", [t.name for t in tools])

# ========== Agent ==========
system_prompt = (
    "You are a helpful AI assistant for Bangladesh. "
    "Use tools when needed. "
    "If you call a tool, use its result to produce a final human-readable answer."
)

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# ========== Tool‑call parser ==========
TOOL_CALL_RE = re.compile(r"<(?P<tool>\w+)>(?P<args>.*?)</\1>", re.DOTALL)

def extract_tool_call(text: str):
    if not text:
        return None, None
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None, None
    tool_name = m.group("tool").strip()
    args_str = m.group("args").strip()
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        args = {"question": args_str}
    return tool_name, args

def get_last_assistant_content(messages):
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if role in ("ai", "assistant") and content is not None:
            return str(content)
    return ""

# ========== Interactive Loop ==========
if __name__ == "__main__":
    print("Bangladesh AI Agent is ready! Type your question (or 'quit' to exit).")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break

        messages = [{"role": "user", "content": user_input}]

        for _ in range(6):
            result = agent.invoke({"messages": messages})
            new_messages = result.get("messages", [])
            last_text = get_last_assistant_content(new_messages)

            tool_name, tool_args = extract_tool_call(last_text)

            if tool_name and tool_name in tool_map:
                try:
                    tool_output = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_output = f"Tool error: {e}"

                messages.append({"role": "assistant", "content": last_text})
                messages.append({
                    "role": "user",
                    "content": f"Tool result (from {tool_name}): {tool_output}"
                })
                continue

            print(f"\nAgent: {last_text}")
            break
        else:
            print("\nAgent: (Stopped after too many tool calls. Something may be looping.)")