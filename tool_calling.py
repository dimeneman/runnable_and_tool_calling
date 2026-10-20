from dotenv import load_dotenv

from rich import print
from rich.rule import Rule

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# -------------------------------------------------
# Tool
# -------------------------------------------------

@tool
def word_counter(text: str) -> str:
    """
    Count the number of words in the given text.
    """
    word_count = len(text.split())
    return f"The text contains {word_count} words."


# -------------------------------------------------
# LLM
# -------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

llm_with_tools = llm.bind_tools([word_counter])

# -------------------------------------------------
# User Question
# -------------------------------------------------

question = (
    "How many words are in this text: "
    "Handling complex PDFs, scanned documents, tables, "
    "and maintaining structural context."
)

# -------------------------------------------------
# First LLM Call
# -------------------------------------------------

print(Rule("[bold cyan]First LLM Call[/bold cyan]"))

response = llm_with_tools.invoke(question)

print(response)
print()

# -------------------------------------------------
# Execute Tool
# -------------------------------------------------

tool_map = {
    "word_counter": word_counter
}

tool_messages = []

print(Rule("[bold green]Executing Tool[/bold green]"))

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print(f"[yellow]Tool Selected:[/yellow] {tool_name}")
    print(f"[yellow]Arguments:[/yellow] {tool_args}")

    tool = tool_map[tool_name]

    tool_result = tool.invoke(tool_args)

    print(f"[green]Tool Result:[/green] {tool_result}")

    tool_messages.append(
        ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"],
        )
    )

# -------------------------------------------------
# Second LLM Call
# -------------------------------------------------

print()
print(Rule("[bold magenta]Second LLM Call[/bold magenta]"))

final_response = llm_with_tools.invoke(
    [
        HumanMessage(content=question),
        response,
        *tool_messages,
    ]
)

print(final_response.content)