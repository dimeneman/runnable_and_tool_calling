from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.status import Status

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------
load_dotenv()

# Rich Console
console = Console()

console.print(Rule("[bold cyan]AI News Summarizer[/bold cyan]"))

# ----------------------------------------------------
# Tavily Search Tool
# ----------------------------------------------------
search_tool = TavilySearchResults(max_results=5)

# ----------------------------------------------------
# Prompt
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are an expert news editor.

Summarize the following news article clearly and objectively.

Instructions:
- Capture the most important facts.
- Make summary with 5 bullet points.
- Maintain a neutral tone.
- Do not add information that is not present in the article.
- Mention the key people, organizations, locations, and events if they are relevant.

News Article:
{article}
"""
)

# ----------------------------------------------------
# LLM
# ----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# ----------------------------------------------------
# Output Parser
# ----------------------------------------------------
parser = StrOutputParser()

# ----------------------------------------------------
# Chain
# ----------------------------------------------------
chain = prompt | llm | parser

# ----------------------------------------------------
# User Query
# ----------------------------------------------------
query = "Latest political news in India"

console.print(
    Panel.fit(
        query,
        title="[bold green]Search Query[/bold green]",
        border_style="green",
    )
)

# ----------------------------------------------------
# Search News
# ----------------------------------------------------
with console.status(
    "[bold yellow]Searching latest news with Tavily...",
    spinner="dots",
):
    news_result = search_tool.run(query)

console.print(
    "[bold green]✓[/bold green] News articles retrieved successfully.\n"
)

# ----------------------------------------------------
# Summarize
# ----------------------------------------------------
with console.status(
    "[bold yellow]Generating summary using Gemini...",
    spinner="dots",
):
    result = chain.invoke({"article": news_result})

# ----------------------------------------------------
# Output
# ----------------------------------------------------
console.print(
    Panel(
        Markdown(result),
        title="[bold cyan]News Summary[/bold cyan]",
        border_style="cyan",
    )
)

console.print(Rule("[bold green]Completed[/bold green]"))