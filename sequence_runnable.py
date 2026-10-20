from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Rich console
console = Console()

console.print(Rule("[bold cyan]LangChain Demo[/bold cyan]"))

# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# 3. Output Parser
parser = StrOutputParser()

# Build the chain
chain = prompt | model | parser

# Input
topic = "Machine Learning"

console.print(
    Panel(
        topic,
        title="[bold green]Input[/bold green]",
        expand=False,
    )
)

# Invoke the chain
result = chain.invoke({"topic": topic})

console.print()

console.print(
    Panel(
        Markdown(result),
        title="[bold blue]LLM Response[/bold blue]",
        border_style="blue",
    )
)

console.print(Rule("[bold green]Done[/bold green]"))