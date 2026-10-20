from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Rich Console
console = Console()

console.print(Rule("[bold cyan]RunnableParallel Demo[/bold cyan]"))

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Output Parser
parser = StrOutputParser()

# Prompt Templates
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# Runnable Parallel
chain = RunnableParallel(
    {
        "short": (
            RunnableLambda(lambda x: x["short"])
            | short_prompt
            | model
            | parser
        ),
        "detailed": (
            RunnableLambda(lambda x: x["detailed"])
            | detailed_prompt
            | model
            | parser
        ),
    }
)

# Input
inputs = {
    "short": {"topic": "Machine Learning"},
    "detailed": {"topic": "Deep Learning"},
}

console.print(
    Panel.fit(
        str(inputs),
        title="[bold green]Input[/bold green]",
        border_style="green",
    )
)

console.print("\n[bold yellow]Running both branches in parallel...[/bold yellow]\n")

# Execute
result = chain.invoke(inputs)

# Output
console.print(
    Panel(
        Markdown(result["short"]),
        title="[bold blue]Short Explanation[/bold blue]",
        border_style="blue",
    )
)

console.print()

console.print(
    Panel(
        Markdown(result["detailed"]),
        title="[bold magenta]Detailed Explanation[/bold magenta]",
        border_style="magenta",
    )
)

console.print(Rule("[bold green]Execution Finished[/bold green]"))