from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Rich Console
console = Console()

console.print(Rule("[bold cyan]RunnablePassthrough Demo[/bold cyan]"))

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Output Parser
parser = StrOutputParser()

# Prompt to generate code
code_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a code generator"),
        ("human", "{topic}")
    ]
)

# Prompt to explain code
explain_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who explains code in simple terms",
        ),
        (
            "human",
            "Explain the following code in simple words:\n{code}",
        ),
    ]
)

# Generate Code
seq = code_prompt | model | parser

# Generate Code + Explanation
seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explanation": explain_prompt | model | parser,
    }
)

# Complete Chain
chain = seq | seq2

# Input
user_input = {
    "topic": "Please write a code of palindrome in python."
}

console.print(
    Panel.fit(
        user_input["topic"],
        title="[bold green]User Prompt[/bold green]",
        border_style="green",
    )
)

console.print("\n[bold yellow]Generating Code and Explanation...[/bold yellow]\n")

# Invoke
result = chain.invoke(user_input)

# Code Panel
console.print(
    Panel(
        Syntax(
            result["code"],
            "python",
            line_numbers=True,
            word_wrap=True,
        ),
        title="[bold cyan]Generated Python Code[/bold cyan]",
        border_style="cyan",
    )
)

console.print()

# Explanation Panel
console.print(
    Panel(
        Markdown(result["explanation"]),
        title="[bold magenta]Code Explanation[/bold magenta]",
        border_style="magenta",
    )
)

console.print(Rule("[bold green]Execution Completed[/bold green]"))