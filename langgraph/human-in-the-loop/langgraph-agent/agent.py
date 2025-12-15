from rich import print
import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

@tool('write_file')
def write_file_tool(file_path: str, content: str) -> str:
    """
    Write content to a file
    
    Args:
        file_path: The path to the file to write to
        content: The content to write to the file
    """

    return f"File written to {file_path} successfully"

@tool('execute_sql')
def execute_sql_tool(query: str) -> str:
    """
    Execute a SQL query
    
    Args:
        query: The SQL query to execute
    """
    
    return f"SQL {query} executed successfully"

@tool('read_data')
def read_data_tool(query: str, limit: int = 10) -> str:
    """
    Read data from a database
    
    Args:
        query: The SQL query to execute
        limit: The maximum number of rows to return
    """
    
    return f"{limit} rows of data read successfully by {query}"


agent = create_agent(
    model="gpt-4o",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware( 
            interrupt_on={
                "write_file": True,  # All decisions (approve, edit, reject) allowed
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # No editing allowed
                # Safe operation, no approval needed
                "read_data": False,
            },
            # Prefix for interrupt messages - combined with tool name and args to form the full message
            # e.g., "Tool execution pending approval: execute_sql with query='DELETE FROM...'"
            # Individual tools can override this by specifying a "description" in their interrupt config
            description_prefix="Tool execution pending approval",
        ),
    ],
    # Human-in-the-loop requires checkpointing to handle interrupts.
    # In production, use a persistent checkpointer like AsyncPostgresSaver.
    # checkpointer=InMemorySaver(),  
)