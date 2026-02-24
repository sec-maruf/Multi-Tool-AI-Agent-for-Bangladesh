import sqlite3
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool

def create_db_tool(db_path, table_name, description, llm):
    """
    Returns a LangChain tool that answers questions about the given SQLite table.
    """

    # Get table schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    schema = cursor.fetchone()[0]
    conn.close()

    # Prompt to convert question to SQL
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert in converting natural language questions into SQL queries.
The database table '{table_name}' has the following schema:
{schema}

Given a question, output **only** the SQL query that answers it. Do not include any explanation or extra text."""),
        ("human", "{question}")
    ])

    sql_chain = prompt | llm | StrOutputParser()

    def tool_func(question: str) -> str:
        """Use this tool to answer questions about the data in the table."""
        try:
            # Generate SQL
            sql_query = sql_chain.invoke({"question": question})
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            # Execute query
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()

            if df.empty:
                return "No results found."

            return df.to_string(index=False)

        except Exception as e:
            return f"Error: {e}"

    # ✅ Create a tool with a clean name (no spaces)
    return StructuredTool.from_function(
        func=tool_func,
        name=f"{table_name}_db",
        description=description
    )