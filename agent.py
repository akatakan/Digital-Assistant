from langchain_community.tools.sql_database.tool import (
    QuerySQLCheckerTool,
    InfoSQLDatabaseTool,
    QuerySQLDataBaseTool,
    ListSQLDatabaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

GROQ_API_KEY = "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"

llm = LLM(model="groq/llama-3.1-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

db = SQLDatabase.from_uri("")


@tool("List tables")
def list_table():
    tool = ListSQLDatabaseTool(db=db)
    return tool.invoke()


@tool("Tables schemas")
def tables_schema(tables):
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)


@tool("Execute SQL")
def execute_sql(sql_query):
    tool = QuerySQLDataBaseTool(db=db)
    return tool.invoke(sql_query)


@tool("Check SQL query")
def check_sql_query(sql_query):
    tool = QuerySQLCheckerTool(db=db, llm=llm)
    return tool.invoke({"query": sql_query})


check_sql_query.run("SELECT * FROM sales")
