from custom_tool import (
    QuerySQLCheckerTool,
    InfoSQLDatabaseTool,
    QuerySQLDataBaseTool,
    ListSQLDatabaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from textwrap import dedent
from dotenv import load_dotenv

import os

load_dotenv()

llm = LLM(
    model="groq/llama-3.1-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)


db = SQLDatabase.from_uri("sqlite:///salaries.db")


@tool("List tables")
def list_table() -> str:
    """List the available tables in the database"""
    tool = ListSQLDatabaseTool(db=db)
    return tool.invoke("")


@tool("Tables schemas")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)


@tool("Execute SQL")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    tool = QuerySQLDataBaseTool(db=db)
    return tool.invoke(sql_query)


@tool("Check SQL query")
def check_sql_query(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    tool = QuerySQLCheckerTool(db=db, llm=llm)
    return tool.run({"query": sql_query})


print(check_sql_query.run("SELECT * FROM salaries WHERE salary > 10000 LIMIT 5"))

# Agents
sql_developer = Agent(
    role="Senior Database Developer",
    goal="Construct and execute SQL queries based on a request",
    backstory=dedent(
        """
        You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        Use the `list_tables` to find available tables.
        Use the `tables_schema` to understand the metadata for the tables.
        Use the `execute_sql` to check your queries for correctness.
        Use the `check_sql` to execute queries against the database.
        """
    ),
    llm=llm,
    tools=[list_table, tables_schema, execute_sql, check_sql_query],
    allow_delegation=False,
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal="You receive data from the database developer and analyze it",
    backstory=dedent(
        """
        You have deep experience with analyzing datasets using Python.
        Your work is always based on the provided data and is clear,
        easy-to-understand and to the point. You have attention
        to detail and always produce very detailed work (as long as you need).
    """
    ),
    llm=llm,
    allow_delegation=False,
)

report_writer = Agent(
    role="Senior Report Editor",
    goal="Write an executive summary type of report based on the work of the analyst",
    backstory=dedent(
        """
        Your writing still is well known for clear and effective communication.
        You always summarize long texts into bullet points that contain the most
        important details.
        """
    ),
    llm=llm,
    allow_delegation=False,
)

# Tasks
extract_data = Task(
    description="Extract data that is required for the query {query}.",
    expected_output="Database result for the query",
    agent=sql_developer,
)


analyze_data = Task(
    description="Analyze the data from the database and write an analysis for {query}.",
    expected_output="Detailed analysis text",
    agent=data_analyst,
    context=[extract_data],
)


write_report = Task(
    description=dedent(
        """
        Write an executive summary of the report from the analysis. The report
        must be less than 100 words.
    """
    ),
    expected_output="Markdown report",
    agent=report_writer,
    context=[analyze_data],
)

# Crew
crew = Crew(
    agents=[sql_developer, data_analyst, report_writer],
    tasks=[extract_data, analyze_data, write_report],
    process=Process.sequential,
    verbose=True,
    memory=False,
    output_log_file="crew.log",
)


inputs = {
    "query": "Analyze correlation between salary and experience",
}

result = crew.kickoff(inputs=inputs)

print(result)
