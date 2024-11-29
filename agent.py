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
    model="ollama/maxkb/chat2db-sql:7b-q8_0",
    temperature=0.1,
    base_url="http://localhost:11434",
)


db = SQLDatabase.from_uri("sqlite:///salaries.db")


@tool("List tables")
def list_table() -> str:
    """List the available tables in the database"""
    tool = ListSQLDatabaseTool(db=db)
    return tool.run("")


@tool("Tables schemas")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.run(tables)


@tool("Execute SQL")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    tool = QuerySQLDataBaseTool(db=db)
    return tool.run(sql_query)


@tool("Check SQL query")
def check_sql_query(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    tool = QuerySQLCheckerTool(db=db, llm=llm)
    return tool.run({"query": sql_query})

# Agents
sql_developer = Agent(
    role="Database Query Expert",
    goal="Generate accurate SQL queries based on available database schema",
    backstory=dedent(
        """
        You are an extremely methodical database engineer who:
        - ALWAYS first checks available tables
        - EXACTLY matches query to database structure
        - Uses ONLY confirmed table and column names
        - Executes queries if gives error verify and correct
        - Provides step-by-step reasoning
        """
    ),
    llm=llm,
    tools=[list_table, tables_schema,execute_sql,check_sql_query],
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
def tasks(inputs) -> Crew:
    discover_schema = Task(
        description="Discover and document the available database tables and their schemas",
        expected_output="Detailed list of tables and their structures",
        agent=sql_developer,
        tools=[list_table, tables_schema]
    )


    extract_data = Task(
        description=f"Generate and execute SQL query to answer: {inputs["query"]}",
        expected_output="Precise database query results",
        agent=sql_developer,
        context=[discover_schema],
        tools=[check_sql_query, execute_sql]
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

    return crew


inputs = {
    "query": "which country has maximum mean of salary?"
}
crew = tasks(inputs)
result = crew.kickoff(inputs=inputs)

print(result)
