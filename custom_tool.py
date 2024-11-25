from langchain_community.utilities.sql_database import SQLDatabase
from pydantic import BaseModel, Field, ConfigDict
from crewai.tools import BaseTool
from typing import Type, Union, Sequence, Dict, Any
from prompt import QUERY_CHECKER
from crewai import LLM
from sqlalchemy import Result


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QuerySQLDataBaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result..
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QuerySQLDataBaseToolInput

    def _run(self, query: str) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)


class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(
            [t.strip() for t in table_names.split(",")]
        )


class _ListSQLDataBaseToolInput(BaseModel):
    tool_input: str = Field("", description="An empty string")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting tables names."""

    name: str = "sql_db_list_tables"
    description: str = (
        "Input is an empty string, output is a comma-separated list of tables in the database."
    )
    args_schema: Type[BaseModel] = _ListSQLDataBaseToolInput

    def _run(self, tool_input: str = "") -> str:
        """Get a comma-separated list of table names."""
        return ", ".join(self.db.get_usable_table_names())


class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed and SQL query to be checked.")


class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: LLM
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    def _run(self, query: str) -> str:
        """Use the LLM to check the query."""
        prompt = self.template.format(query=query, dialect=self.db.dialect)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.call(messages)

    async def _arun(self, query: str) -> str:
        prompt = self.template.format(query=query, dialect=self.db.dialect)
        messages = [{"role": "user", "content": prompt}]
        return await self.llm.call(messages)
