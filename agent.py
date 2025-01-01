from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import logging
from typing import Optional
import os

class SQLQueryAgent:
    def __init__(self, db_uri: str, model_name: str = "maxkb/chat2db-sql:7b-q8_0"):
        load_dotenv()
        logging.basicConfig(level=logging.INFO)
        
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=0
            )
            self.db = SQLDatabase.from_uri(db_uri)
            self.agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_executor_kwargs={"handle_parsing_errors": True}  # Add parsing error handling
            )
        except Exception as e:
            logging.error(f"Failed to initialize SQL agent: {str(e)}")
            raise

    def query(self, question: str, max_retries: int = 3) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                response = self.agent.invoke(question)
                return response
            except Exception as e:
                logging.error(f"Query attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return None

def main():
    try:
        agent = SQLQueryAgent("sqlite:///salaries.db")
        response = agent.query("What is the average salary of each country?")
        if response:
            print(response)
        else:
            print("Failed to get response after all retries")
    except Exception as e:
        logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()