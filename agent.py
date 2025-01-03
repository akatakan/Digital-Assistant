from sqlalchemy import create_engine, inspect, text
from smolagents import tool,CodeAgent,HfApiModel,LiteLLMModel
from dotenv import load_dotenv
import os

load_dotenv()


# Kullanıcıdan veritabanı bağlantısı al
db_url = "sqlite:///salaries.db"
# Bağlantıyı oluştur
try:
    engine = create_engine(db_url)
    with engine.connect() as con:
        print("Bağlantı başarıyla kuruldu!")
except Exception as e:
    print(f"Veritabanına bağlanırken bir hata oluştu: {e}")
    exit()

# SQL sorguları çalıştırmak için bir araç
@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the given database schema. Returns a string representation of the result.

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    try:
        with engine.connect() as con:
            rows = con.execute(text(query))
            for row in rows:
                output += "\n" + str(row)
    except Exception as e:
        output = f"Bir hata oluştu: {e}"
    return output

@tool
def list_tables_and_schemas() -> str:
    """
    Lists all tables and their schemas in the connected database.
    """
    output = "Veritabanındaki Tablolar ve Şemalar:\n"
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        for table in tables:
            output += f"\nTablo: {table}\n"
            columns = inspector.get_columns(table)
            for col in columns:
                output += f"  - {col['name']}: {col['type']}\n"
    except Exception as e:
        output = f"Bir hata oluştu: {e}"
    return output

agent = CodeAgent(
    tools=[sql_engine, list_tables_and_schemas],
    model=LiteLLMModel("groq/llama3-70b-8192",api_key=os.getenv("GROQ_API_KEY")),
)

agent.run("""
write top 5 average salaries for each company location.
""")
