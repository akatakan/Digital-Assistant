import sqlite3 as sql

conn = sql.connect('salaries.db')
cur = conn.cursor()

cur.execute("SELECT company_location, AVG(salary_in_usd) AS average_salary FROM salaries GROUP BY company_location ORDER BY average_salary DESC LIMIT 5")
rows = cur.fetchall()
for row in rows:
    print(row)