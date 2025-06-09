# Databricks notebook source
# MAGIC %pip install langchain langchain-community databricks-sql-connector langchain-openai tabulate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# 📦 Minimal & Improved Databricks SQL Agent with Chat Loop + Donation Follow-up

import re
import random
import pandas as pd
from databricks import sql
from langchain_community.chat_models import ChatDatabricks
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

# 🔐 Credentials
DATABRICKS_TOKEN = ""
DATABRICKS_HOST = "dbc-32826a1f-a515.cloud.databricks.com"
WAREHOUSE_PATH = "/sql/1.0/warehouses/031f2b46b9c894c1"

# 📘 Tables to describe
tables = [
    "Hackathon_Team_Winners.homeless_assist_agent.food_pantries",
    "Hackathon_Team_Winners.homeless_assist_agent.restaurants_with_donations"
]

# 🔍 Column description from Unity Catalog
def get_column_descriptions(table: str) -> str:
    try:
        catalog, schema, table_name = table.split(".")
        query = f"""
        SELECT column_name, comment
        FROM {catalog}.information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table_name}'
        """
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=WAREHOUSE_PATH,
            access_token=DATABRICKS_TOKEN
        )
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        connection.close()
        return f"\nColumns in {table}:\n" + "\n".join(
            f"- {col[0]}: {col[1] if col[1] else 'No description'}" for col in rows
        )
    except Exception as e:
        return f"Failed to get column descriptions for {table}: {str(e)}"

column_context = "\n\n".join(get_column_descriptions(t) for t in tables)

# ✅ SQL execution function with fallback
def run_sql(query: str) -> str:
    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=WAREHOUSE_PATH,
            access_token=DATABRICKS_TOKEN
        )
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        connection.close()
        df = pd.DataFrame(rows, columns=columns)
        return df.to_markdown(index=False)
    except Exception as e:
        fallback_match = re.search(r"WHERE\\s+(.+?)\\s*=\\s*'(.+?)'", query, re.IGNORECASE)
        if fallback_match:
            field, value = fallback_match.groups()
            alt_query = query.replace(f"{field} = '{value}'", f"LOWER({field}) LIKE LOWER('%{value}%')")
            try:
                connection = sql.connect(
                    server_hostname=DATABRICKS_HOST,
                    http_path=WAREHOUSE_PATH,
                    access_token=DATABRICKS_TOKEN
                )
                with connection.cursor() as cursor:
                    cursor.execute(alt_query)
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                connection.close()
                df = pd.DataFrame(rows, columns=columns)
                return df.to_markdown(index=False)
            except Exception as e2:
                return f"SQL ERROR (fallback failed): {str(e2)}"
        return f"SQL ERROR: {str(e)}"

# ✅ LangChain tool wrapping SQL
databricks_sql_tool = Tool(
    name="DatabricksSQLTool",
    func=run_sql,
    description="""
Use this tool to query the following Databricks tables:

- Hackathon_Team_Winners.homeless_assist_agent.food_pantries
- Hackathon_Team_Winners.homeless_assist_agent.restaurants_with_donations

Write valid SQL SELECT statements. Do not try to INSERT or UPDATE. Only SELECT.
"""
)

# ✅ Set up LLM + memory
llm = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    temperature=0
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[databricks_sql_tool],
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# ✅ Chat loop with decision-making based on user follow-up
print("🟢 SQL Agent Chat Ready. Type 'exit' to stop.\n")

last_agent_response = ""

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("👋 Exiting chat.")
        break

    full_prompt = f"""
You are an intelligent SQL assistant with access to the following two Databricks tables:

- Hackathon_Team_Winners.homeless_assist_agent.food_pantries
- Hackathon_Team_Winners.homeless_assist_agent.restaurants_with_donations

Use the following schema information:
{column_context}

Conversation so far:
{memory.buffer_as_str}

User just said:
{user_input}

Instructions:
- If the user asks a new question, answer it with SQL.
- If they mention not liking a restaurant from a previous answer, rerun the query excluding that.
- If you list restaurants who are willing to donate, ask if you'd like to contact them to coordinate donations.
"""

    try:
        response = agent.run(full_prompt)
        print(f"\n🧠 Agent:\n{response}\n")
        last_agent_response = response

        if (
            "success rate" in response.lower() and 
            "restaurant" in response.lower() and 
            ("phone" in response.lower() or "contact" in response.lower())
        ):
            followup = input("🤖 Would you like me to contact these restaurants to coordinate food donations? (yes/no): ")
            if followup.strip().lower() in {"yes", "y"}:
                print("👍 Got it! (Contact feature will be implemented later.)\n")
            else:
                print("✅ Okay, we won't reach out for now.\n")

    except Exception as e:
        print(f"❌ Error: {e}")

# COMMAND ----------

# How many restaurants are willing to donate to First United Presbyterian Church - Food Distribution Center? List them ordered by success rate. Give me the top 3 and all their contact info.

# I don't like restaurant Barley, remove that from the list and give me the next best option

# COMMAND ----------

