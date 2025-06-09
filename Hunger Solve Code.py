# Databricks notebook source
# MAGIC %pip install langchain langchain-community databricks-sql-connector langchain-openai tabulate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# 📦 Minimal & Improved Databricks SQL Agent with Fallback Retry (Chat Loop Version + Contact Template)

from databricks import sql
from langchain_community.chat_models import ChatDatabricks
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import pandas as pd
import re
import json
import os

# 🔐 Credentials
DATABRICKS_TOKEN = ""
DATABRICKS_HOST = "dbc-32826a1f-a515.cloud.databricks.com"
WAREHOUSE_PATH = "/sql/1.0/warehouses/031f2b46b9c894c1"

# 📘 Tables to describe
tables = [
    "Hackathon_Team_Winners.homeless_assist_agent.food_pantries",
    "Hackathon_Team_Winners.homeless_assist_agent.restaurants_with_donations"
]

# 🔍 Describe table columns using Unity Catalog metadata (INFORMATION_SCHEMA)
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

# Collect schema descriptions from all tables
column_context = "\n\n".join(get_column_descriptions(t) for t in tables)

# ✅ SQL execution wrapper with fallback matching
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

# ✅ Tool wrapping the SQL execution
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

# ✅ LLM and Memory
llm = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    temperature=0
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Initialize the agent
agent = initialize_agent(
    tools=[databricks_sql_tool],
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# ✅ Load intake form template
def load_intake_template():
    try:
        with open("Vikram_Sample_JSON_IntakeForm", "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Could not load intake form: {str(e)}"}

# ✅ Chat loop
print("🟢 Databricks SQL Agent is ready. Type 'exit' to quit.\n")

last_sql_response = ""

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("👋 Exiting chat.")
        break

    full_prompt = f"""
{column_context}

Task:
{user_input}
"""

    try:
        response = agent.run(full_prompt)
        print(f"\n🧠 Agent:\n{response}\n")
        last_sql_response = response

        contact_keywords = ["contact", "reach out", "send message", "notify"]
        if any(kw in user_input.lower() for kw in contact_keywords):
            intake_template = load_intake_template()
            if "error" in intake_template:
                print(f"⚠️ {intake_template['error']}")
            else:
                pattern = re.compile(r"^\s*\d+\.\s+([^\-–—]*)", re.IGNORECASE)
                matched_restaurants = []
                for line in last_sql_response.splitlines():
                    match = pattern.match(line)
                    if match:
                        restaurant_name = match.group(1).strip()
                        matched_restaurants.append(restaurant_name)

                if not matched_restaurants:
                    print("⚠️ Could not detect restaurants from previous response.")
                else:
                    for name in matched_restaurants:
                        form = json.loads(json.dumps(intake_template))  # Deep copy
                        form["restaurant_details"]["restaurant_name"] = name
                        form["restaurant_name"] = name  # Optional top-level consistency
                        if "primary_contact" in form["restaurant_details"]:
                            form["restaurant_details"]["primary_contact"]["name"] = ""
                            form["restaurant_details"]["primary_contact"]["phone"] = ""
                        print(f"\n📨 Contact form for {name}:")
                        print(json.dumps(form, indent=2))

    except Exception as e:
        print(f"❌ Error: {e}")

# COMMAND ----------

# How many restaurants are willing to donate to First United Presbyterian Church - Food Distribution Center? List them ordered by success rate. Give me the top 3

# I don't like restaurant Barley, remove that from the list and give me the next best option

# COMMAND ----------

# 📦 Minimal & Improved Databricks SQL Agent with Fallback Retry (Chat Loop Version + Contact Template)

from databricks import sql
from langchain_community.chat_models import ChatDatabricks
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import pandas as pd
import re
import json
import os

# 🔐 Credentials
DATABRICKS_TOKEN = "dapi84e0793bce8667c46e5a8661e9386de2"
DATABRICKS_HOST = "dbc-32826a1f-a515.cloud.databricks.com"
WAREHOUSE_PATH = "/sql/1.0/warehouses/031f2b46b9c894c1"

# 📘 Tables to describe
tables = [
    "Hackathon_Team_Winners.homeless_assist_agent.food_pantries",
    "Hackathon_Team_Winners.homeless_assist_agent.restaurants_with_donations"
]

# 🔍 Describe table columns using Unity Catalog metadata (INFORMATION_SCHEMA)
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

# Collect schema descriptions from all tables
column_context = "\n\n".join(get_column_descriptions(t) for t in tables)

# ✅ SQL execution wrapper with fallback matching
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

# ✅ Tool wrapping the SQL execution
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

# ✅ LLM and Memory
llm = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    temperature=0
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Initialize the agent
agent = initialize_agent(
    tools=[databricks_sql_tool],
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# ✅ Load intake form template
def load_intake_template():
    try:
        with open("Vikram_Sample_JSON_IntakeForm", "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Could not load intake form: {str(e)}"}

# ✅ Chat loop
print("🟢 Databricks SQL Agent is ready. Type 'exit' to quit.\n")

last_sql_response = ""

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("👋 Exiting chat.")
        break

    full_prompt = f"""
{column_context}

Task:
{user_input}
"""

    try:
        response = agent.run(full_prompt)
        print(f"\n🧠 Agent:\n{response}\n")
        last_sql_response = response

        contact_keywords = ["contact", "reach out", "send message", "notify"]
        if any(kw in user_input.lower() for kw in contact_keywords):
            intake_template = load_intake_template()
            if "error" in intake_template:
                print(f"⚠️ {intake_template['error']}")
            else:
                pattern = re.compile(r"^\s*\d+\.\s+([^\-\u2013\u2014]*)", re.IGNORECASE)
                matched_restaurants = []
                for line in last_sql_response.splitlines():
                    match = pattern.match(line)
                    if match:
                        restaurant_name = match.group(1).strip()
                        matched_restaurants.append(restaurant_name)

                if not matched_restaurants:
                    print("⚠️ Could not detect restaurants from previous response.")
                else:
                    for name in matched_restaurants:
                        form = json.loads(json.dumps(intake_template))  # Deep copy
                        form["restaurant_details"]["restaurant_name"] = name
                        form["restaurant_name"] = name

                        try:
                            detailed_info_query = f"""
                            SELECT * FROM Hackathon_Team_Winners.homeless_assist_agent.restaurants_with_donations
                            WHERE LOWER(name) = LOWER('{name}')
                            LIMIT 1
                            """
                            connection = sql.connect(
                                server_hostname=DATABRICKS_HOST,
                                http_path=WAREHOUSE_PATH,
                                access_token=DATABRICKS_TOKEN
                            )
                            with connection.cursor() as cursor:
                                cursor.execute(detailed_info_query)
                                row = cursor.fetchone()
                                columns = [desc[0] for desc in cursor.description]
                            connection.close()
                            if row:
                                info = dict(zip(columns, row))
                                form["restaurant_details"].update(info)
                                form.update(info)
                        except Exception as e:
                            print(f"⚠️ Could not load details for {name}: {e}")

                        print(f"\n📨 Contact form for {name}:")
                        print(json.dumps(form, indent=2))

    except Exception as e:
        print(f"❌ Error: {e}")


# COMMAND ----------

