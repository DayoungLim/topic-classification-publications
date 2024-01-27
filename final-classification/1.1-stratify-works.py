
import sqlite3
import pandas as pd

read_conn = sqlite3.connect('works-with-abstract-title-topics.db')

query_total_count = "SELECT COUNT(*) AS total_rows FROM expanded_abstract;"
df_total_count = pd.read_sql_query(query_total_count, read_conn)
total_rows = df_total_count.iloc[0]['total_rows']

query_group_counts = "SELECT work_names, COUNT(*) AS group_count FROM expanded_abstract GROUP BY work_names;"
df_group_counts = pd.read_sql_query(query_group_counts, read_conn)

df_group_counts['sample_size'] = (df_group_counts['group_count'] * 20000 / total_rows).astype(int)

df_sampled = pd.DataFrame()

for index, row in df_group_counts.iterrows():
    print(f"Processing group: {row['work_names']}...")
    query_sample = f"""
    SELECT * FROM expanded_abstract 
    WHERE work_names = '{row['work_names']}'
    ORDER BY RANDOM()
    LIMIT {row['sample_size']};
    """
    df_temp = pd.read_sql_query(query_sample, read_conn)
    df_sampled = pd.concat([df_sampled, df_temp])

read_conn.close()

write_conn = sqlite3.connect('stratified-samples.db')

df_sampled.to_sql('stratified_sample', write_conn, if_exists='replace', index=False)

write_conn.close()

conn = sqlite3.connect('stratified-samples.db')

query = """
SELECT work_id, work_title, work_abstract, GROUP_CONCAT(work_names) as work_names
FROM stratified_sample
GROUP BY work_id, work_title, work_abstract;
"""

df_aggregated = pd.read_sql_query(query, conn)

df_aggregated.to_sql('stratified_sample_agg', conn, if_exists='replace', index=False)

conn.close()
print("Aggregated data processed and saved to 'aggregated_data' table in the database.")
