import sqlite3
import pandas as pd

conn = sqlite3.connect('crossref_data.db')

query = """
WITH SelectedSubjects AS (
    SELECT name
    FROM work_subjects
    GROUP BY name
),
RankedWorks AS (
    SELECT 
        w.id,
        w.title,
        w.abstract,
        ws.name,
        ROW_NUMBER() OVER (PARTITION BY ws.name ORDER BY w.id) AS rn
    FROM 
        works w
    JOIN  
        work_subjects ws ON w.id = ws.work_id
    JOIN 
        SelectedSubjects ss ON ws.name = ss.name
    WHERE 
        w.abstract IS NOT NULL
)
SELECT
    rw.id AS work_id,
    rw.title AS work_title,
    rw.abstract AS work_abstract,
    GROUP_CONCAT(DISTINCT rw.name) AS work_names
FROM 
    RankedWorks rw
GROUP BY 
    rw.id, rw.title, rw.abstract;
"""

df = pd.read_sql_query(query, conn)

conn.close()

new_db_path = 'works-with-abstract-title-topics.db'

new_conn = sqlite3.connect(new_db_path)

df.to_sql('expanded_abstract', new_conn, if_exists='replace', index=False)

new_conn.close()
