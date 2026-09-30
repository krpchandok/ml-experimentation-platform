import psycopg

conn = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="ml_platform",
    user="mluser",
    password="mlpassword"
)

print("Connected")