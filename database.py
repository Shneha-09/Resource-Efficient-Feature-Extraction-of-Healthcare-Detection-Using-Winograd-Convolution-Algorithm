import sqlite3

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create a table for users
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

conn.commit()
conn.close()
print("Database and table created successfully!")
