import sqlite3
import json
import os
from datetime import datetime, date

DB_PATH = "fitnessbuddy.db"

# -------------------- DATABASE CONNECTION --------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


# -------------------- INITIALIZE DATABASE --------------------
def init_db():
    conn = get_db()
    cur = conn.cursor()

    # Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT,
        settings TEXT DEFAULT '{}'
    )
    """)

    # Foods table (shared catalog + custom foods)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS foods (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        kcal_per_100g REAL,
        protein_g REAL,
        carbs_g REAL,
        fat_g REAL,
        added_by_user_id INTEGER DEFAULT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Food logs (per user)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS food_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        meal TEXT,
        food_name TEXT,
        grams REAL,
        kcal REAL,
        protein REAL,
        carbs REAL,
        fat REAL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Sleep logs (optional)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sleep_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        sleep_date TEXT,
        bed_time TEXT,
        wake_time TEXT,
        duration_hours REAL,
        quality TEXT
    )
    """)

    # Water logs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS water_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        liters REAL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Mood logs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mood_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        mood INTEGER,
        emoji TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# -------------------- USER MANAGEMENT --------------------
def create_user(username, password, email=None, settings=None):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password, email, settings) VALUES (?, ?, ?, ?)",
            (username, password, email, json.dumps(settings or {}))
        )
        conn.commit()
        conn.close()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, str(e)


def authenticate_user(username, password):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password, email, settings FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False, "User not found."
    if row[2] != password:
        return False, "Incorrect password."
    user = {
        "id": row[0],
        "username": row[1],
        "email": row[3],
        "settings": json.loads(row[4] or "{}")
    }
    return True, user


def get_user_by_username(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username, email, settings FROM users WHERE username=?", (username,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "username": r[1],
        "email": r[2],
        "settings": json.loads(r[3] or "{}")
    }


def get_user_id(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=?", (username,))
    r = cur.fetchone()
    conn.close()
    return r[0] if r else None


def save_user_settings(username, settings):
    conn = get_db()
    conn.execute("UPDATE users SET settings=? WHERE username=?", (json.dumps(settings), username))
    conn.commit()
    conn.close()


# -------------------- FOOD SYSTEM --------------------
def load_default_foods():
    """
    Inserts default food items into the foods table if empty.
    """
    defaults = [
        ("Idli", 65, 2.0, 14.0, 0.4),
        ("Dosa", 168, 2.7, 22.0, 7.0),
        ("Upma", 120, 3.0, 23.0, 2.5),
        ("Poha", 130, 2.5, 27.0, 1.5),
        ("Paratha (Plain)", 270, 7.0, 45.0, 8.0),
        ("Omelette", 154, 11.0, 1.0, 12.0),
        ("Boiled Egg", 155, 13.0, 1.1, 11.0),
        ("Paneer Sandwich", 210, 10.0, 24.0, 8.0),
        ("Cooked Rice", 130, 2.7, 28.0, 0.3),
        ("Chapati", 250, 9.6, 45.0, 4.0),
        ("Dal (Cooked)", 120, 9.0, 12.0, 3.0),
        ("Rajma", 127, 8.7, 22.8, 0.5),
        ("Chole", 164, 8.9, 27.6, 2.6),
        ("Sambar", 70, 3.0, 10.0, 2.0),
        ("Pulao", 150, 3.0, 24.0, 4.0),
        ("Paneer (Raw)", 265, 18.3, 1.2, 20.8),
        ("Grilled Chicken Breast", 165, 31.0, 0.0, 3.6),
        ("Fish Curry", 120, 14.0, 5.0, 4.0),
        ("Tofu (Raw)", 76, 8.0, 1.9, 4.8),
        ("Sprouts Salad", 90, 8.0, 12.0, 1.0),
        ("Brown Bread", 247, 9.0, 45.0, 4.0),
        ("Peanut Butter", 588, 25.0, 20.0, 50.0),
        ("Oats (Cooked)", 71, 2.5, 12.0, 1.5),
        ("Yogurt (Plain)", 61, 3.5, 4.7, 3.3),
        ("Apple", 52, 0.3, 14.0, 0.2),
        ("Banana", 89, 1.1, 23.0, 0.3),
        ("Orange", 47, 0.9, 12.0, 0.1),
        ("Mango", 60, 0.8, 15.0, 0.4),
        ("Milk (Whole)", 60, 3.2, 5.0, 3.3),
        ("Burger (Veg)", 250, 9.0, 30.0, 10.0),
        ("Pizza (Cheese)", 280, 11.0, 33.0, 12.0),
        ("French Fries", 312, 3.4, 41.0, 15.0),
        ("Samosa", 262, 5.4, 31.0, 13.0),
        ("Pav Bhaji", 190, 4.0, 28.0, 7.0),
        ("Maggi Noodles", 365, 8.0, 55.0, 14.0),
        ("Gulab Jamun", 300, 4.0, 45.0, 10.0),
        ("Ice Cream (Vanilla)", 207, 3.5, 24.0, 11.0),
        ("Tea (With Milk & Sugar)", 45, 1.0, 8.0, 1.0),
        ("Black Coffee", 1, 0.1, 0.0, 0.0),
        ("Fruit Juice (Fresh)", 45, 0.5, 11.0, 0.2)
    ]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM foods")
    cnt = cur.fetchone()[0]
    if cnt == 0:
        cur.executemany(
            "INSERT INTO foods (name, kcal_per_100g, protein_g, carbs_g, fat_g) VALUES (?, ?, ?, ?, ?)",
            defaults
        )
        conn.commit()
    conn.close()


def get_foods(user_id=None, include_shared=True):
    conn = get_db()
    cur = conn.cursor()
    if user_id:
        cur.execute("""
            SELECT id, name, kcal_per_100g, protein_g, carbs_g, fat_g, added_by_user_id
            FROM foods
            WHERE added_by_user_id IS NULL OR added_by_user_id = ?
            ORDER BY added_by_user_id IS NOT NULL, name
        """, (user_id,))
    else:
        cur.execute("""
            SELECT id, name, kcal_per_100g, protein_g, carbs_g, fat_g, added_by_user_id
            FROM foods
            WHERE added_by_user_id IS NULL
            ORDER BY name
        """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "name": r[1],
            "kcal_per_100g": r[2],
            "protein_g": r[3],
            "carbs_g": r[4],
            "fat_g": r[5],
            "added_by_user_id": r[6],
        }
        for r in rows
    ]


def add_custom_food(user_id, name, kcal_per_100g, protein_g=0.0, carbs_g=0.0, fat_g=0.0):
    conn = get_db()
    conn.execute("""
        INSERT INTO foods (name, kcal_per_100g, protein_g, carbs_g, fat_g, added_by_user_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, kcal_per_100g, protein_g, carbs_g, fat_g, user_id))
    conn.commit()
    conn.close()
    return True


# -------------------- FOOD LOGS --------------------
def add_food_log(user_id, meal, food_name, grams, kcal, protein, carbs, fat):
    conn = get_db()
    conn.execute("""
        INSERT INTO food_logs (user_id, date, meal, food_name, grams, kcal, protein, carbs, fat)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, date.today().isoformat(), meal, food_name, grams, kcal, protein, carbs, fat))
    conn.commit()
    conn.close()


def get_food_logs(user_id, date=None):
    conn = get_db()
    cur = conn.cursor()
    if date:
        cur.execute("SELECT * FROM food_logs WHERE user_id=? AND date=? ORDER BY timestamp DESC", (user_id, date))
    else:
        cur.execute("SELECT * FROM food_logs WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows
