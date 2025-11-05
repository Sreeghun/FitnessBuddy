# demo_app.py (Full updated version with per-user data, SQLite auth, and wellness dashboard)
import os
import json
import time
import itertools
import datetime as dt
from datetime import datetime, date
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from auth_utils import (
    init_db,
    create_user,
    authenticate_user,
    get_user_by_username,
    save_user_settings,
    load_default_foods,
    get_user_id,
    get_foods,
    add_custom_food,
    add_food_log,
    get_food_logs
)

from model_utils import extract_features_from_window, estimate_calories, LABEL_ENCODER_PATH
from synthetic_data import gen_window

# --- Daily Reminder Notifications ---
from datetime import datetime
import streamlit as st

now = datetime.now()
current_hour = now.hour

if 6 <= current_hour < 8:
    st.toast("☀️ Good morning! Don’t skip breakfast.")
elif 12 <= current_hour < 14:
    st.toast("🍴 Time for a healthy lunch!")
elif 19 <= current_hour < 21:
    st.toast("🌇 Dinner time! Keep it light tonight.")
elif 21 <= current_hour < 23:
    st.toast("😴 Wind down and log your sleep soon.")


# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ------------------------------------------------------------
# SETTINGS FILE FALLBACK (for guest/local mode)
# ------------------------------------------------------------
SETTINGS_FILE = "user_settings.json"

def default_settings():
    return {
        "name": "Guest User",
        "age": 25,
        "height_cm": 170,
        "weight_kg": 70,
        "goal": "Maintain Weight",
        "daily_calorie_target": 2000,
        "daily_water_target_l": 2.5,
        "animate_metrics": True
    }

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                s = json.load(f)
                for k, v in default_settings().items():
                    if k not in s:
                        s[k] = v
                return s
        except Exception:
            return default_settings()
    return default_settings()

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ------------------------------------------------------------
# USER HELPERS (per-user file management)
# ------------------------------------------------------------
def username_safe(name: str):
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).lower()

def get_current_username():
    u = st.session_state.get("user")
    if u and u.get("username"):
        return u["username"]
    return None

def user_file(base_filename: str):
    uname = get_current_username()
    if uname and uname.lower() != "guest":
        return f"{username_safe(uname)}_{base_filename}"
    return base_filename

def activity_key():
    uname = get_current_username() or "guest"
    return f"activity_calories_{username_safe(uname)}"

def set_activity_calories(val):
    st.session_state[activity_key()] = float(val)

def get_activity_calories():
    return float(st.session_state.get(activity_key(), 0.0))

# ------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="FitnessBuddy", layout="centered")
st.title("FitnessBuddy — Activity & Wellness (ML + Tracker)")

# Init DB (for user accounts)
init_db()
from auth_utils import load_default_foods, get_foods, add_custom_food, add_food_log, get_user_id
load_default_foods()

# ------------------------------------------------------------
# ACCOUNT / AUTH
# ------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "user_settings" not in st.session_state:
    st.session_state["user_settings"] = {}

st.sidebar.header("Account")
if st.session_state["user"] is None:
    mode = st.sidebar.selectbox("Action", ["Login", "Register", "Continue as Guest"])
    if mode == "Register":
        r_user = st.sidebar.text_input("Username")
        r_email = st.sidebar.text_input("Email (optional)")
        r_pwd = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Create account"):
            ok, err = create_user(r_user.strip(), r_pwd, email=r_email.strip() or None, settings=default_settings())
            if ok:
                st.sidebar.success("Account created! Please log in.")
            else:
                st.sidebar.error(f"Failed: {err}")
    elif mode == "Login":
        l_user = st.sidebar.text_input("Username")
        l_pwd = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            ok, data = authenticate_user(l_user.strip(), l_pwd)
            if ok:
                defs = default_settings()
                user_settings = data.get("settings", {}) or {}
                for k, v in defs.items():
                    user_settings.setdefault(k, v)
                st.session_state["user"] = data
                st.session_state["user_settings"] = user_settings
                st.sidebar.success(f"Hi {data['username']} — logged in.")
            else:
                st.sidebar.error(data)
    else:
        if st.sidebar.button("Continue as Guest"):
            st.session_state["user"] = {"username": "guest", "id": None}
            st.session_state["user_settings"] = load_settings()
            st.sidebar.info("Continuing as Guest")
else:
    user = st.session_state["user"]
    st.sidebar.write(f"Signed in as **{user['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state["user"] = None
        st.session_state["user_settings"] = {}
        st.experimental_rerun()

# ------------------------------------------------------------
# DISPLAY USER INFO AT TOP
# ------------------------------------------------------------
user = st.session_state.get("user")
if user and user.get("username"):
    us = st.session_state.get("user_settings", {})
    st.caption(f"Signed in as **{user['username']}**")
    st.markdown(f"**{us.get('name', user['username'])}** · Age: **{us.get('age', '—')}**")
else:
    st.caption("Signed in as **guest**")

# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    rf, cnn, le = None, None, None
    if os.path.exists("rf_model.joblib"):
        rf = joblib.load("rf_model.joblib")
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists("cnn_model.keras"):
            cnn = load_model("cnn_model.keras")
    except Exception:
        pass
    if os.path.exists(LABEL_ENCODER_PATH):
        le = joblib.load(LABEL_ENCODER_PATH)
    return rf, cnn, le

rf_model, cnn_model, label_encoder = load_models()

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def csv_to_windows(df, fs, window_seconds, overlap=0.0):
    n_window = int(window_seconds * fs)
    step = int(n_window * (1 - overlap))
    if step <= 0:
        step = n_window
    cols = [c for c in df.columns if c not in ("timestamp", "activity")]
    arr = df[cols].values
    windows = []
    for start in range(0, len(arr) - n_window + 1, step):
        windows.append(arr[start:start+n_window, :])
    return windows

def predict_window_rf(feats):
    if rf_model is None:
        return "RF_MISSING"
    f = np.array(feats).ravel().astype(np.float32)
    expected = getattr(rf_model, "n_features_in_", None)
    if expected and len(f) != expected:
        if len(f) < expected:
            f = np.concatenate([f, np.zeros(expected - len(f))])
        else:
            f = f[:expected]
    try:
        p = rf_model.predict([f])[0]
        return label_encoder.inverse_transform([p])[0] if label_encoder is not None else str(p)
    except Exception:
        return "RF_ERR"

def predict_window_cnn(window):
    if cnn_model is None:
        return "CNN_MISSING"
    arr = np.array(window).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    arr = arr[..., -3:]
    preds = cnn_model.predict(arr, verbose=0)
    pidx = np.argmax(preds, axis=1)[0]
    return label_encoder.inverse_transform([pidx])[0] if label_encoder is not None else str(pidx)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tabs = st.tabs(["Activity", "Food", "Sleep", "Water", "Mood", "Dashboard", "⚙️ Settings"])

# ---------------- ACTIVITY TAB ----------------
with tabs[0]:
    st.header("🏃 Activity Tracking")
    weight = float(st.session_state["user_settings"].get("weight_kg", 70))
    fs = 50
    window_seconds = 2.56

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run synthetic demo"):
            acts = ["SITTING","WALKING","JOGGING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","STANDING"]
            rows = []
            for act in acts:
                dfw = gen_window(act, fs=fs, seconds=window_seconds)
                arr = dfw[["ax","ay","az"]].values
                feats = extract_features_from_window(arr)
                label_rf = predict_window_rf(feats)
                label_cnn = predict_window_cnn(arr)
                used = label_cnn if label_cnn != "CNN_MISSING" else label_rf
                cals = estimate_calories(used, window_seconds, weight)
                rows.append((act, label_rf, label_cnn, used, cals))
            df = pd.DataFrame(rows, columns=["True","RF","CNN","Used","Calories"])
            st.dataframe(df)
            set_activity_calories(df["Calories"].sum())
    with col2:
        uploaded = st.file_uploader("Upload CSV (timestamp, ax, ay, az)", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            windows = csv_to_windows(df, fs, window_seconds)
            rows = []
            total = 0
            for w in windows:
                feats = extract_features_from_window(w[:, :3])
                label_rf = predict_window_rf(feats)
                label_cnn = predict_window_cnn(w)
                used = label_cnn if label_cnn != "CNN_MISSING" else label_rf
                cals = estimate_calories(used, window_seconds, weight)
                total += cals
                rows.append((label_rf,label_cnn,used,cals))
            st.dataframe(pd.DataFrame(rows, columns=["RF","CNN","Used","Calories"]))
            set_activity_calories(total)
            st.success(f"Total calories: {total:.2f} kcal")

# ---------------- FOOD TAB ----------------
# ---------------- FOOD TAB (DB-backed with smart search + weekly chart) ----------------
with tabs[1]:
    st.header("🍽 Food (DB-backed, with smart search & weekly chart)")

    # imports used locally in this block
    import os
    from datetime import date, datetime, timedelta
    import matplotlib.pyplot as plt

    # --- attempt to use DB functions; if not available, fallback to CSV files ---
    use_db = True
    try:
        # these should be imported at top of file: get_foods, add_custom_food, add_food_log, get_food_logs, load_default_foods, get_user_id
        load_default_foods()  # safe to call multiple times
    except Exception:
        use_db = False

    # CSV fallback paths
    FOOD_DB_CSV = "foods_db.csv"
    FOOD_LOG_CSV = "food_logs.csv"

    # Ensure a minimal CSV foods DB exists if using CSV fallback
    if not use_db:
        if not os.path.exists(FOOD_DB_CSV):
            sample_foods = pd.DataFrame([
                {"food_name": "Banana", "kcal_per_100g": 89, "protein_g": 1.1, "carbs_g": 22.8, "fat_g": 0.3},
                {"food_name": "Apple", "kcal_per_100g": 52, "protein_g": 0.3, "carbs_g": 14, "fat_g": 0.2},
                {"food_name": "Boiled Egg", "kcal_per_100g": 155, "protein_g": 13, "carbs_g": 1.1, "fat_g": 11},
                {"food_name": "Grilled Chicken Breast", "kcal_per_100g": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6},
                {"food_name": "Paneer", "kcal_per_100g": 265, "protein_g": 18.3, "carbs_g": 1.2, "fat_g": 20.8},
                {"food_name": "Rice (cooked)", "kcal_per_100g": 130, "protein_g": 2.7, "carbs_g": 28, "fat_g": 0.3},
                {"food_name": "Chapati", "kcal_per_100g": 250, "protein_g": 9.6, "carbs_g": 45, "fat_g": 4},
                {"food_name": "Dosa", "kcal_per_100g": 168, "protein_g": 2.7, "carbs_g": 22, "fat_g": 7},
            ])
            sample_foods.to_csv(FOOD_DB_CSV, index=False)

    # --- Load foods list (shared catalog + user custom if DB) or CSV ---
    uid = None
    cur_user = st.session_state.get("user")
    if cur_user:
        uid = cur_user.get("id") or (get_user_id(cur_user.get("username")) if "get_user_id" in globals() else None)

    foods_list = []
    if use_db:
        try:
            foods = get_foods(user_id=uid)
            foods_list = [f["name"] for f in foods]
        except Exception as e:
            # fallback to CSV if DB call fails unexpectedly
            st.warning(f"Food DB access failed, falling back to CSV: {e}")
            use_db = False

    if not use_db:
        foods_df = pd.read_csv(FOOD_DB_CSV)
        foods_list = foods_df["food_name"].tolist()

    # --- Smart Food Suggestion / Search ---
    st.subheader("Log a meal")
    search_term = st.text_input("Search food", placeholder="Type to filter foods (e.g. 'chi' -> chicken)")

    if search_term:
        matched = [n for n in foods_list if search_term.lower() in n.lower()]
        if not matched:
            st.info("No matches found — showing full list.")
            food_choice = st.selectbox("Choose food", foods_list)
        else:
            food_choice = st.selectbox("Suggested foods", matched)
    else:
        food_choice = st.selectbox("Choose food", foods_list)

    grams = st.number_input("Amount (g)", min_value=1.0, value=100.0, step=10.0)
    meal = st.selectbox("Meal", ["Breakfast", "Lunch", "Dinner", "Snack"])

    # --- Add custom food (user-level via DB or local CSV fallback) ---
    with st.expander("➕ Add custom food (private)"):
        cf_name = st.text_input("Food name (custom)", key="cf_name")
        cf_kcal = st.number_input("Calories per 100g", min_value=0.0, value=100.0, key="cf_kcal")
        cf_pro = st.number_input("Protein per 100g", min_value=0.0, value=0.0, key="cf_pro")
        cf_carbs = st.number_input("Carbs per 100g", min_value=0.0, value=0.0, key="cf_carbs")
        cf_fat = st.number_input("Fat per 100g", min_value=0.0, value=0.0, key="cf_fat")
        if st.button("Add custom food", key="add_custom_food_btn"):
            if use_db and uid:
                try:
                    add_custom_food(uid, cf_name.strip(), float(cf_kcal), float(cf_pro), float(cf_carbs), float(cf_fat))
                    st.success("Custom food added to your account.")
                    foods = get_foods(user_id=uid)
                    foods_list = [f["name"] for f in foods]
                except Exception as e:
                    st.error(f"Failed to add custom food to DB: {e}")
            else:
                # CSV fallback: append to foods CSV
                if cf_name.strip() == "":
                    st.warning("Enter a name for the custom food.")
                else:
                    new_row = {"food_name": cf_name.strip(), "kcal_per_100g": cf_kcal, "protein_g": cf_pro, "carbs_g": cf_carbs, "fat_g": cf_fat}
                    foods_df = pd.concat([foods_df, pd.DataFrame([new_row])], ignore_index=True)
                    foods_df.to_csv(FOOD_DB_CSV, index=False)
                    st.success("Custom food added locally.")

    # --- Log meal button (save to DB or CSV fallback) ---
    if st.button("Log this meal"):
        if use_db and uid:
            # find food record from foods (DB)
            rec = next((f for f in foods if f["name"] == food_choice), None)
            if rec is None:
                st.error("Selected food not found in DB.")
            else:
                kcal = float(rec.get("kcal_per_100g", 0.0)) * float(grams) / 100.0
                protein = float(rec.get("protein_g", 0.0)) * float(grams) / 100.0
                carbs = float(rec.get("carbs_g", 0.0)) * float(grams) / 100.0
                fat = float(rec.get("fat_g", 0.0)) * float(grams) / 100.0
                try:
                    add_food_log(uid, meal, rec["name"], float(grams), kcal, protein, carbs, fat)
                    st.success(f"Logged {grams} g {rec['name']} — {kcal:.1f} kcal")
                except Exception as e:
                    st.error(f"Failed to log meal: {e}")
        else:
            # CSV fallback: append to food_logs.csv
            row = {
                "user": cur_user.get("username") if cur_user else "guest",
                "date": date.today().isoformat(),
                "meal": meal,
                "food_name": food_choice,
                "grams": float(grams),
                "kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            # try to lookup nutrition from foods_df
            try:
                info = foods_df[foods_df["food_name"] == food_choice].iloc[0]
                row["kcal"] = float(info["kcal_per_100g"]) * row["grams"] / 100.0
                row["protein"] = float(info.get("protein_g", 0.0)) * row["grams"] / 100.0
                row["carbs"] = float(info.get("carbs_g", 0.0)) * row["grams"] / 100.0
                row["fat"] = float(info.get("fat_g", 0.0)) * row["grams"] / 100.0
            except Exception:
                pass
            if os.path.exists(FOOD_LOG_CSV):
                ldf = pd.read_csv(FOOD_LOG_CSV)
                ldf = pd.concat([ldf, pd.DataFrame([row])], ignore_index=True)
            else:
                ldf = pd.DataFrame([row])
            ldf.to_csv(FOOD_LOG_CSV, index=False)
            st.success(f"Logged {grams} g {food_choice} (local CSV)")

    # --- Show today's food logs (DB or CSV) with safe DataFrame handling ---
    st.markdown("---")
    st.subheader("Today's Meals")

    today_iso = date.today().isoformat()
    logs_df = None

    if use_db and uid:
        try:
            rows = get_food_logs(uid, date=today_iso)
            if rows:
                logs_df = pd.DataFrame(rows)
                # expected db columns: id,user_id,date,meal,food_name,grams,kcal,protein,carbs,fat,timestamp
                expected_cols = ["id","user_id","date","meal","food_name","grams","kcal","protein","carbs","fat","timestamp"]
                if logs_df.shape[1] == len(expected_cols):
                    logs_df.columns = expected_cols
                else:
                    logs_df.columns = [f"col_{i}" for i in range(logs_df.shape[1])]
        except Exception as e:
            st.error(f"Could not read logs from DB: {e}")
    else:
        if os.path.exists(FOOD_LOG_CSV):
            all_logs = pd.read_csv(FOOD_LOG_CSV)
            try:
                logs_df = all_logs[all_logs["date"] == today_iso]
            except Exception:
                logs_df = all_logs[all_logs["date"] == today_iso] if "date" in all_logs.columns else all_logs

    if logs_df is None or logs_df.empty:
        st.info("No meals logged today.")
    else:
        # select safe columns for display
        display_cols = [c for c in ["date","meal","food_name","grams","kcal"] if c in logs_df.columns]
        st.dataframe(logs_df[display_cols].tail(10))

    # --- Weekly Nutrition Summary (last 7 days) ---
    st.markdown("---")
    st.subheader("📊 Weekly Nutrition Summary (last 7 days)")

    # Build a DataFrame 'nutr' with columns date,kcal,protein,carbs,fat
    nutr = None
    try:
        if use_db and uid:
            # fetch last 14 days for safety and group
            all_rows = []
            for d in range(0, 14):
                dd = (date.today() - timedelta(days=d)).isoformat()
                rows = get_food_logs(uid, date=dd)
                if rows:
                    for r in rows:
                        all_rows.append(r)
            if all_rows:
                df_all = pd.DataFrame(all_rows)
                expected_cols = ["id","user_id","date","meal","food_name","grams","kcal","protein","carbs","fat","timestamp"]
                if df_all.shape[1] == len(expected_cols):
                    df_all.columns = expected_cols
                nutr = df_all.groupby("date")[["kcal","protein","carbs","fat"]].sum().sort_index().tail(7)
        else:
            if os.path.exists(FOOD_LOG_CSV):
                fl = pd.read_csv(FOOD_LOG_CSV)
                if "date" in fl.columns:
                    fl["date"] = pd.to_datetime(fl["date"], errors="coerce").dt.date
                    last7 = (date.today() - timedelta(days=6))
                    fl7 = fl[fl["date"] >= last7]
                    if not fl7.empty:
                        # ensure kcal/protein/carbs/fat column names match
                        for col in ["kcal","protein","carbs","fat"]:
                            if col not in fl7.columns:
                                fl7[col] = 0.0
                        nutr = fl7.groupby("date")[["kcal","protein","carbs","fat"]].sum().sort_index()
    except Exception as e:
        st.error(f"Failed to prepare weekly summary: {e}")

    if nutr is None or nutr.empty:
        st.info("Not enough data for weekly summary.")
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        nutr.plot(kind="bar", ax=ax)
        ax.set_title("Nutrition Intake (Last 7 Days)")
        ax.set_ylabel("Amount (kcal / g)")
        ax.set_xlabel("Date")
        ax.legend(["Calories (kcal)","Protein (g)","Carbs (g)","Fat (g)"])
        st.pyplot(fig)
# --- Update session total calories eaten for today ---
if logs_df is not None and not logs_df.empty:
    total_eaten_today = float(logs_df["kcal"].sum()) if "kcal" in logs_df.columns else 0.0
    st.session_state["food_calories"] = total_eaten_today
else:
    st.session_state["food_calories"] = 0.0



# ---------------- SLEEP TAB ----------------
with tabs[2]:
    st.header("😴 Sleep Log")
    SLEEP_LOG = user_file("sleep_logs.csv")
    sleep_date = st.date_input("Sleep Date", date.today())
    sleep_time = st.time_input("Bed Time", dt.time(22,30))
    wake_time = st.time_input("Wake Time", dt.time(6,30))
    if st.button("Save Sleep"):
        start = datetime.combine(sleep_date, sleep_time)
        end = datetime.combine(sleep_date, wake_time)
        if end <= start:
            end += dt.timedelta(days=1)
        dur = (end - start).total_seconds() / 3600
        rec = {"date": sleep_date.isoformat(), "duration_hours": round(dur,2)}
        df = pd.read_csv(SLEEP_LOG) if os.path.exists(SLEEP_LOG) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        df.to_csv(SLEEP_LOG, index=False)
        st.success(f"Logged {dur:.1f} hours")

# ---------------- WATER TAB ----------------
with tabs[3]:
    st.header("💧 Water")
    WATER_LOG = user_file("water_logs.csv")
    amt = st.number_input("Liters", 0.0, 5.0, 0.25)
    if st.button("Log Water"):
        rec = {"date": date.today().isoformat(), "liters": amt}
        df = pd.read_csv(WATER_LOG) if os.path.exists(WATER_LOG) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        df.to_csv(WATER_LOG, index=False)
        st.success(f"Logged {amt:.2f} L")

# ---------------- MOOD TAB ----------------
with tabs[4]:
    st.header("🙂 Mood")
    MOOD_LOG = user_file("mood_logs.csv")
    mood = st.slider("Mood (1=bad,5=great)", 1, 5, 3)
    if st.button("Save Mood"):
        rec = {"date": date.today().isoformat(), "mood": mood}
        df = pd.read_csv(MOOD_LOG) if os.path.exists(MOOD_LOG) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        df.to_csv(MOOD_LOG, index=False)
        st.success("Mood saved")

# ---------------- DASHBOARD TAB ----------------
with tabs[5]:
    st.header("📅 Dashboard — Daily Summary")
    sel = date.today().isoformat()

    # --- Load safe user settings with defaults ---
    settings = st.session_state.get("user_settings", {}) or {}
    defaults = {
        "daily_calorie_target": 2000,
        "daily_water_target_l": 2.5,
        "weight_kg": 70,
        "height_cm": 170,
        "goal": "Maintain Weight",
    }
    for k, v in defaults.items():
        settings.setdefault(k, v)

    cal_target = float(settings["daily_calorie_target"])
    water_target = float(settings["daily_water_target_l"])

    # --- File paths ---
    FOOD_LOG = user_file("food_logs.csv")
    SLEEP_LOG = user_file("sleep_logs.csv")
    WATER_LOG = user_file("water_logs.csv")
    MOOD_LOG = user_file("mood_logs.csv")

    # --- Calories eaten (safe) ---
    eaten = float(st.session_state.get("food_calories", 0.0))
    if os.path.exists(FOOD_LOG):
        try:
            food_df = pd.read_csv(FOOD_LOG)
            if "date" in food_df.columns:
                food_df = food_df[food_df["date"] == sel]
            # Try to find the right column for calories
            for col in ["kcal", "calories", "energy"]:
                if col in food_df.columns:
                    eaten = float(food_df[col].sum())
                    break
        except Exception as e:
            st.warning(f"⚠️ Could not load calories eaten: {e}")

    # --- Water intake (safe) ---
    water = 0.0
    if os.path.exists(WATER_LOG):
        try:
            water_df = pd.read_csv(WATER_LOG)
            if "date" in water_df.columns:
                water_df = water_df[water_df["date"] == sel]
            if "liters" in water_df.columns:
                water = float(water_df["liters"].sum())
        except Exception as e:
            st.warning(f"⚠️ Could not load water data: {e}")

    # --- Sleep hours (safe) ---
    sleep = 0.0
    if os.path.exists(SLEEP_LOG):
        try:
            s_df = pd.read_csv(SLEEP_LOG)
            if not s_df.empty and "duration_hours" in s_df.columns:
                sleep = float(s_df.iloc[-1]["duration_hours"])
        except Exception as e:
            st.warning(f"⚠️ Could not load sleep data: {e}")

    # --- Mood (safe) ---
    mood = 3
    if os.path.exists(MOOD_LOG):
        try:
            m_df = pd.read_csv(MOOD_LOG)
            if not m_df.empty and "mood" in m_df.columns:
                mood = int(m_df.iloc[-1]["mood"])
        except Exception as e:
            st.warning(f"⚠️ Could not load mood data: {e}")

    # --- Activity (from session or saved function) ---
    activity = 0.0
    try:
        if "get_activity_calories" in globals():
            activity = float(get_activity_calories())
        else:
            activity = float(st.session_state.get("activity_calories", 0.0))
    except Exception:
        activity = 0.0

    # --- Display metrics nicely ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔥 Calories Burned", f"{activity:.1f} kcal")
    col2.metric("🍽 Calories Eaten", f"{eaten:.1f} kcal")
    col3.metric("💧 Water Intake", f"{water:.1f} / {water_target:.1f} L")
    col4.metric("😴 Sleep", f"{sleep:.1f} hrs")

    # --- Mood emoji ---
    mood_map = {1: "😢", 2: "😞", 3: "😐", 4: "😊", 5: "🤩"}
    mood_emoji = mood_map.get(mood, "😐")
    st.markdown(f"**Today's Mood:** {mood_emoji} ({mood})")

    # --- Optional debug line ---
    # st.write(f"DEBUG → Calories eaten today: {eaten} kcal")

    # --- Simple wellness summary ---
    score = 0
    if cal_target > 0 and water_target > 0:
        sleep_score = min((sleep / 8) * 100, 100)
        calorie_score = max(0, 100 - abs((eaten - activity) / 20))
        water_score = min((water / water_target) * 100, 100)
        mood_score = (mood / 5) * 100
        score = int((0.3 * sleep_score) + (0.3 * calorie_score) + (0.2 * water_score) + (0.2 * mood_score))
    st.progress(score / 100)
    st.write(f"💯 **Wellness Score:** {score}/100")

    # --- Quick summary sentence ---
    st.markdown(
        f"📊 *Today you’ve eaten {eaten:.0f} kcal, burned {activity:.0f} kcal, slept {sleep:.1f} hours, "
        f"and drank {water:.1f} L of water.*"
    )


# ---------------- SETTINGS TAB (safe) ----------------
with tabs[6]:
    st.header("⚙️ Settings & Profile")
    # Safe load of settings (put this wherever you used settings = st.session_state["user_settings"])
    settings = st.session_state.get("user_settings", None)
    if not settings:
        # fallback to local file defaults if session doesn't have user settings
        settings = load_settings()
    # ensure all default keys exist
    defs = default_settings()
    for k, v in defs.items():
        settings.setdefault(k, v)
    # optionally update session_state so settings persist during this session
    st.session_state["user_settings"] = settings

    # show current account
    cur = st.session_state.get("user")
    if cur and cur.get("username"):
        st.caption(f"Account: **{cur['username']}**")

    with st.form("settings_form"):
        st.subheader("👤 User Profile")
        # use settings.get(...) to avoid KeyError
        name = st.text_input("Name", value=settings.get("name", "Guest User"))
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=int(settings.get("age", 25)))
        with col2:
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=int(settings.get("height_cm", 170)))
        with col3:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=float(settings.get("weight_kg", 70.0)))

        st.subheader("🎯 Fitness Goal")
        goal = st.selectbox("Select Goal", ["Lose Weight", "Maintain Weight", "Gain Weight"], index=["Lose Weight", "Maintain Weight", "Gain Weight"].index(settings.get("goal", "Maintain Weight")))

        st.subheader("📅 Daily Targets")
        col1, col2 = st.columns(2)
        with col1:
            daily_calorie_target = st.number_input("Calorie Target (kcal)", min_value=1000, max_value=5000, value=int(settings.get("daily_calorie_target", 2000)))
        with col2:
            daily_water_target_l = st.number_input("Water Target (liters)", min_value=0.5, max_value=10.0, value=float(settings.get("daily_water_target_l", 2.5)))

        st.subheader("✨ UI Effects")
        animate_metrics = st.checkbox("Animate metrics (count-up)", value=bool(settings.get("animate_metrics", True)))

        # BMI preview
        st.markdown("---")
        colL, colR = st.columns([2,1])
        with colR:
            h_m = height_cm / 100.0
            bmi_val = weight_kg / (h_m*h_m) if h_m > 0 else 0.0
            def bmi_status(bmi_val):
                if bmi_val < 18.5:
                    return "Underweight"
                elif bmi_val < 25:
                    return "Normal"
                elif bmi_val < 30:
                    return "Overweight"
                else:
                    return "Obese"
            st.metric("BMI", f"{bmi_val:.1f}", bmi_status(bmi_val))

        submitted = st.form_submit_button("💾 Save Settings")
        if submitted:
            new_settings = {
                "name": name,
                "age": int(age),
                "height_cm": int(height_cm),
                "weight_kg": float(weight_kg),
                "goal": goal,
                "daily_calorie_target": int(daily_calorie_target),
                "daily_water_target_l": float(daily_water_target_l),
                "animate_metrics": bool(animate_metrics)
            }

            # persist: DB for logged in users, local file for guests
            current_user = st.session_state.get("user")
            if current_user and current_user.get("username") and current_user["username"].lower() != "guest":
                try:
                    save_user_settings(current_user["username"], new_settings)
                    st.session_state["user_settings"] = new_settings
                    st.success("Settings saved to your account.")
                except Exception as e:
                    st.error(f"Failed to save to account: {e}")
            else:
                save_settings(new_settings)
                st.session_state["user_settings"] = new_settings
                st.success("Settings saved locally.")


# ---------------- FOOTER ----------------
st.markdown("---")
st.write(f"RandomForest: {'✅' if rf_model else '❌'} | CNN: {'✅' if cnn_model else '❌'} | LabelEncoder: {'✅' if label_encoder else '❌'}")
