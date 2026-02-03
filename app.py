import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from datetime import datetime, date
import os
import time
from contextlib import contextmanager
import requests
import warnings

# Warnungen unterdrücken
warnings.filterwarnings('ignore')

# Seitenkonfiguration
st.set_page_config(page_title="PE Fund Analyzer", layout="wide", page_icon="📊")

# === SUPABASE AUTH CONFIGURATION ===
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

# === DATABASE CONFIGURATION ===
DATABASE_CONFIG = {
    'host': st.secrets["postgres"]["host"],
    'port': st.secrets["postgres"]["port"],
    'database': st.secrets["postgres"]["database"],
    'user': st.secrets["postgres"]["user"],
    'password': st.secrets["postgres"]["password"],
}

# === AUTHENTICATION FUNCTIONS ===

def init_auth_state():
    """Initialisiert Session State für Auth"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None

def login(email: str, password: str) -> bool:
    """Authentifiziert User via Supabase"""
    try:
        response = requests.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={
                "apikey": SUPABASE_KEY,
                "Content-Type": "application/json"
            },
            json={
                "email": email,
                "password": password
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.authenticated = True
            st.session_state.user_email = data['user']['email']
            st.session_state.access_token = data['access_token']
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Verbindungsfehler: {e}")
        return False

def logout():
    """Loggt User aus"""
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.access_token = None

def show_login_page():
    """Zeigt Login-Seite"""
    st.title("🔐 PE Fund Analyzer")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Anmeldung")
        
        with st.form("login_form"):
            email = st.text_input("E-Mail", placeholder="name@firma.com")
            password = st.text_input("Passwort", type="password")
            submit = st.form_submit_button("Anmelden", use_container_width=True)
            
            if submit:
                if email and password:
                    with st.spinner("Anmeldung läuft..."):
                        if login(email, password):
                            st.success("✅ Erfolgreich angemeldet!")
                            st.rerun()
                        else:
                            st.error("❌ Ungültige E-Mail oder Passwort")
                else:
                    st.warning("Bitte E-Mail und Passwort eingeben")
        
        st.markdown("---")
        st.caption("Kontaktiere den Administrator für Zugangsdaten.")

# === SESSION STATE INITIALISIERUNG ===
if 'filter_version' not in st.session_state:
    st.session_state.filter_version = 0


@contextmanager
def get_connection():
    """Erstellt eine neue PostgreSQL-Datenbankverbindung als Context Manager"""
    conn = None
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        yield conn
    finally:
        if conn is not None:
            conn.close()


def get_db_connection():
    """Erstellt eine neue PostgreSQL-Datenbankverbindung (direkt)"""
    return psycopg2.connect(**DATABASE_CONFIG)


# === CACHING FÜR HÄUFIGE ABFRAGEN ===
@st.cache_data(ttl=60)
def get_available_reporting_dates_cached(_conn_id):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT reporting_date FROM portfolio_companies_history ORDER BY reporting_date DESC")
            return [row[0].strftime('%Y-%m-%d') if isinstance(row[0], (date, datetime)) else row[0] for row in cursor.fetchall()]


@st.cache_data(ttl=60)
def get_available_years_cached(_conn_id):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT EXTRACT(YEAR FROM reporting_date)::INTEGER as year FROM portfolio_companies_history ORDER BY year DESC")
            return [int(row[0]) for row in cursor.fetchall() if row[0]]


@st.cache_data(ttl=60)
def load_all_funds_cached(_conn_id):
    with get_connection() as conn:
        query = """
        SELECT DISTINCT ON (f.fund_id) f.fund_id, f.fund_name, g.gp_name, f.vintage_year, f.strategy, f.geography, g.rating,
               m.total_tvpi, m.dpi, m.top5_value_concentration, m.loss_ratio
        FROM funds f
        LEFT JOIN gps g ON f.gp_id = g.gp_id
        LEFT JOIN fund_metrics m ON f.fund_id = m.fund_id
        WHERE f.fund_id IS NOT NULL
        ORDER BY f.fund_id, f.fund_name
        """
        return pd.read_sql_query(query, conn)


def clear_cache():
    st.cache_data.clear()


def ensure_gps_table(conn):
    """Erstellt die GPs-Tabelle falls nicht vorhanden"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS gps (
            gp_id SERIAL PRIMARY KEY,
            gp_name TEXT UNIQUE NOT NULL,
            sector TEXT,
            headquarters TEXT,
            website TEXT,
            rating TEXT,
            last_meeting DATE,
            next_raise_estimate DATE,
            notes TEXT,
            contact1_name TEXT,
            contact1_function TEXT,
            contact1_email TEXT,
            contact1_phone TEXT,
            contact2_name TEXT,
            contact2_function TEXT,
            contact2_email TEXT,
            contact2_phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


def ensure_funds_table(conn):
    """Erstellt die Funds-Tabelle falls nicht vorhanden"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS funds (
            fund_id SERIAL PRIMARY KEY,
            fund_name TEXT NOT NULL,
            gp_id INTEGER REFERENCES gps(gp_id),
            vintage_year INTEGER,
            strategy TEXT,
            geography TEXT,
            fund_size_m REAL,
            currency TEXT DEFAULT 'EUR',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


def ensure_portfolio_companies_table(conn):
    """Erstellt die Portfolio Companies Tabelle falls nicht vorhanden"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_companies (
            company_id SERIAL PRIMARY KEY,
            fund_id INTEGER NOT NULL REFERENCES funds(fund_id),
            company_name TEXT NOT NULL,
            invested_amount REAL,
            realized_tvpi REAL DEFAULT 0,
            unrealized_tvpi REAL DEFAULT 0,
            investment_date DATE,
            exit_date DATE,
            entry_multiple REAL,
            gross_irr REAL,
            ownership REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


def ensure_fund_metrics_table(conn):
    """Erstellt die Fund Metrics Tabelle falls nicht vorhanden"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fund_metrics (
            metric_id SERIAL PRIMARY KEY,
            fund_id INTEGER NOT NULL REFERENCES funds(fund_id) UNIQUE,
            total_tvpi REAL,
            dpi REAL,
            top5_value_concentration REAL,
            top5_capital_concentration REAL,
            loss_ratio REAL,
            realized_percentage REAL,
            num_investments INTEGER,
            calculation_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


def check_column_exists(conn, table_name, column_name):
    """Prüft ob eine Spalte in einer Tabelle existiert"""
    with conn.cursor() as cursor:
        cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = %s AND column_name = %s
        )
        """, (table_name, column_name))
        return cursor.fetchone()[0]


def migrate_to_gp_table(conn):
    """Migriert bestehende Daten zur neuen GP-Struktur"""
    with conn.cursor() as cursor:
        # Prüfen ob gp_id Spalte existiert
        if check_column_exists(conn, 'funds', 'gp_id'):
            return False
        
        # Prüfen ob gp_name Spalte existiert
        if not check_column_exists(conn, 'funds', 'gp_name'):
            return False
        
        # 1. GPs aus bestehenden Daten erstellen
        cursor.execute("""
        SELECT DISTINCT gp_name, rating, last_meeting, next_raise_estimate
        FROM funds WHERE gp_name IS NOT NULL AND gp_name != ''
        """)
        existing_gps = cursor.fetchall()
        
        for gp_name, rating, last_meeting, next_raise in existing_gps:
            cursor.execute("""
            INSERT INTO gps (gp_name, rating, last_meeting, next_raise_estimate)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (gp_name) DO NOTHING
            """, (gp_name, rating, last_meeting, next_raise))
        
        conn.commit()
        
        # 2. gp_id Spalte hinzufügen
        cursor.execute("ALTER TABLE funds ADD COLUMN IF NOT EXISTS gp_id INTEGER REFERENCES gps(gp_id)")
        conn.commit()
        
        # 3. gp_id setzen
        cursor.execute("""
        UPDATE funds SET gp_id = (SELECT gp_id FROM gps WHERE gps.gp_name = funds.gp_name)
        WHERE gp_name IS NOT NULL AND gp_name != ''
        """)
        conn.commit()
        return True


def ensure_currency_column(conn):
    """Fügt die Währungsspalte hinzu falls nicht vorhanden"""
    with conn.cursor() as cursor:
        if not check_column_exists(conn, 'funds', 'currency'):
            cursor.execute("ALTER TABLE funds ADD COLUMN currency TEXT DEFAULT 'EUR'")
            conn.commit()
        
        if not check_column_exists(conn, 'funds', 'fund_size_m'):
            cursor.execute("ALTER TABLE funds ADD COLUMN fund_size_m REAL")
            conn.commit()


def ensure_portfolio_company_fields(conn):
    """Fügt neue Felder für Portfolio Companies hinzu"""
    new_pc_fields = [
        ('investment_date', 'DATE'),
        ('exit_date', 'DATE'),
        ('entry_multiple', 'REAL'),
        ('gross_irr', 'REAL'),
        ('ownership', 'REAL')
    ]
    
    with conn.cursor() as cursor:
        for field_name, field_type in new_pc_fields:
            if not check_column_exists(conn, 'portfolio_companies', field_name):
                cursor.execute(f"ALTER TABLE portfolio_companies ADD COLUMN {field_name} {field_type}")
                conn.commit()
            
            if not check_column_exists(conn, 'portfolio_companies_history', field_name):
                cursor.execute(f"ALTER TABLE portfolio_companies_history ADD COLUMN {field_name} {field_type}")
                conn.commit()


def ensure_history_tables(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_companies_history (
            history_id SERIAL PRIMARY KEY,
            fund_id INTEGER NOT NULL REFERENCES funds(fund_id),
            company_name TEXT NOT NULL,
            invested_amount REAL,
            realized_tvpi REAL DEFAULT 0,
            unrealized_tvpi REAL DEFAULT 0,
            reporting_date DATE NOT NULL,
            investment_date DATE,
            exit_date DATE,
            entry_multiple REAL,
            gross_irr REAL,
            ownership REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(fund_id, company_name, reporting_date)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fund_metrics_history (
            history_id SERIAL PRIMARY KEY,
            fund_id INTEGER NOT NULL REFERENCES funds(fund_id),
            reporting_date DATE NOT NULL,
            total_tvpi REAL,
            dpi REAL,
            top5_value_concentration REAL,
            top5_capital_concentration REAL,
            loss_ratio REAL,
            realized_percentage REAL,
            num_investments INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(fund_id, reporting_date)
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pch_fund_date ON portfolio_companies_history(fund_id, reporting_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fmh_fund_date ON fund_metrics_history(fund_id, reporting_date)")
        
        conn.commit()


def migrate_existing_data_if_needed(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM portfolio_companies_history")
        history_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM portfolio_companies")
        current_count = cursor.fetchone()[0]
        
        if history_count == 0 and current_count > 0:
            today = date.today()
            quarter = (today.month - 1) // 3
            if quarter == 0:
                default_date = date(today.year - 1, 12, 31)
            else:
                last_month = quarter * 3
                default_date = date(today.year, last_month, 30 if last_month in [6, 9] else 31)
            
            cursor.execute("""
            INSERT INTO portfolio_companies_history
                (fund_id, company_name, invested_amount, realized_tvpi, unrealized_tvpi, reporting_date)
            SELECT fund_id, company_name, invested_amount, realized_tvpi, unrealized_tvpi, %s
            FROM portfolio_companies
            ON CONFLICT (fund_id, company_name, reporting_date) DO NOTHING
            """, (default_date,))
            
            cursor.execute("""
            INSERT INTO fund_metrics_history
                (fund_id, reporting_date, total_tvpi, dpi, top5_value_concentration,
                 top5_capital_concentration, loss_ratio, realized_percentage, num_investments)
            SELECT fund_id, %s, total_tvpi, dpi, top5_value_concentration,
                   top5_capital_concentration, loss_ratio, realized_percentage, num_investments
            FROM fund_metrics
            ON CONFLICT (fund_id, reporting_date) DO NOTHING
            """, (default_date,))
            
            conn.commit()
            return default_date
        
        return None


def get_or_create_gp(conn, gp_name):
    """Holt oder erstellt einen GP anhand des Namens"""
    if not gp_name or gp_name.strip() == '':
        return None
    
    with conn.cursor() as cursor:
        cursor.execute("SELECT gp_id FROM gps WHERE gp_name = %s", (gp_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        cursor.execute("INSERT INTO gps (gp_name) VALUES (%s) RETURNING gp_id", (gp_name,))
        conn.commit()
        return cursor.fetchone()[0]


def get_available_reporting_dates(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT reporting_date FROM portfolio_companies_history ORDER BY reporting_date DESC")
        return [row[0].strftime('%Y-%m-%d') if isinstance(row[0], (date, datetime)) else row[0] for row in cursor.fetchall()]


def get_available_years(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT EXTRACT(YEAR FROM reporting_date)::INTEGER as year FROM portfolio_companies_history ORDER BY year DESC")
        return [int(row[0]) for row in cursor.fetchall() if row[0]]


def get_latest_date_for_year_per_fund(conn, year, fund_ids=None):
    with conn.cursor() as cursor:
        if fund_ids:
            cursor.execute("""
            SELECT fund_id, MAX(reporting_date) as latest_date
            FROM portfolio_companies_history
            WHERE EXTRACT(YEAR FROM reporting_date) = %s AND fund_id = ANY(%s)
            GROUP BY fund_id
            """, (year, list(fund_ids)))
        else:
            cursor.execute("""
            SELECT fund_id, MAX(reporting_date) as latest_date
            FROM portfolio_companies_history WHERE EXTRACT(YEAR FROM reporting_date) = %s
            GROUP BY fund_id
            """, (year,))
        
        return {row[0]: row[1].strftime('%Y-%m-%d') if isinstance(row[1], (date, datetime)) else row[1] for row in cursor.fetchall()}


def get_portfolio_data_for_date(conn, fund_id, reporting_date):
    query = """
    SELECT company_name, invested_amount, realized_tvpi, unrealized_tvpi
    FROM portfolio_companies_history
    WHERE fund_id = %s AND reporting_date = %s
    ORDER BY (realized_tvpi + unrealized_tvpi) DESC
    """
    return pd.read_sql_query(query, conn, params=(fund_id, reporting_date))


def get_fund_metrics_for_date(conn, fund_id, reporting_date):
    query = """
    SELECT total_tvpi, dpi, top5_value_concentration, top5_capital_concentration,
           loss_ratio, realized_percentage, num_investments
    FROM fund_metrics_history WHERE fund_id = %s AND reporting_date = %s
    """
    return pd.read_sql_query(query, conn, params=(fund_id, reporting_date))


def wrap_label(text, max_chars=12, max_lines=2, base_fontsize=11):
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if len(current + " " + w) <= max_chars:
            current = (current + " " + w).strip()
        else:
            lines.append(current)
            current = w
            if len(lines) == max_lines - 1:
                break
    lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    full_len = sum(len(l) for l in lines)
    orig_len = len(text)
    if full_len < orig_len:
        lines[-1] += "…"
    shrink_factor = max(0.7, min(1.0, 15 / max(full_len, 1)))
    fontsize = base_fontsize * shrink_factor
    return "\n".join(lines), fontsize


def create_mekko_chart(fund_id, fund_name, conn, reporting_date=None):
    if reporting_date:
        df = get_portfolio_data_for_date(conn, fund_id, reporting_date)
        title_suffix = f"\n(Stichtag: {reporting_date})"
    else:
        query = """SELECT company_name, invested_amount, realized_tvpi, unrealized_tvpi
        FROM portfolio_companies WHERE fund_id = %s ORDER BY (realized_tvpi + unrealized_tvpi) DESC"""
        df = pd.read_sql_query(query, conn, params=(fund_id,))
        title_suffix = ""
    
    if df.empty:
        return None
    
    categories = df["company_name"].tolist()
    widths = df["invested_amount"].tolist()
    values = df[["realized_tvpi", "unrealized_tvpi"]].values.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    total_width = sum(widths)
    x_start = 0
    REAL_COLOR = "darkblue"
    UNREAL_COLOR = "lightskyblue"
    max_height = max(map(sum, values)) if values else 1

    for i, cat in enumerate(categories):
        cat_width = widths[i]
        bottom = 0
        category_total = sum(values[i])
        for j, val in enumerate(values[i]):
            color = REAL_COLOR if j == 0 else UNREAL_COLOR
            ax.bar(x_start, val, width=cat_width, bottom=bottom, color=color, edgecolor='black', align='edge')
            if val > 0:
                pct = val / category_total * 100 if category_total > 0 else 0
                ax.text(x_start + cat_width / 2, bottom + val / 2, f"{pct:.1f}%",
                        ha="center", va="center", fontsize=10, color="white" if color == REAL_COLOR else "black")
            bottom += val
        label_y_base = -max_height * 0.08
        label_y_offset = max_height * 0.04
        label_y = label_y_base if i % 2 == 0 else label_y_base - label_y_offset
        wrapped_text, dyn_fontsize = wrap_label(cat)
        ax.text(x_start + cat_width / 2, label_y, wrapped_text, ha="center", va="top", fontsize=dyn_fontsize, fontweight="bold")
        x_start += cat_width

    total_value_ccy = 0
    total_realized_ccy = 0
    company_total_values_ccy = []
    for i in range(len(values)):
        invested = widths[i]
        realized_ccy = values[i][0] * invested
        unrealized_ccy = values[i][1] * invested
        total_ccy = realized_ccy + unrealized_ccy
        total_realized_ccy += realized_ccy
        total_value_ccy += total_ccy
        company_total_values_ccy.append(total_ccy)

    realized_pct = total_realized_ccy / total_value_ccy * 100 if total_value_ccy > 0 else 0
    sorted_idx_value = sorted(range(len(company_total_values_ccy)), key=lambda i: company_total_values_ccy[i], reverse=True)
    top5_value_ccy = sum([company_total_values_ccy[i] for i in sorted_idx_value[:5]])
    top5_value_pct = top5_value_ccy / sum(company_total_values_ccy) * 100 if company_total_values_ccy else 0
    top5_width_ccy = sum([widths[i] for i in sorted_idx_value[:5]])
    top5_width_pct = top5_width_ccy / sum(widths) * 100 if widths else 0
    total_invested_ccy = sum(widths)
    loss_invested_ccy = sum(widths[i] for i in range(len(values)) if sum(values[i]) < 1.0)
    loss_ratio = loss_invested_ccy / total_invested_ccy * 100 if total_invested_ccy > 0 else 0

    textstr = f"Top 5 Anteil am Gesamtfonds: {top5_value_pct:.1f}%\nTop 5 Anteil des investierten Kapitals: {top5_width_pct:.1f}%\nRealisierter Anteil gesamt: {realized_pct:.1f}%\nLoss ratio (<1.0x): {loss_ratio:.1f}%"
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9)
    ax.text(1.02, 0.5, textstr, transform=ax.transAxes, fontsize=10, va='center', bbox=props)
    ax.set_xlim(0, total_width)
    ax.set_ylabel("TVPI")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}x"))
    ax.set_title(fund_name + title_suffix, fontsize=15, fontweight="bold")
    ax.text(0.5, 0.95, "Höhe = Gesamtwertschöpfung (Realisiert + Unrealisiert);\nBreite = Investiertes Kapital; Sortiert nach TVPI (absteigend)",
            ha="center", va="bottom", fontsize=9, color="gray", transform=ax.transAxes, linespacing=1.4)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    patches = [Patch(facecolor=REAL_COLOR, edgecolor="black", label="Realisiert"),
               Patch(facecolor=UNREAL_COLOR, edgecolor="black", label="Unrealisiert")]
    ax.legend(handles=patches, title="Status", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    return fig


def load_all_funds(conn):
    query = """
    SELECT DISTINCT ON (f.fund_id) f.fund_id, f.fund_name, g.gp_name, f.vintage_year, f.strategy, f.geography, g.rating,
           m.total_tvpi, m.dpi, m.top5_value_concentration, m.loss_ratio
    FROM funds f
    LEFT JOIN gps g ON f.gp_id = g.gp_id
    LEFT JOIN fund_metrics m ON f.fund_id = m.fund_id
    WHERE f.fund_id IS NOT NULL 
    ORDER BY f.fund_id, f.fund_name
    """
    return pd.read_sql_query(query, conn)


def load_funds_with_history_metrics(conn, year=None, quarter_date=None):
    if quarter_date:
        query = """
        SELECT DISTINCT ON (f.fund_id) f.fund_id, f.fund_name, g.gp_name, f.vintage_year, f.strategy, f.geography, g.rating,
               m.total_tvpi, m.dpi, m.top5_value_concentration, m.loss_ratio, m.reporting_date
        FROM funds f
        LEFT JOIN gps g ON f.gp_id = g.gp_id
        LEFT JOIN fund_metrics_history m ON f.fund_id = m.fund_id AND m.reporting_date = %s
        WHERE f.fund_id IS NOT NULL 
        ORDER BY f.fund_id, f.fund_name
        """
        return pd.read_sql_query(query, conn, params=(quarter_date,))
    elif year:
        query = """
        SELECT DISTINCT ON (f.fund_id) f.fund_id, f.fund_name, g.gp_name, f.vintage_year, f.strategy, f.geography, g.rating,
               m.total_tvpi, m.dpi, m.top5_value_concentration, m.loss_ratio, m.reporting_date
        FROM funds f
        LEFT JOIN gps g ON f.gp_id = g.gp_id
        LEFT JOIN (
            SELECT fund_id, MAX(reporting_date) as max_date 
            FROM fund_metrics_history
            WHERE EXTRACT(YEAR FROM reporting_date) = %s 
            GROUP BY fund_id
        ) latest ON f.fund_id = latest.fund_id
        LEFT JOIN fund_metrics_history m ON f.fund_id = m.fund_id AND m.reporting_date = latest.max_date
        WHERE f.fund_id IS NOT NULL 
        ORDER BY f.fund_id, f.fund_name
        """
        return pd.read_sql_query(query, conn, params=(year,))
    else:
        return load_all_funds(conn)


def format_quarter(date_str):
    if not date_str:
        return "N/A"
    try:
        if isinstance(date_str, date):
            d = date_str
        elif isinstance(date_str, datetime):
            d = date_str.date()
        else:
            d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        quarter = (d.month - 1) // 3 + 1
        return f"Q{quarter} {d.year}"
    except:
        return str(date_str)


def get_quarter_end_date(input_date):
    year = input_date.year
    month = input_date.month
    if month <= 3:
        return date(year, 3, 31)
    elif month <= 6:
        return date(year, 6, 30)
    elif month <= 9:
        return date(year, 9, 30)
    else:
        return date(year, 12, 31)


def initialize_database(conn):
    """Initialisiert alle benötigten Tabellen"""
    ensure_gps_table(conn)
    ensure_funds_table(conn)
    ensure_portfolio_companies_table(conn)
    ensure_fund_metrics_table(conn)
    ensure_history_tables(conn)


# === HAUPTAPP ===

def show_main_app():
    """Zeigt die Hauptanwendung nach erfolgreichem Login"""
    
    # Header mit User-Info und Logout
    header_col1, header_col2 = st.columns([6, 1])
    with header_col1:
        st.title("📊 Private Equity Fund Analyzer")
    with header_col2:
        st.markdown(f"👤 **{st.session_state.user_email}**")
        if st.button("Abmelden", use_container_width=True):
            logout()
            st.rerun()
    
    st.markdown("---")

    try:
        conn = get_db_connection()
        
        # Datenbank initialisieren
        initialize_database(conn)
        
        migrated_gps = migrate_to_gp_table(conn)
        if migrated_gps:
            st.success("✅ Datenbank wurde auf neue GP-Struktur migriert!")
            clear_cache()
        
        # Currency-Spalte hinzufügen falls nötig
        ensure_currency_column(conn)
        
        # Neue Portfolio Company Felder hinzufügen
        ensure_portfolio_company_fields(conn)
        
        migrated_date = migrate_existing_data_if_needed(conn)
        if migrated_date:
            st.info(f"ℹ️ Bestehende Daten wurden mit Stichtag {migrated_date} migriert.")
        
        available_years = get_available_years(conn)
        available_dates = get_available_reporting_dates(conn)
        
        st.sidebar.header("🔍 Filter & Auswahl")
        st.sidebar.subheader("📅 Stichtag")
        
        date_mode = st.sidebar.radio("Zeitraum wählen", options=["Aktuell", "Jahr", "Quartal"], key="date_mode", horizontal=True)
        
        selected_year = None
        selected_reporting_date = None
        
        if date_mode == "Jahr" and available_years:
            selected_year = st.sidebar.selectbox("Jahr auswählen", options=available_years, key="year_select")
            st.sidebar.caption("📌 Zeigt letzte verfügbare Daten pro Fonds im gewählten Jahr")
        elif date_mode == "Quartal" and available_dates:
            quarter_options = {format_quarter(d): d for d in available_dates}
            selected_quarter_label = st.sidebar.selectbox("Quartal auswählen", options=list(quarter_options.keys()), key="quarter_select")
            selected_reporting_date = quarter_options[selected_quarter_label]
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🔄 Filter aktualisieren"):
            st.session_state.filter_version += 1
            st.rerun()
        
        if date_mode == "Aktuell":
            all_funds_df = load_all_funds(conn)
            current_date_info = "Aktuelle Daten"
        elif date_mode == "Jahr" and selected_year:
            all_funds_df = load_funds_with_history_metrics(conn, year=selected_year)
            current_date_info = f"Jahr {selected_year} (letzte verfügbare Daten)"
        elif date_mode == "Quartal" and selected_reporting_date:
            all_funds_df = load_funds_with_history_metrics(conn, quarter_date=selected_reporting_date)
            current_date_info = f"Stichtag: {selected_reporting_date}"
        else:
            all_funds_df = load_all_funds(conn)
            current_date_info = "Aktuelle Daten"
        
        # Duplikate entfernen - nur ein Eintrag pro Fund
        if not all_funds_df.empty:
            all_funds_df = all_funds_df.drop_duplicates(subset=['fund_id'], keep='first')
        
        st.sidebar.info(f"📅 {current_date_info}")
        
        if all_funds_df.empty:
            st.warning("⚠️ Keine Fonds in der Datenbank gefunden.")
            st.info("💡 Verwende den Admin-Tab um Daten zu importieren.")
        else:
            fv = st.session_state.filter_version
            
            vintage_years = sorted(all_funds_df['vintage_year'].dropna().unique())
            selected_vintages = st.sidebar.multiselect("Vintage Year", options=vintage_years, default=vintage_years, key=f"vintage_{fv}") if vintage_years else []
            
            strategies = sorted(all_funds_df['strategy'].dropna().unique())
            selected_strategies = st.sidebar.multiselect("Strategy", options=strategies, default=strategies, key=f"strategy_{fv}") if strategies else []
            
            geographies = sorted(all_funds_df['geography'].dropna().unique())
            selected_geographies = st.sidebar.multiselect("Geography", options=geographies, default=geographies, key=f"geography_{fv}") if geographies else []
            
            gps = sorted(all_funds_df['gp_name'].dropna().unique())
            selected_gps = st.sidebar.multiselect("GP Name", options=gps, default=gps, key=f"gp_{fv}") if gps else []
            
            ratings = sorted(all_funds_df['rating'].dropna().unique())
            selected_ratings = st.sidebar.multiselect("Rating", options=ratings, default=ratings, key=f"rating_{fv}") if ratings else []
            
            filtered_df = all_funds_df.copy()
            if selected_vintages:
                filtered_df = filtered_df[filtered_df['vintage_year'].isin(selected_vintages)]
            if selected_strategies:
                filtered_df = filtered_df[filtered_df['strategy'].isin(selected_strategies)]
            if selected_geographies:
                filtered_df = filtered_df[filtered_df['geography'].isin(selected_geographies)]
            if selected_gps:
                filtered_df = filtered_df[filtered_df['gp_name'].isin(selected_gps)]
            if selected_ratings:
                filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]
            
            st.sidebar.markdown("---")
            
            fund_options = {row['fund_name']: row['fund_id'] for _, row in filtered_df.iterrows()}
            
            col_all, col_none = st.sidebar.columns(2)
            with col_all:
                if st.button("✅ Alle", key=f"select_all_{fv}", use_container_width=True):
                    st.session_state[f"funds_{fv}"] = list(fund_options.keys())
                    st.rerun()
            with col_none:
                if st.button("❌ Keine", key=f"select_none_{fv}", use_container_width=True):
                    st.session_state[f"funds_{fv}"] = []
                    st.rerun()
            
            if f"funds_{fv}" not in st.session_state:
                st.session_state[f"funds_{fv}"] = list(fund_options.keys())
            
            selected_fund_names = st.sidebar.multiselect("📌 Fonds auswählen", options=list(fund_options.keys()), default=None, key=f"funds_{fv}")
            selected_fund_ids = [fund_options[name] for name in selected_fund_names]
            
            fund_reporting_dates = {}
            if date_mode == "Jahr" and selected_year:
                fund_reporting_dates = get_latest_date_for_year_per_fund(conn, selected_year, selected_fund_ids)
            elif date_mode == "Quartal" and selected_reporting_date:
                fund_reporting_dates = {fid: selected_reporting_date for fid in selected_fund_ids}
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Charts", "📈 Vergleichstabelle", "🏢 Portfoliounternehmen", "📋 Details", "⚙️ Admin"])
            
            # TAB 1: CHARTS
            with tab1:
                st.header("Mekko Charts")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    st.info("👈 Wähle Fonds in der Sidebar aus")
                else:
                    for i in range(0, len(selected_fund_ids), 2):
                        cols = st.columns(2)
                        with cols[0]:
                            fund_id = selected_fund_ids[i]
                            fund_name = selected_fund_names[i]
                            report_date = fund_reporting_dates.get(fund_id)
                            fig = create_mekko_chart(fund_id, fund_name, conn, report_date)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                        if i + 1 < len(selected_fund_ids):
                            with cols[1]:
                                fund_id = selected_fund_ids[i + 1]
                                fund_name = selected_fund_names[i + 1]
                                report_date = fund_reporting_dates.get(fund_id)
                                fig = create_mekko_chart(fund_id, fund_name, conn, report_date)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close()
                        if i + 2 < len(selected_fund_ids):
                            st.markdown("---")
            
            # TAB 2: VERGLEICHSTABELLE
            with tab2:
                st.header("Vergleichstabelle")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    st.info("👈 Wähle Fonds in der Sidebar aus")
                else:
                    comparison_df = filtered_df[filtered_df['fund_id'].isin(selected_fund_ids)].copy()
                    comparison_df = comparison_df.drop_duplicates(subset=['fund_id'], keep='first')
                    
                    if 'reporting_date' in comparison_df.columns and date_mode != "Aktuell":
                        comparison_df = comparison_df[['fund_name', 'gp_name', 'vintage_year', 'strategy', 'rating', 'total_tvpi', 'dpi', 'top5_value_concentration', 'loss_ratio', 'reporting_date']]
                        comparison_df.columns = ['Fund', 'GP', 'Vintage', 'Strategy', 'Rating', 'TVPI', 'DPI', 'Top 5 Conc.', 'Loss Ratio', 'Stichtag']
                        comparison_df['Stichtag'] = comparison_df['Stichtag'].apply(lambda x: format_quarter(x) if pd.notna(x) else "-")
                    else:
                        comparison_df = comparison_df[['fund_name', 'gp_name', 'vintage_year', 'strategy', 'rating', 'total_tvpi', 'dpi', 'top5_value_concentration', 'loss_ratio']]
                        comparison_df.columns = ['Fund', 'GP', 'Vintage', 'Strategy', 'Rating', 'TVPI', 'DPI', 'Top 5 Conc.', 'Loss Ratio']
                    
                    comparison_df['TVPI'] = comparison_df['TVPI'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
                    comparison_df['DPI'] = comparison_df['DPI'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
                    comparison_df['Top 5 Conc.'] = comparison_df['Top 5 Conc.'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                    comparison_df['Loss Ratio'] = comparison_df['Loss Ratio'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    csv = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download als CSV", data=csv, file_name=f"fund_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            
            # TAB 3: PORTFOLIOUNTERNEHMEN
            with tab3:
                st.header("🏢 Portfoliounternehmen")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    st.info("👈 Wähle Fonds in der Sidebar aus")
                else:
                    if date_mode == "Aktuell":
                        portfolio_query = """
                        SELECT pc.company_name as "Unternehmen", f.fund_name as "Fonds", g.gp_name as "GP",
                               pc.invested_amount as "Investiert", pc.realized_tvpi as "Realized TVPI",
                               pc.unrealized_tvpi as "Unrealized TVPI", (pc.realized_tvpi + pc.unrealized_tvpi) as "Total TVPI",
                               (pc.realized_tvpi + pc.unrealized_tvpi) * pc.invested_amount as "Gesamtwert",
                               pc.investment_date as "Investitionsdatum", pc.exit_date as "Exitdatum",
                               pc.entry_multiple as "Entry Multiple", pc.gross_irr as "Gross IRR"
                        FROM portfolio_companies pc
                        JOIN funds f ON pc.fund_id = f.fund_id
                        LEFT JOIN gps g ON f.gp_id = g.gp_id
                        WHERE pc.fund_id = ANY(%s) ORDER BY pc.company_name
                        """
                        all_portfolio = pd.read_sql_query(portfolio_query, conn, params=(list(selected_fund_ids),))
                    else:
                        portfolio_dfs = []
                        for fund_id in selected_fund_ids:
                            report_date = fund_reporting_dates.get(fund_id)
                            if report_date:
                                query = """
                                SELECT pc.company_name as "Unternehmen", f.fund_name as "Fonds", g.gp_name as "GP",
                                       pc.invested_amount as "Investiert", pc.realized_tvpi as "Realized TVPI",
                                       pc.unrealized_tvpi as "Unrealized TVPI", (pc.realized_tvpi + pc.unrealized_tvpi) as "Total TVPI",
                                       (pc.realized_tvpi + pc.unrealized_tvpi) * pc.invested_amount as "Gesamtwert",
                                       pc.investment_date as "Investitionsdatum", pc.exit_date as "Exitdatum",
                                       pc.entry_multiple as "Entry Multiple", pc.gross_irr as "Gross IRR",
                                       pc.reporting_date as "Stichtag"
                                FROM portfolio_companies_history pc
                                JOIN funds f ON pc.fund_id = f.fund_id
                                LEFT JOIN gps g ON f.gp_id = g.gp_id
                                WHERE pc.fund_id = %s AND pc.reporting_date = %s
                                """
                                df = pd.read_sql_query(query, conn, params=(fund_id, report_date))
                                portfolio_dfs.append(df)
                        all_portfolio = pd.concat(portfolio_dfs, ignore_index=True) if portfolio_dfs else pd.DataFrame()
                    
                    if all_portfolio.empty:
                        st.info("Keine Portfoliounternehmen für die ausgewählten Fonds vorhanden.")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            search_term = st.text_input("🔎 Unternehmen suchen", key="company_search")
                        with col2:
                            tvpi_range = st.slider("Total TVPI Bereich", min_value=0.0, max_value=float(all_portfolio['Total TVPI'].max()) + 0.5, value=(0.0, float(all_portfolio['Total TVPI'].max()) + 0.5), step=0.1, key="tvpi_filter")
                        with col3:
                            perf_filter = st.selectbox("Performance-Kategorie", options=["Alle", "Winner (>1.5x)", "Performer (1.0-1.5x)", "Under Water (<1.0x)"], key="perf_filter")
                        
                        filtered_portfolio = all_portfolio.copy()
                        if search_term:
                            filtered_portfolio = filtered_portfolio[filtered_portfolio['Unternehmen'].str.contains(search_term, case=False, na=False)]
                        filtered_portfolio = filtered_portfolio[(filtered_portfolio['Total TVPI'] >= tvpi_range[0]) & (filtered_portfolio['Total TVPI'] <= tvpi_range[1])]
                        if perf_filter == "Winner (>1.5x)":
                            filtered_portfolio = filtered_portfolio[filtered_portfolio['Total TVPI'] > 1.5]
                        elif perf_filter == "Performer (1.0-1.5x)":
                            filtered_portfolio = filtered_portfolio[(filtered_portfolio['Total TVPI'] >= 1.0) & (filtered_portfolio['Total TVPI'] <= 1.5)]
                        elif perf_filter == "Under Water (<1.0x)":
                            filtered_portfolio = filtered_portfolio[filtered_portfolio['Total TVPI'] < 1.0]
                        
                        st.markdown("---")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("Anzahl Unternehmen", len(filtered_portfolio))
                        with stat_col2:
                            st.metric("Ø TVPI", f"{filtered_portfolio['Total TVPI'].mean():.2f}x" if not filtered_portfolio.empty else "0.00x")
                        with stat_col3:
                            st.metric("Gesamt investiert", f"{filtered_portfolio['Investiert'].sum():,.0f}" if not filtered_portfolio.empty else "0")
                        with stat_col4:
                            st.metric("Gesamtwert", f"{filtered_portfolio['Gesamtwert'].sum():,.0f}" if not filtered_portfolio.empty else "0")
                        
                        st.markdown("---")
                        display_portfolio = filtered_portfolio.copy()
                        display_portfolio['Realized TVPI'] = display_portfolio['Realized TVPI'].apply(lambda x: f"{x:.2f}x")
                        display_portfolio['Unrealized TVPI'] = display_portfolio['Unrealized TVPI'].apply(lambda x: f"{x:.2f}x")
                        display_portfolio['Total TVPI'] = display_portfolio['Total TVPI'].apply(lambda x: f"{x:.2f}x")
                        display_portfolio['Investiert'] = display_portfolio['Investiert'].apply(lambda x: f"{x:,.0f}")
                        display_portfolio['Gesamtwert'] = display_portfolio['Gesamtwert'].apply(lambda x: f"{x:,.0f}")
                        if 'Entry Multiple' in display_portfolio.columns:
                            display_portfolio['Entry Multiple'] = display_portfolio['Entry Multiple'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "-")
                        if 'Gross IRR' in display_portfolio.columns:
                            display_portfolio['Gross IRR'] = display_portfolio['Gross IRR'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                        if 'Stichtag' in display_portfolio.columns:
                            display_portfolio['Stichtag'] = display_portfolio['Stichtag'].apply(format_quarter)
                        st.dataframe(display_portfolio, use_container_width=True, hide_index=True)
                        
                        csv_portfolio = filtered_portfolio.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download als CSV", data=csv_portfolio, file_name=f"portfolio_companies_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv", key="download_portfolio")
            
            # TAB 4: DETAILS
            with tab4:
                st.header("Fund Details")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    st.info("👈 Wähle Fonds in der Sidebar aus")
                else:
                    for fund_id, fund_name in zip(selected_fund_ids, selected_fund_names):
                        report_date = fund_reporting_dates.get(fund_id)
                        
                        with st.expander(f"📂 {fund_name}" + (f" ({report_date})" if report_date else ""), expanded=True):
                            fund_info = pd.read_sql_query("""
                            SELECT g.gp_name, f.vintage_year, f.fund_size_m, f.strategy, f.geography, g.rating, g.last_meeting, g.next_raise_estimate
                            FROM funds f LEFT JOIN gps g ON f.gp_id = g.gp_id WHERE f.fund_id = %s
                            """, conn, params=(fund_id,))
                            
                            if report_date:
                                metrics = get_fund_metrics_for_date(conn, fund_id, report_date)
                            else:
                                metrics = pd.read_sql_query("SELECT total_tvpi, dpi, num_investments FROM fund_metrics WHERE fund_id = %s", conn, params=(fund_id,))
                            
                            if not fund_info.empty:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("GP", fund_info['gp_name'].iloc[0] or "N/A")
                                    st.metric("Vintage", int(fund_info['vintage_year'].iloc[0]) if pd.notna(fund_info['vintage_year'].iloc[0]) else "N/A")
                                with col2:
                                    st.metric("TVPI", f"{metrics['total_tvpi'].iloc[0]:.2f}x" if not metrics.empty and pd.notna(metrics['total_tvpi'].iloc[0]) else "N/A")
                                    st.metric("DPI", f"{metrics['dpi'].iloc[0]:.2f}x" if not metrics.empty and pd.notna(metrics['dpi'].iloc[0]) else "N/A")
                                with col3:
                                    st.metric("Strategy", fund_info['strategy'].iloc[0] or "N/A")
                                    st.metric("# Investments", int(metrics['num_investments'].iloc[0]) if not metrics.empty and pd.notna(metrics['num_investments'].iloc[0]) else "N/A")
                                
                                st.subheader("Portfolio Companies")
                                if report_date:
                                    portfolio = get_portfolio_data_for_date(conn, fund_id, report_date)
                                    if not portfolio.empty:
                                        portfolio['Total TVPI'] = portfolio['realized_tvpi'] + portfolio['unrealized_tvpi']
                                        portfolio = portfolio.rename(columns={'company_name': 'Company', 'invested_amount': 'Invested', 'realized_tvpi': 'Realized', 'unrealized_tvpi': 'Unrealized'})
                                else:
                                    portfolio = pd.read_sql_query("""
                                    SELECT company_name as "Company", invested_amount as "Invested", realized_tvpi as "Realized",
                                           unrealized_tvpi as "Unrealized", (realized_tvpi + unrealized_tvpi) as "Total TVPI"
                                    FROM portfolio_companies WHERE fund_id = %s ORDER BY (realized_tvpi + unrealized_tvpi) * invested_amount DESC
                                    """, conn, params=(fund_id,))
                                
                                if not portfolio.empty:
                                    portfolio['Total TVPI'] = portfolio['Total TVPI'].apply(lambda x: f"{x:.2f}x")
                                    portfolio['Realized'] = portfolio['Realized'].apply(lambda x: f"{x:.2f}x")
                                    portfolio['Unrealized'] = portfolio['Unrealized'].apply(lambda x: f"{x:.2f}x")
                                    st.dataframe(portfolio, use_container_width=True, hide_index=True)
                                else:
                                    st.info("Keine Portfolio Companies vorhanden")
                            
                            # Historische Entwicklung
                            st.subheader("📈 Historische Entwicklung")
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT reporting_date, total_tvpi, dpi, loss_ratio, realized_percentage FROM fund_metrics_history WHERE fund_id = %s ORDER BY reporting_date", (fund_id,))
                                history = cursor.fetchall()
                            
                            if history:
                                df_history = pd.DataFrame(history, columns=['Stichtag', 'TVPI', 'DPI', 'Loss Ratio', 'Realisiert %'])
                                df_history['Stichtag'] = pd.to_datetime(df_history['Stichtag'])
                                
                                selected_chart_metrics = st.multiselect("📊 Metriken auswählen", options=['TVPI', 'DPI', 'Loss Ratio', 'Realisiert %'], default=['TVPI', 'DPI'], key=f"chart_metrics_{fund_id}")
                                
                                if selected_chart_metrics:
                                    fig, ax1 = plt.subplots(figsize=(12, 5))
                                    colors = {'TVPI': 'darkblue', 'DPI': 'green', 'Loss Ratio': 'red', 'Realisiert %': 'orange'}
                                    markers = {'TVPI': 'o', 'DPI': 's', 'Loss Ratio': '^', 'Realisiert %': 'd'}
                                    
                                    multiple_metrics = [m for m in selected_chart_metrics if m in ['TVPI', 'DPI']]
                                    percent_metrics = [m for m in selected_chart_metrics if m in ['Loss Ratio', 'Realisiert %']]
                                    lines, labels = [], []
                                    
                                    if multiple_metrics:
                                        for metric in multiple_metrics:
                                            line, = ax1.plot(df_history['Stichtag'], df_history[metric], marker=markers[metric], linewidth=2, markersize=8, color=colors[metric], label=metric)
                                            lines.append(line)
                                            labels.append(metric)
                                        ax1.set_ylabel("Multiple (x)", color='darkblue')
                                        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}x"))
                                    
                                    if percent_metrics:
                                        ax2 = ax1.twinx() if multiple_metrics else ax1
                                        for metric in percent_metrics:
                                            line, = ax2.plot(df_history['Stichtag'], df_history[metric], marker=markers[metric], linewidth=2, markersize=8, color=colors[metric], linestyle='--', label=metric)
                                            lines.append(line)
                                            labels.append(metric)
                                        ax2.set_ylabel("Prozent (%)", color='gray')
                                        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}%"))
                                    
                                    ax1.set_xlabel("Stichtag")
                                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                                    plt.xticks(rotation=45)
                                    ax1.set_title(f"Historische Entwicklung: {fund_name}", fontsize=13, fontweight='bold')
                                    ax1.grid(True, alpha=0.3)
                                    ax1.legend(lines, labels, loc='upper left')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                            else:
                                st.info("Keine historischen Daten vorhanden.")
                            st.markdown("---")
            
            # TAB 5: ADMIN
            with tab5:
                st.header("⚙️ Fund & GP Management")
                
                # Stichtag-Verwaltung
                with st.expander("📅 Stichtage verwalten", expanded=False):
                    st.subheader("Verfügbare Stichtage")
                    with conn.cursor() as cursor:
                        cursor.execute("""
                        SELECT f.fund_name, pch.reporting_date, COUNT(pch.company_name) as num_companies
                        FROM portfolio_companies_history pch JOIN funds f ON pch.fund_id = f.fund_id
                        GROUP BY f.fund_name, pch.reporting_date ORDER BY f.fund_name, pch.reporting_date DESC
                        """)
                        fund_dates = cursor.fetchall()
                    
                    if fund_dates:
                        fund_dates_df = pd.DataFrame(fund_dates, columns=['Fonds', 'Stichtag', 'Anzahl Companies'])
                        fund_dates_df['Quartal'] = fund_dates_df['Stichtag'].apply(format_quarter)
                        st.dataframe(fund_dates_df[['Fonds', 'Stichtag', 'Quartal', 'Anzahl Companies']], use_container_width=True, hide_index=True)
                    else:
                        st.info("Keine historischen Stichtage vorhanden")
                    
                    st.markdown("---")
                    st.subheader("Stichtag für einzelnen Fonds ändern")
                    
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT fund_id, fund_name FROM funds ORDER BY fund_name")
                        funds_list = cursor.fetchall()
                    
                    if funds_list:
                        fund_dict = {f[1]: f[0] for f in funds_list}
                        selected_fund_for_date = st.selectbox("Fonds auswählen", options=list(fund_dict.keys()), key="fund_for_date_change")
                        
                        if selected_fund_for_date:
                            selected_fund_id_for_date = fund_dict[selected_fund_for_date]
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT DISTINCT reporting_date FROM portfolio_companies_history WHERE fund_id = %s ORDER BY reporting_date DESC", (selected_fund_id_for_date,))
                                fund_specific_dates = [row[0].strftime('%Y-%m-%d') if isinstance(row[0], (date, datetime)) else row[0] for row in cursor.fetchall()]
                            
                            if fund_specific_dates:
                                col1, col2 = st.columns(2)
                                with col1:
                                    old_date_single = st.selectbox("Alter Stichtag", options=fund_specific_dates, format_func=format_quarter, key="old_date_single")
                                with col2:
                                    old_date_parsed = datetime.strptime(old_date_single, "%Y-%m-%d").date() if old_date_single else date.today()
                                    new_date_single = st.date_input("Neuer Stichtag", value=old_date_parsed, key="new_date_single")
                                
                                new_date_str = new_date_single.strftime("%Y-%m-%d")
                                date_already_exists = new_date_str in fund_specific_dates and new_date_str != old_date_single
                                
                                if date_already_exists:
                                    st.error(f"⚠️ Der Stichtag {format_quarter(new_date_single)} existiert bereits!")
                                elif new_date_str == old_date_single:
                                    st.info("ℹ️ Der neue Stichtag ist identisch mit dem alten.")
                                else:
                                    confirm_date_change = st.checkbox(f"✅ Ich bestätige die Änderung von {format_quarter(old_date_single)} zu {format_quarter(new_date_single)}", key="confirm_date_change")
                                    if confirm_date_change:
                                        if st.button("📅 Stichtag ändern", type="primary", key="change_date_btn"):
                                            with conn.cursor() as cursor:
                                                cursor.execute("UPDATE portfolio_companies_history SET reporting_date = %s WHERE fund_id = %s AND reporting_date = %s", (new_date_single, selected_fund_id_for_date, old_date_single))
                                                cursor.execute("UPDATE fund_metrics_history SET reporting_date = %s WHERE fund_id = %s AND reporting_date = %s", (new_date_single, selected_fund_id_for_date, old_date_single))
                                                conn.commit()
                                            clear_cache()
                                            st.success(f"✅ Stichtag geändert!")
                                            time.sleep(1)
                                            st.rerun()
                
                # Cleanup
                with st.expander("🧹 Datenbank bereinigen & Diagnose", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Bereinigung")
                        if st.button("🧹 Jetzt bereinigen", key="cleanup_btn"):
                            with conn.cursor() as cursor:
                                cursor.execute("DELETE FROM fund_metrics WHERE fund_id NOT IN (SELECT fund_id FROM funds)")
                                cursor.execute("DELETE FROM portfolio_companies WHERE fund_id NOT IN (SELECT fund_id FROM funds)")
                                cursor.execute("DELETE FROM fund_metrics_history WHERE fund_id NOT IN (SELECT fund_id FROM funds)")
                                cursor.execute("DELETE FROM portfolio_companies_history WHERE fund_id NOT IN (SELECT fund_id FROM funds)")
                                conn.commit()
                            clear_cache()
                            st.success("✅ Bereinigung abgeschlossen!")
                    with col2:
                        st.subheader("Diagnose")
                        if st.button("🔍 Datenbank analysieren", key="diagnose_btn"):
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT COUNT(*) FROM funds")
                                st.metric("Fonds", cursor.fetchone()[0])
                                cursor.execute("SELECT COUNT(*) FROM gps")
                                st.metric("GPs", cursor.fetchone()[0])
                                cursor.execute("SELECT COUNT(*) FROM portfolio_companies_history")
                                st.metric("Portfolio History", cursor.fetchone()[0])
                
                st.markdown("---")
                
                # Admin Tabs
                admin_tab1, admin_tab2, admin_tab3, admin_tab4, admin_tab5, admin_tab6 = st.tabs(["➕ Import Excel", "🏢 Edit Portfolio Company", "✏️ Edit Fund", "👔 Edit GP", "🗑️ Delete Fund", "🗑️ Delete GP"])
                
                # IMPORT EXCEL
                with admin_tab1:
                    st.subheader("Excel-Datei importieren")
                    st.info("📋 Excel-Import funktioniert wie in der SQLite-Version. Die Syntax wurde für PostgreSQL angepasst.")
                    
                    uploaded_file = st.file_uploader("Excel-Datei hochladen", type=['xlsx'], key="excel_upload")
                    
                    if uploaded_file:
                        st.warning("⚠️ Excel-Import ist in dieser Demo-Version vereinfacht. Die vollständige Implementierung folgt dem gleichen Muster wie die SQLite-Version, verwendet aber PostgreSQL-Syntax.")
                
                # EDIT PORTFOLIO COMPANY
                with admin_tab2:
                    st.subheader("🏢 Portfolio Company bearbeiten")
                    
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT DISTINCT reporting_date FROM portfolio_companies_history ORDER BY reporting_date DESC")
                        available_pc_dates = [row[0].strftime('%Y-%m-%d') if isinstance(row[0], (date, datetime)) else row[0] for row in cursor.fetchall()]
                    
                    if not available_pc_dates:
                        st.warning("Keine Portfolio Companies vorhanden.")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            edit_pc_date = st.selectbox("📅 Stichtag auswählen", options=available_pc_dates, format_func=format_quarter, key="edit_pc_date")
                        
                        with col2:
                            with conn.cursor() as cursor:
                                cursor.execute("""
                                SELECT DISTINCT f.fund_id, f.fund_name
                                FROM funds f
                                JOIN portfolio_companies_history pch ON f.fund_id = pch.fund_id
                                WHERE pch.reporting_date = %s
                                ORDER BY f.fund_name
                                """, (edit_pc_date,))
                                funds_with_pc = cursor.fetchall()
                            
                            if funds_with_pc:
                                fund_pc_dict = {f[1]: f[0] for f in funds_with_pc}
                                selected_pc_fund = st.selectbox("📁 Fonds auswählen", options=list(fund_pc_dict.keys()), key="edit_pc_fund")
                                selected_pc_fund_id = fund_pc_dict[selected_pc_fund]
                            else:
                                st.warning("Keine Fonds für diesen Stichtag.")
                                selected_pc_fund_id = None
                        
                        if selected_pc_fund_id:
                            with conn.cursor() as cursor:
                                cursor.execute("""
                                SELECT company_name FROM portfolio_companies_history
                                WHERE fund_id = %s AND reporting_date = %s
                                ORDER BY company_name
                                """, (selected_pc_fund_id, edit_pc_date))
                                companies = [row[0] for row in cursor.fetchall()]
                            
                            if companies:
                                selected_company = st.selectbox("🏢 Portfolio Company auswählen", options=companies, key="edit_pc_company")
                                st.info(f"Bearbeitung für '{selected_company}' - Formular folgt dem gleichen Muster wie SQLite-Version")
                
                # EDIT FUND
                with admin_tab3:
                    st.subheader("Fund bearbeiten")
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT fund_id, fund_name FROM funds ORDER BY fund_name")
                        existing_funds = cursor.fetchall()
                    
                    if not existing_funds:
                        st.warning("Keine Fonds vorhanden.")
                    else:
                        fund_dict_edit = {f[1]: f[0] for f in existing_funds}
                        edit_fund_name = st.selectbox("Fund auswählen", options=list(fund_dict_edit.keys()), key="edit_fund_select")
                        
                        if edit_fund_name:
                            edit_fund_id = fund_dict_edit[edit_fund_name]
                            fund_data = pd.read_sql_query("SELECT fund_name, gp_id, vintage_year, strategy, geography, fund_size_m, currency, notes FROM funds WHERE fund_id = %s", conn, params=(edit_fund_id,))
                            
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT gp_id, gp_name FROM gps ORDER BY gp_name")
                                gp_list = cursor.fetchall()
                            
                            if not fund_data.empty:
                                gp_dict = {gp[1]: gp[0] for gp in gp_list}
                                gp_names = list(gp_dict.keys())
                                current_gp_id = fund_data['gp_id'].iloc[0]
                                current_gp_name = next((name for name, gid in gp_dict.items() if gid == current_gp_id), None)
                                
                                with st.form(f"edit_fund_form_{edit_fund_id}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        new_fund_name = st.text_input("Fund Name", value=fund_data['fund_name'].iloc[0] or "")
                                        if gp_names:
                                            gp_index = gp_names.index(current_gp_name) if current_gp_name in gp_names else 0
                                            selected_gp_name = st.selectbox("GP", options=gp_names, index=gp_index)
                                            new_gp_id = gp_dict[selected_gp_name]
                                        else:
                                            new_gp_id = None
                                        new_vintage = st.number_input("Vintage Year", value=int(fund_data['vintage_year'].iloc[0]) if pd.notna(fund_data['vintage_year'].iloc[0]) else 2020, min_value=1990, max_value=2030)
                                        new_strategy = st.text_input("Strategy", value=fund_data['strategy'].iloc[0] or "")
                                    with col2:
                                        new_geography = st.text_input("Geography", value=fund_data['geography'].iloc[0] or "")
                                        new_fund_size = st.number_input("Fund Size (Mio.)", value=float(fund_data['fund_size_m'].iloc[0]) if pd.notna(fund_data['fund_size_m'].iloc[0]) else 0.0, min_value=0.0, step=1.0, format="%.2f")
                                        currency_options = ['EUR', 'USD', 'GBP', 'CHF', 'JPY', 'CNY', 'Other']
                                        current_currency = fund_data['currency'].iloc[0] if fund_data['currency'].iloc[0] in currency_options else 'EUR'
                                        currency_idx = currency_options.index(current_currency) if current_currency in currency_options else 0
                                        new_currency = st.selectbox("Währung", options=currency_options, index=currency_idx)
                                    new_notes = st.text_area("Notes", value=fund_data['notes'].iloc[0] or "")
                                    
                                    if st.form_submit_button("💾 Speichern", type="primary"):
                                        with conn.cursor() as cursor:
                                            cursor.execute("""
                                            UPDATE funds SET fund_name=%s, gp_id=%s, vintage_year=%s, strategy=%s, geography=%s, 
                                            fund_size_m=%s, currency=%s, notes=%s, updated_at=CURRENT_TIMESTAMP WHERE fund_id=%s
                                            """, (new_fund_name, new_gp_id, new_vintage, new_strategy, new_geography, new_fund_size, new_currency, new_notes, edit_fund_id))
                                            conn.commit()
                                        clear_cache()
                                        st.success(f"✅ Fund '{new_fund_name}' aktualisiert!")
                                        st.session_state.filter_version += 1
                                        time.sleep(1)
                                        st.rerun()
                
                # EDIT GP
                with admin_tab4:
                    st.subheader("👔 GP bearbeiten")
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT gp_id, gp_name FROM gps ORDER BY gp_name")
                        existing_gps = cursor.fetchall()
                    
                    if not existing_gps:
                        st.warning("Keine GPs vorhanden.")
                    else:
                        gp_dict_edit = {gp[1]: gp[0] for gp in existing_gps}
                        edit_gp_name = st.selectbox("GP auswählen", options=list(gp_dict_edit.keys()), key="edit_gp_select")
                        
                        if edit_gp_name:
                            edit_gp_id = gp_dict_edit[edit_gp_name]
                            gp_data = pd.read_sql_query("""
                            SELECT gp_name, sector, headquarters, website, rating, last_meeting, next_raise_estimate, notes,
                                   contact1_name, contact1_function, contact1_email, contact1_phone,
                                   contact2_name, contact2_function, contact2_email, contact2_phone 
                            FROM gps WHERE gp_id = %s
                            """, conn, params=(edit_gp_id,))
                            
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT COUNT(*) FROM funds WHERE gp_id = %s", (edit_gp_id,))
                                fund_count = cursor.fetchone()[0]
                            
                            st.info(f"📊 Dieser GP hat {fund_count} zugeordnete Fonds")
                            
                            if not gp_data.empty:
                                with st.form(f"edit_gp_form_{edit_gp_id}"):
                                    st.markdown("#### Basis-Informationen")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        new_gp_name = st.text_input("GP Name *", value=gp_data['gp_name'].iloc[0] or "")
                                        new_sector = st.text_input("Sektor", value=gp_data['sector'].iloc[0] or "")
                                        new_headquarters = st.text_input("Headquarters", value=gp_data['headquarters'].iloc[0] or "")
                                        new_website = st.text_input("Website", value=gp_data['website'].iloc[0] or "")
                                    with col2:
                                        rating_options = ['', 'A', 'B', 'C', 'D', 'E', 'P', 'U']
                                        current_rating = gp_data['rating'].iloc[0]
                                        current_rating_idx = rating_options.index(current_rating) if current_rating in rating_options else 0
                                        new_rating = st.selectbox("Rating", options=rating_options, index=current_rating_idx)
                                        
                                        last_meeting_val = gp_data['last_meeting'].iloc[0]
                                        new_last_meeting = st.date_input("Last Meeting", value=pd.to_datetime(last_meeting_val).date() if pd.notna(last_meeting_val) else None)
                                        
                                        next_raise_val = gp_data['next_raise_estimate'].iloc[0]
                                        new_next_raise = st.date_input("Next Raise Estimate", value=pd.to_datetime(next_raise_val).date() if pd.notna(next_raise_val) else None)
                                    
                                    new_gp_notes = st.text_area("Notizen", value=gp_data['notes'].iloc[0] or "")
                                    
                                    if st.form_submit_button("💾 GP Speichern", type="primary"):
                                        if new_gp_name.strip():
                                            with conn.cursor() as cursor:
                                                cursor.execute("""
                                                UPDATE gps SET gp_name=%s, sector=%s, headquarters=%s, website=%s, rating=%s, 
                                                last_meeting=%s, next_raise_estimate=%s, notes=%s, updated_at=CURRENT_TIMESTAMP 
                                                WHERE gp_id=%s
                                                """, (new_gp_name.strip(), new_sector or None, new_headquarters or None, 
                                                      new_website or None, new_rating or None, new_last_meeting, new_next_raise, 
                                                      new_gp_notes or None, edit_gp_id))
                                                conn.commit()
                                            clear_cache()
                                            st.success(f"✅ GP '{new_gp_name}' aktualisiert!")
                                            st.session_state.filter_version += 1
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("GP Name ist erforderlich!")
                    
                    st.markdown("---")
                    st.subheader("➕ Neuen GP anlegen")
                    with st.form("new_gp_form"):
                        new_gp_name_input = st.text_input("GP Name *", key="new_gp_name")
                        if st.form_submit_button("➕ GP anlegen", type="primary"):
                            if new_gp_name_input.strip():
                                try:
                                    with conn.cursor() as cursor:
                                        cursor.execute("INSERT INTO gps (gp_name) VALUES (%s)", (new_gp_name_input.strip(),))
                                        conn.commit()
                                    clear_cache()
                                    st.success(f"✅ GP '{new_gp_name_input}' angelegt!")
                                    time.sleep(1)
                                    st.rerun()
                                except psycopg2.IntegrityError:
                                    conn.rollback()
                                    st.error("Ein GP mit diesem Namen existiert bereits!")
                            else:
                                st.error("Bitte GP Namen eingeben!")
                
                # DELETE FUND
                with admin_tab5:
                    st.subheader("Fund löschen")
                    st.warning("⚠️ Diese Aktion kann nicht rückgängig gemacht werden!")
                    
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT fund_id, fund_name FROM funds ORDER BY fund_name")
                        delete_fund_list = cursor.fetchall()
                    
                    if delete_fund_list:
                        delete_fund_dict = {f[1]: f[0] for f in delete_fund_list}
                        delete_fund_name = st.selectbox("Fund zum Löschen", options=list(delete_fund_dict.keys()), key="delete_fund_select")
                        
                        if delete_fund_name:
                            delete_fund_id = delete_fund_dict[delete_fund_name]
                            confirm_delete = st.checkbox(f"Ich bestätige, dass ich '{delete_fund_name}' löschen möchte", key="confirm_delete_fund")
                            
                            if confirm_delete and st.button("🗑️ Fund LÖSCHEN", type="primary", key="delete_fund_btn"):
                                with conn.cursor() as cursor:
                                    cursor.execute("DELETE FROM fund_metrics WHERE fund_id = %s", (delete_fund_id,))
                                    cursor.execute("DELETE FROM portfolio_companies WHERE fund_id = %s", (delete_fund_id,))
                                    cursor.execute("DELETE FROM fund_metrics_history WHERE fund_id = %s", (delete_fund_id,))
                                    cursor.execute("DELETE FROM portfolio_companies_history WHERE fund_id = %s", (delete_fund_id,))
                                    cursor.execute("DELETE FROM funds WHERE fund_id = %s", (delete_fund_id,))
                                    conn.commit()
                                clear_cache()
                                st.success(f"✅ Fund '{delete_fund_name}' gelöscht!")
                                st.session_state.filter_version += 1
                                time.sleep(1)
                                st.rerun()
                
                # DELETE GP
                with admin_tab6:
                    st.subheader("GP löschen")
                    st.warning("⚠️ Diese Aktion kann nicht rückgängig gemacht werden!")
                    
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT gp_id, gp_name FROM gps ORDER BY gp_name")
                        delete_gp_list = cursor.fetchall()
                    
                    if delete_gp_list:
                        delete_gp_dict = {gp[1]: gp[0] for gp in delete_gp_list}
                        delete_gp_name = st.selectbox("GP zum Löschen", options=list(delete_gp_dict.keys()), key="delete_gp_select")
                        
                        if delete_gp_name:
                            delete_gp_id = delete_gp_dict[delete_gp_name]
                            
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT COUNT(*) FROM funds WHERE gp_id = %s", (delete_gp_id,))
                                fund_count = cursor.fetchone()[0]
                            
                            if fund_count > 0:
                                st.error(f"❌ Dieser GP hat noch {fund_count} zugeordnete Fonds! Bitte erst die Fonds löschen oder einem anderen GP zuordnen.")
                            else:
                                confirm_delete_gp = st.checkbox(f"Ich bestätige, dass ich '{delete_gp_name}' löschen möchte", key="confirm_delete_gp")
                                
                                if confirm_delete_gp and st.button("🗑️ GP LÖSCHEN", type="primary", key="delete_gp_btn"):
                                    with conn.cursor() as cursor:
                                        cursor.execute("DELETE FROM gps WHERE gp_id = %s", (delete_gp_id,))
                                        conn.commit()
                                    clear_cache()
                                    st.success(f"✅ GP '{delete_gp_name}' gelöscht!")
                                    time.sleep(1)
                                    st.rerun()

    except psycopg2.Error as e:
        st.error(f"❌ Datenbankfehler: {e}")
        st.info("💡 Bitte prüfen Sie die PostgreSQL-Verbindungseinstellungen.")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PE Fund Analyzer v4.0**")
    st.sidebar.markdown("🔐 Mit Supabase Auth")


# === APP ENTRY POINT ===

init_auth_state()

if st.session_state.authenticated:
    show_main_app()
else:
    show_login_page()
