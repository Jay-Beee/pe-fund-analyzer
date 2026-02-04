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
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
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
            
            # Rolle aus user_metadata auslesen (Default: 'user')
            user_metadata = data['user'].get('user_metadata', {})
            st.session_state.user_role = user_metadata.get('role', 'user')
            
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
    st.session_state.user_role = None
    st.session_state.access_token = None

def is_admin() -> bool:
    """Prüft ob der aktuelle User Admin-Rechte hat"""
    return st.session_state.get('user_role') == 'admin'

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
            submit = st.form_submit_button("Anmelden", width='stretch')
            
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
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

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


def ensure_placement_agents_table(conn):
    """Erstellt die Placement Agents-Tabelle falls nicht vorhanden"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS placement_agents (
            pa_id SERIAL PRIMARY KEY,
            pa_name TEXT UNIQUE NOT NULL,
            headquarters TEXT,
            website TEXT,
            rating TEXT,
            last_meeting DATE,
            contact1_name TEXT,
            contact1_function TEXT,
            contact1_email TEXT,
            contact1_phone TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


def ensure_placement_agent_contact_fields(conn):
    """Fügt Kontaktfelder zur Placement Agents Tabelle hinzu falls nicht vorhanden"""
    contact_fields = [
        ('contact1_name', 'TEXT'),
        ('contact1_function', 'TEXT'),
        ('contact1_email', 'TEXT'),
        ('contact1_phone', 'TEXT')
    ]
    
    with conn.cursor() as cursor:
        for field_name, field_type in contact_fields:
            if not check_column_exists(conn, 'placement_agents', field_name):
                cursor.execute(f"ALTER TABLE placement_agents ADD COLUMN {field_name} {field_type}")
                conn.commit()
        
        # Entferne sector falls vorhanden (nicht mehr benötigt)
        if check_column_exists(conn, 'placement_agents', 'sector'):
            cursor.execute("ALTER TABLE placement_agents DROP COLUMN sector")
            conn.commit()


def ensure_funds_table(conn):
    """Erstellt die Funds-Tabelle falls nicht vorhanden"""
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS funds (
            fund_id SERIAL PRIMARY KEY,
            fund_name TEXT NOT NULL,
            gp_id INTEGER REFERENCES gps(gp_id),
            placement_agent_id INTEGER REFERENCES placement_agents(pa_id),
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


def ensure_placement_agent_column(conn):
    """Fügt die Placement Agent Spalte zu Funds hinzu falls nicht vorhanden"""
    with conn.cursor() as cursor:
        if not check_column_exists(conn, 'funds', 'placement_agent_id'):
            cursor.execute("ALTER TABLE funds ADD COLUMN placement_agent_id INTEGER REFERENCES placement_agents(pa_id)")
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


def get_or_create_placement_agent(conn, pa_name):
    """Holt oder erstellt einen Placement Agent anhand des Namens"""
    if not pa_name or pa_name.strip() == '':
        return None
    
    with conn.cursor() as cursor:
        cursor.execute("SELECT pa_id FROM placement_agents WHERE pa_name = %s", (pa_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        cursor.execute("INSERT INTO placement_agents (pa_name) VALUES (%s) RETURNING pa_id", (pa_name,))
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
           f.currency, pa.pa_name, m.total_tvpi, m.dpi, m.top5_value_concentration, m.loss_ratio
    FROM funds f
    LEFT JOIN gps g ON f.gp_id = g.gp_id
    LEFT JOIN placement_agents pa ON f.placement_agent_id = pa.pa_id
    LEFT JOIN fund_metrics m ON f.fund_id = m.fund_id
    WHERE f.fund_id IS NOT NULL 
    ORDER BY f.fund_id, f.fund_name
    """
    return pd.read_sql_query(query, conn)


def load_funds_with_history_metrics(conn, year=None, quarter_date=None):
    if quarter_date:
        query = """
        SELECT DISTINCT ON (f.fund_id) f.fund_id, f.fund_name, g.gp_name, f.vintage_year, f.strategy, f.geography, g.rating,
               f.currency, pa.pa_name, m.total_tvpi, m.dpi, m.top5_value_concentration, m.loss_ratio, m.reporting_date
        FROM funds f
        LEFT JOIN gps g ON f.gp_id = g.gp_id
        LEFT JOIN placement_agents pa ON f.placement_agent_id = pa.pa_id
        LEFT JOIN fund_metrics_history m ON f.fund_id = m.fund_id AND m.reporting_date = %s
        WHERE f.fund_id IS NOT NULL 
        ORDER BY f.fund_id, f.fund_name
        """
        return pd.read_sql_query(query, conn, params=(quarter_date,))
    elif year:
        query = """
        SELECT DISTINCT ON (f.fund_id) f.fund_id, f.fund_name, g.gp_name, f.vintage_year, f.strategy, f.geography, g.rating,
               f.currency, pa.pa_name, m.total_tvpi, m.dpi, m.top5_value_concentration, m.loss_ratio, m.reporting_date
        FROM funds f
        LEFT JOIN gps g ON f.gp_id = g.gp_id
        LEFT JOIN placement_agents pa ON f.placement_agent_id = pa.pa_id
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
    ensure_placement_agents_table(conn)
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
        role_badge = "🔑 Admin" if is_admin() else "👤 User"
        st.markdown(f"{role_badge}")
        st.caption(st.session_state.user_email)
        if st.button("Abmelden", width='stretch'):
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
        
        # Placement Agent Spalte hinzufügen falls nötig
        ensure_placement_agent_column(conn)
        
        # Placement Agent Kontaktfelder hinzufügen falls nötig
        ensure_placement_agent_contact_fields(conn)
        
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
        
        if st.sidebar.button("🔄 Filter zurücksetzen"):
            st.session_state.filter_version += 1
            st.session_state.filters_applied = False
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
            selected_vintages = st.sidebar.multiselect("Vintage Year", options=vintage_years, default=[], key=f"vintage_{fv}") if vintage_years else []
            
            strategies = sorted(all_funds_df['strategy'].dropna().unique())
            selected_strategies = st.sidebar.multiselect("Strategy", options=strategies, default=[], key=f"strategy_{fv}") if strategies else []
            
            geographies = sorted(all_funds_df['geography'].dropna().unique())
            selected_geographies = st.sidebar.multiselect("Geography", options=geographies, default=[], key=f"geography_{fv}") if geographies else []
            
            gps = sorted(all_funds_df['gp_name'].dropna().unique())
            selected_gps = st.sidebar.multiselect("GP Name", options=gps, default=[], key=f"gp_{fv}") if gps else []
            
            placement_agents = sorted([pa for pa in all_funds_df['pa_name'].dropna().unique() if pa])
            if placement_agents:
                pa_options = ["(Alle)"] + placement_agents + ["(Ohne PA)"]
                selected_pas = st.sidebar.multiselect("Placement Agent", options=pa_options, default=[], key=f"pa_{fv}")
            else:
                selected_pas = []
            
            ratings = sorted(all_funds_df['rating'].dropna().unique())
            selected_ratings = st.sidebar.multiselect("Rating", options=ratings, default=[], key=f"rating_{fv}") if ratings else []
            
# Prüfen ob mindestens ein Filter gesetzt wurde
            any_filter_set = (
                len(selected_vintages) > 0 or
                len(selected_strategies) > 0 or
                len(selected_geographies) > 0 or
                len(selected_gps) > 0 or
                (len(selected_pas) > 0 and "(Alle)" not in selected_pas) or
                len(selected_ratings) > 0
            )
            
            if any_filter_set:
                st.session_state.filters_applied = True
                
                # Filter anwenden
                filtered_df = all_funds_df.copy()
                if selected_vintages:
                    filtered_df = filtered_df[filtered_df['vintage_year'].isin(selected_vintages)]
                if selected_strategies:
                    filtered_df = filtered_df[filtered_df['strategy'].isin(selected_strategies)]
                if selected_geographies:
                    filtered_df = filtered_df[filtered_df['geography'].isin(selected_geographies)]
                if selected_gps:
                    filtered_df = filtered_df[filtered_df['gp_name'].isin(selected_gps)]
                if selected_pas and "(Alle)" not in selected_pas:
                    if "(Ohne PA)" in selected_pas:
                        pa_filter = [pa for pa in selected_pas if pa != "(Ohne PA)"]
                        filtered_df = filtered_df[(filtered_df['pa_name'].isin(pa_filter)) | (filtered_df['pa_name'].isna())]
                    else:
                        filtered_df = filtered_df[filtered_df['pa_name'].isin(selected_pas)]
                if selected_ratings:
                    filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]
                
                selected_fund_ids = filtered_df['fund_id'].tolist()
                selected_fund_names = filtered_df['fund_name'].tolist()
                
                st.sidebar.success(f"✅ {len(selected_fund_ids)} Fonds gefunden")
            else:
                st.session_state.filters_applied = False
                filtered_df = pd.DataFrame()
                selected_fund_ids = []
                selected_fund_names = []
                
                st.sidebar.info("👆 Wähle mindestens einen Filter um Fonds anzuzeigen")
            
            fund_reporting_dates = {}
            if date_mode == "Jahr" and selected_year:
                fund_reporting_dates = get_latest_date_for_year_per_fund(conn, selected_year, selected_fund_ids)
            elif date_mode == "Quartal" and selected_reporting_date:
                fund_reporting_dates = {fid: selected_reporting_date for fid in selected_fund_ids}
          
            # Tabs basierend auf Rolle erstellen
            if is_admin():
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Charts", "📈 Vergleichstabelle Fonds", "🏢 Portfoliounternehmen", "📋 Fonds", "👔 GPs", "🤝 Placement Agents", "⚙️ Admin"])
            else:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Charts", "📈 Vergleichstabelle Fonds", "🏢 Portfoliounternehmen", "📋 Fonds", "👔 GPs", "🤝 Placement Agents"])
                tab7 = None  # Kein Admin-Tab für normale User

            # TAB 1: CHARTS
            with tab1:
                st.header("Mekko Charts")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    if not st.session_state.filters_applied:
                        st.info("👈 Wähle mindestens einen Filter in der Sidebar um Fonds anzuzeigen")
                    else:
                        st.warning("Keine Fonds entsprechen den gewählten Filterkriterien")
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
                                    st.pyplot(fig, bbox_inches='tight', pad_inches=0.1)
                                    plt.close()
                        if i + 2 < len(selected_fund_ids):
                            st.markdown("---")
            
            # TAB 2: VERGLEICHSTABELLE
            with tab2:
                st.header("Vergleichstabelle")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    if not st.session_state.filters_applied:
                        st.info("👈 Wähle mindestens einen Filter in der Sidebar um Fonds anzuzeigen")
                    else:
                        st.warning("Keine Fonds entsprechen den gewählten Filterkriterien")
                else:
                    comparison_df = filtered_df[filtered_df['fund_id'].isin(selected_fund_ids)].copy()
                    comparison_df = comparison_df.drop_duplicates(subset=['fund_id'], keep='first')
                    
                    if 'reporting_date' in comparison_df.columns and date_mode != "Aktuell":
                        comparison_df = comparison_df[['fund_name', 'gp_name', 'pa_name', 'vintage_year', 'strategy', 'currency', 'rating', 'total_tvpi', 'dpi', 'top5_value_concentration', 'loss_ratio', 'reporting_date']]
                        comparison_df.columns = ['Fund', 'GP', 'Placement Agent', 'Vintage', 'Strategy', 'Währung', 'Rating', 'TVPI', 'DPI', 'Top 5 Conc.', 'Loss Ratio', 'Stichtag']
                        comparison_df['Stichtag'] = comparison_df['Stichtag'].apply(lambda x: format_quarter(x) if pd.notna(x) else "-")
                    else:
                        comparison_df = comparison_df[['fund_name', 'gp_name', 'pa_name', 'vintage_year', 'strategy', 'currency', 'rating', 'total_tvpi', 'dpi', 'top5_value_concentration', 'loss_ratio']]
                        comparison_df.columns = ['Fund', 'GP', 'Placement Agent', 'Vintage', 'Strategy', 'Währung', 'Rating', 'TVPI', 'DPI', 'Top 5 Conc.', 'Loss Ratio']
                    
                    comparison_df['Placement Agent'] = comparison_df['Placement Agent'].apply(lambda x: x if pd.notna(x) else "-")
                    comparison_df['Währung'] = comparison_df['Währung'].apply(lambda x: x if pd.notna(x) else "-")
                    comparison_df['TVPI'] = comparison_df['TVPI'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
                    comparison_df['DPI'] = comparison_df['DPI'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
                    comparison_df['Top 5 Conc.'] = comparison_df['Top 5 Conc.'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                    comparison_df['Loss Ratio'] = comparison_df['Loss Ratio'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                    
                    st.dataframe(comparison_df, width='stretch', hide_index=True)
                    
                    csv = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download als CSV", data=csv, file_name=f"fund_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            
            # TAB 3: PORTFOLIOUNTERNEHMEN
            with tab3:
                st.header("🏢 Portfoliounternehmen")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    if not st.session_state.filters_applied:
                        st.info("👈 Wähle mindestens einen Filter in der Sidebar um Fonds anzuzeigen")
                    else:
                        st.warning("Keine Fonds entsprechen den gewählten Filterkriterien")
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
                        st.dataframe(display_portfolio, width='stretch', hide_index=True)
                        
                        csv_portfolio = filtered_portfolio.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download als CSV", data=csv_portfolio, file_name=f"portfolio_companies_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv", key="download_portfolio")
            
            # TAB 4: DETAILS
            with tab4:
                st.header("📋 Fonds")
                if date_mode != "Aktuell":
                    st.caption(f"📅 {current_date_info}")
                
                if not selected_fund_ids:
                    if not st.session_state.filters_applied:
                        st.info("👈 Wähle mindestens einen Filter in der Sidebar um Fonds anzuzeigen")
                    else:
                        st.warning("Keine Fonds entsprechen den gewählten Filterkriterien")
                else:
                    for fund_id, fund_name in zip(selected_fund_ids, selected_fund_names):
                        report_date = fund_reporting_dates.get(fund_id)
                        
                        with st.expander(f"📂 {fund_name}" + (f" ({report_date})" if report_date else ""), expanded=True):
                            fund_info = pd.read_sql_query("""
                            SELECT g.gp_name, f.vintage_year, f.fund_size_m, f.currency, f.strategy, f.geography, g.rating, g.last_meeting, g.next_raise_estimate, pa.pa_name, f.notes
                            FROM funds f 
                            LEFT JOIN gps g ON f.gp_id = g.gp_id 
                            LEFT JOIN placement_agents pa ON f.placement_agent_id = pa.pa_id
                            WHERE f.fund_id = %s
                            """, conn, params=(fund_id,))
                            
                            if report_date:
                                metrics = get_fund_metrics_for_date(conn, fund_id, report_date)
                            else:
                                metrics = pd.read_sql_query("SELECT total_tvpi, dpi, num_investments FROM fund_metrics WHERE fund_id = %s", conn, params=(fund_id,))
                            
                            if not fund_info.empty:
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("GP", fund_info['gp_name'].iloc[0] or "N/A")
                                    st.metric("Vintage", int(fund_info['vintage_year'].iloc[0]) if pd.notna(fund_info['vintage_year'].iloc[0]) else "N/A")
                                with col2:
                                    st.metric("TVPI", f"{metrics['total_tvpi'].iloc[0]:.2f}x" if not metrics.empty and pd.notna(metrics['total_tvpi'].iloc[0]) else "N/A")
                                    st.metric("DPI", f"{metrics['dpi'].iloc[0]:.2f}x" if not metrics.empty and pd.notna(metrics['dpi'].iloc[0]) else "N/A")
                                with col3:
                                    st.metric("Strategy", fund_info['strategy'].iloc[0] or "N/A")
                                    st.metric("# Investments", int(metrics['num_investments'].iloc[0]) if not metrics.empty and pd.notna(metrics['num_investments'].iloc[0]) else "N/A")
                                with col4:
                                    st.metric("Währung", fund_info['currency'].iloc[0] or "N/A")
                                    fund_size = fund_info['fund_size_m'].iloc[0]
                                    currency = fund_info['currency'].iloc[0] or ""
                                    st.metric("Fund Size", f"{fund_size:,.0f} Mio. {currency}" if pd.notna(fund_size) else "N/A")
                                with col5:
                                    st.metric("Placement Agent", fund_info['pa_name'].iloc[0] or "N/A")
                                
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
                                    st.dataframe(portfolio, width='stretch', hide_index=True)
                                else:
                                    st.info("Keine Portfolio Companies vorhanden")
                            
                            # Historische Entwicklung
                            st.subheader("📈 Historische Entwicklung")

                            col_chart, col_empty = st.columns([1, 1])

                            with col_chart:
                                
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
                                        st.pyplot(fig, width='content', bbox_inches='tight', pad_inches=0.1)
                                        plt.close()
                                else:
                                    st.info("Keine historischen Daten vorhanden.")
                            
                            with col_empty:
                                st.markdown("📝 Notizen")
                                notes = fund_info['notes'].iloc[0] if not fund_info.empty and pd.notna(fund_info['notes'].iloc[0]) else "Keine Notizen vorhanden"
                                st.markdown(notes, unsafe_allow_html=True)
                                            
                            st.markdown("---")
            
            # TAB 5: GPs
            with tab5:
                st.header("👔 General Partners (GPs)")
                
                # Alle GPs mit Placement Agents laden
                gp_query = """
                SELECT g.gp_id, g.gp_name, g.sector, g.headquarters, g.website, g.rating, 
                       g.last_meeting, g.next_raise_estimate,
                       g.contact1_name, g.contact1_function, g.contact1_email, g.contact1_phone,
                       g.contact2_name, g.contact2_function, g.contact2_email, g.contact2_phone,
                       COUNT(f.fund_id) as fund_count,
                       STRING_AGG(DISTINCT pa.pa_name, ', ' ORDER BY pa.pa_name) as placement_agents
                FROM gps g
                LEFT JOIN funds f ON g.gp_id = f.gp_id
                LEFT JOIN placement_agents pa ON f.placement_agent_id = pa.pa_id
                GROUP BY g.gp_id, g.gp_name, g.sector, g.headquarters, g.website, g.rating,
                         g.last_meeting, g.next_raise_estimate,
                         g.contact1_name, g.contact1_function, g.contact1_email, g.contact1_phone,
                         g.contact2_name, g.contact2_function, g.contact2_email, g.contact2_phone
                ORDER BY g.gp_name
                """
                all_gps_df = pd.read_sql_query(gp_query, conn)
                
                if all_gps_df.empty:
                    st.info("ℹ️ Keine GPs vorhanden. GPs können im Admin-Tab erstellt oder über Excel importiert werden.")
                else:
                    # Übersichtstabelle formatieren
                    display_gps = all_gps_df.copy()
                    display_gps['last_meeting'] = display_gps['last_meeting'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and x else "-")
                    display_gps['next_raise_estimate'] = display_gps['next_raise_estimate'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and x else "-")
                    display_gps['placement_agents'] = display_gps['placement_agents'].apply(lambda x: x if x else "-")
                    
                    # Spalten für Anzeige auswählen und umbenennen
                    display_columns = {
                        'gp_name': 'GP Name',
                        'sector': 'Sektor',
                        'headquarters': 'Headquarters',
                        'rating': 'Rating',
                        'last_meeting': 'Last Meeting',
                        'next_raise_estimate': 'Next Raise',
                        'contact1_name': 'Kontakt 1',
                        'contact1_email': 'E-Mail 1',
                        'fund_count': 'Anzahl Fonds',
                        'placement_agents': 'Placement Agent'
                    }
                    
                    display_df = display_gps[list(display_columns.keys())].rename(columns=display_columns)
                    display_df = display_df.fillna("-")
                    
                    st.dataframe(display_df, width='stretch', hide_index=True)
                    
                    # Export als CSV
                    csv_gps = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download GPs als CSV", 
                        data=csv_gps, 
                        file_name=f"gps_overview_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", 
                        mime="text/csv"
                    )

            # TAB 6: PLACEMENT AGENTS
            with tab6:
                st.header("🤝 Placement Agents")
                
                # Alle Placement Agents laden
                with conn.cursor() as cursor:
                    cursor.execute("""
                    SELECT pa.pa_id, pa.pa_name, pa.headquarters, pa.website, pa.rating, pa.last_meeting,
                           pa.contact1_name, pa.contact1_function, pa.contact1_email, pa.contact1_phone,
                           COUNT(f.fund_id) as fund_count,
                           STRING_AGG(f.fund_name, ', ' ORDER BY f.fund_name) as funds
                    FROM placement_agents pa
                    LEFT JOIN funds f ON pa.pa_id = f.placement_agent_id
                    GROUP BY pa.pa_id, pa.pa_name, pa.headquarters, pa.website, pa.rating, pa.last_meeting,
                             pa.contact1_name, pa.contact1_function, pa.contact1_email, pa.contact1_phone
                    ORDER BY pa.pa_name
                    """)
                    all_pas = cursor.fetchall()
                
                if not all_pas:
                    st.info("ℹ️ Keine Placement Agents vorhanden. Placement Agents können im Admin-Tab erstellt oder über Excel importiert werden.")
                else:
                    # Übersichtstabelle
                    pa_df = pd.DataFrame(all_pas, columns=['ID', 'Name', 'Headquarters', 'Website', 'Rating', 'Last Meeting', 
                                                           'Kontakt Name', 'Kontakt Funktion', 'Kontakt E-Mail', 'Kontakt Telefon',
                                                           'Anzahl Fonds', 'Zugeordnete Fonds'])
                    pa_df['Last Meeting'] = pa_df['Last Meeting'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and x else "-")
                    pa_df['Zugeordnete Fonds'] = pa_df['Zugeordnete Fonds'].apply(lambda x: x if x else "-")
                    
                    st.dataframe(
                        pa_df[['Name', 'Headquarters', 'Rating', 'Last Meeting', 'Kontakt Name', 'Anzahl Fonds', 'Zugeordnete Fonds']],
                        width='stretch',
                        hide_index=True
                    )
                    
                    st.markdown("---")
                    
                    # Detailansicht für ausgewählten PA
                    pa_names = [pa[1] for pa in all_pas]
                    selected_pa_name = st.selectbox("📋 Placement Agent Details anzeigen", options=["(Auswählen)"] + pa_names, key="pa_detail_select")
                    
                    if selected_pa_name != "(Auswählen)":
                        selected_pa = next((pa for pa in all_pas if pa[1] == selected_pa_name), None)
                        if selected_pa:
                            st.subheader(f"📋 {selected_pa_name}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Headquarters", selected_pa[2] or "N/A")
                                st.metric("Rating", selected_pa[4] or "N/A")
                            with col2:
                                st.metric("Website", selected_pa[3] or "N/A")
                                st.metric("Last Meeting", selected_pa[5].strftime('%Y-%m-%d') if selected_pa[5] else "N/A")
                            with col3:
                                st.metric("Anzahl Fonds", selected_pa[10])
                            
                            # Kontaktperson anzeigen
                            if selected_pa[6]:  # contact1_name
                                st.markdown("**Kontaktperson:**")
                                contact_info = f"**{selected_pa[6]}**"
                                if selected_pa[7]:  # function
                                    contact_info += f" - {selected_pa[7]}"
                                st.markdown(contact_info)
                                if selected_pa[8]:  # email
                                    st.markdown(f"📧 {selected_pa[8]}")
                                if selected_pa[9]:  # phone
                                    st.markdown(f"📞 {selected_pa[9]}")
                            
                            if selected_pa[11]:  # funds
                                st.markdown("**Zugeordnete Fonds:**")
                                for fund in selected_pa[11].split(', '):
                                    st.markdown(f"- {fund}")
            
            # TAB 7: ADMIN (nur für Admins sichtbar)
            if is_admin() and tab7 is not None:
                with tab7:
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
                            st.dataframe(fund_dates_df[['Fonds', 'Stichtag', 'Quartal', 'Anzahl Companies']], width='stretch', hide_index=True)
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
                    admin_tab1, admin_tab2, admin_tab3, admin_tab4, admin_tab5, admin_tab6, admin_tab7, admin_tab8 = st.tabs(["➕ Import Excel", "🏢 Edit Portfolio Company", "✏️ Edit Fund", "👔 Edit GP", "🤝 Edit Placement Agent", "🗑️ Delete Fund", "🗑️ Delete GP", "🗑️ Delete Placement Agent"])
                
                    # IMPORT EXCEL
                    with admin_tab1:
                        st.subheader("Excel-Datei importieren")
                    
                        # Format-Hilfe anzeigen
                        with st.expander("📋 Excel-Format", expanded=False):
                            st.markdown("""
                            **Zeile 1 (GP Header):**
                            ```
                            GP Name | Strategy | Rating | Sektor | Headquarters | Website | Last Meeting | Next Raise Estimate | Kontakt1 Name | Kontakt1 Funktion | Kontakt1 E-Mail | Kontakt1 Telefon | Kontakt2 Name | Kontakt2 Funktion | Kontakt2 E-Mail | Kontakt2 Telefon | PA Name | PA Rating | PA Headquarters | PA Website | PA Last Meeting | PA Kontakt1 Name | PA Kontakt1 Funktion | PA Kontakt1 E-Mail | PA Kontakt1 Telefon
                            ```
                        
                            **Zeile 2:** GP-Werte (inkl. Placement Agent Daten)
                        
                            **Zeile 3:** [Leer]
                        
                            **Zeile 4 (Fund/Portfolio Header):**
                            ```
                            Fund Name | Stichtag | Vintage Year | Fund Size | Currency | Geography | Portfolio Company | Investment Date | Exit Date | Ownership % | Investiert | Realisiert | Unrealisiert | Entry Multiple | Gross IRR
                            ```
                        
                            **Zeile 5+:** Fund- und Portfolio-Daten (mehrere Fonds möglich)
                        
                            **Hinweise:**
                            - Leere Zellen = Bestehende Daten bleiben erhalten
                            - Datumsformat: YYYY-MM-DD oder YYYY-MM
                            - Fund-Metadaten (Vintage, Size, etc.) nur bei erster Zeile pro Fund nötig
                            - Placement Agent ist optional - wenn PA Name leer, wird kein PA zugeordnet
                            """)
                    
                        uploaded_file = st.file_uploader("Excel-Datei hochladen", type=['xlsx'], key="excel_upload")
                    
                        # Session State für Import-Workflow
                        if 'import_preview' not in st.session_state:
                            st.session_state.import_preview = None
                        if 'import_data' not in st.session_state:
                            st.session_state.import_data = None
                    
                        if uploaded_file and st.button("🔍 Vorschau & Änderungen prüfen", type="secondary", key="preview_btn"):
                            try:
                                # Excel komplett einlesen
                                uploaded_file.seek(0)
                                raw_df = pd.read_excel(uploaded_file, header=None)
                            
                                # GP-Daten aus Zeile 1-2
                                gp_header = [str(h).strip() if pd.notna(h) else '' for h in raw_df.iloc[0]]
                                gp_values = raw_df.iloc[1]
                            
                                # GP-Spalten-Mapping
                                gp_col_map = {}
                                for i, header in enumerate(gp_header):
                                    header_lower = header.lower()
                                    if 'gp name' in header_lower or header_lower == 'gp':
                                        gp_col_map['gp_name'] = i
                                    elif header_lower == 'strategy' or (header_lower.startswith('strategy') and 'pa' not in header_lower):
                                        gp_col_map['strategy'] = i
                                    elif header_lower == 'rating' or (header_lower.startswith('rating') and 'pa' not in header_lower):
                                        gp_col_map['rating'] = i
                                    elif ('sektor' in header_lower or 'sector' in header_lower) and 'pa' not in header_lower:
                                        gp_col_map['sector'] = i
                                    elif ('headquarters' in header_lower or 'hq' in header_lower) and 'pa' not in header_lower:
                                        gp_col_map['headquarters'] = i
                                    elif 'website' in header_lower and 'pa' not in header_lower:
                                        gp_col_map['website'] = i
                                    elif 'last meeting' in header_lower and 'pa' not in header_lower:
                                        gp_col_map['last_meeting'] = i
                                    elif 'next raise' in header_lower:
                                        gp_col_map['next_raise_estimate'] = i
                                    elif 'kontakt1 name' in header_lower or 'kontaktperson 1 name' in header_lower:
                                        gp_col_map['contact1_name'] = i
                                    elif 'kontakt1 funktion' in header_lower or 'kontaktperson 1 funktion' in header_lower:
                                        gp_col_map['contact1_function'] = i
                                    elif 'kontakt1 e-mail' in header_lower or 'kontaktperson 1 e-mail' in header_lower:
                                        gp_col_map['contact1_email'] = i
                                    elif 'kontakt1 telefon' in header_lower or 'kontaktperson 1 telefon' in header_lower:
                                        gp_col_map['contact1_phone'] = i
                                    elif 'kontakt2 name' in header_lower or 'kontaktperson 2 name' in header_lower:
                                        gp_col_map['contact2_name'] = i
                                    elif 'kontakt2 funktion' in header_lower or 'kontaktperson 2 funktion' in header_lower:
                                        gp_col_map['contact2_function'] = i
                                    elif 'kontakt2 e-mail' in header_lower or 'kontaktperson 2 e-mail' in header_lower:
                                        gp_col_map['contact2_email'] = i
                                    elif 'kontakt2 telefon' in header_lower or 'kontaktperson 2 telefon' in header_lower:
                                        gp_col_map['contact2_phone'] = i
                                    # Placement Agent Felder
                                    elif 'pa name' in header_lower or 'placement agent name' in header_lower:
                                        gp_col_map['pa_name'] = i
                                    elif 'pa rating' in header_lower:
                                        gp_col_map['pa_rating'] = i
                                    elif 'pa headquarters' in header_lower or 'pa hq' in header_lower:
                                        gp_col_map['pa_headquarters'] = i
                                    elif 'pa website' in header_lower:
                                        gp_col_map['pa_website'] = i
                                    elif 'pa last meeting' in header_lower:
                                        gp_col_map['pa_last_meeting'] = i
                                    elif 'pa kontakt1 name' in header_lower or 'pa kontaktperson 1 name' in header_lower:
                                        gp_col_map['pa_contact1_name'] = i
                                    elif 'pa kontakt1 funktion' in header_lower or 'pa kontaktperson 1 funktion' in header_lower:
                                        gp_col_map['pa_contact1_function'] = i
                                    elif 'pa kontakt1 e-mail' in header_lower or 'pa kontaktperson 1 e-mail' in header_lower:
                                        gp_col_map['pa_contact1_email'] = i
                                    elif 'pa kontakt1 telefon' in header_lower or 'pa kontaktperson 1 telefon' in header_lower:
                                        gp_col_map['pa_contact1_phone'] = i
                            
                                # GP-Werte extrahieren
                                def get_gp_val(key):
                                    if key in gp_col_map:
                                        val = gp_values.iloc[gp_col_map[key]]
                                        if pd.notna(val) and str(val).strip() != '':
                                            return str(val).strip()
                                    return None
                            
                                gp_data = {
                                    'gp_name': get_gp_val('gp_name'),
                                    'strategy': get_gp_val('strategy'),
                                    'rating': get_gp_val('rating'),
                                    'sector': get_gp_val('sector'),
                                    'headquarters': get_gp_val('headquarters'),
                                    'website': get_gp_val('website'),
                                    'last_meeting': get_gp_val('last_meeting'),
                                    'next_raise_estimate': get_gp_val('next_raise_estimate'),
                                    'contact1_name': get_gp_val('contact1_name'),
                                    'contact1_function': get_gp_val('contact1_function'),
                                    'contact1_email': get_gp_val('contact1_email'),
                                    'contact1_phone': get_gp_val('contact1_phone'),
                                    'contact2_name': get_gp_val('contact2_name'),
                                    'contact2_function': get_gp_val('contact2_function'),
                                    'contact2_email': get_gp_val('contact2_email'),
                                    'contact2_phone': get_gp_val('contact2_phone'),
                                }
                                
                                # Placement Agent Daten extrahieren
                                pa_data = {
                                    'pa_name': get_gp_val('pa_name'),
                                    'rating': get_gp_val('pa_rating'),
                                    'headquarters': get_gp_val('pa_headquarters'),
                                    'website': get_gp_val('pa_website'),
                                    'last_meeting': get_gp_val('pa_last_meeting'),
                                    'contact1_name': get_gp_val('pa_contact1_name'),
                                    'contact1_function': get_gp_val('pa_contact1_function'),
                                    'contact1_email': get_gp_val('pa_contact1_email'),
                                    'contact1_phone': get_gp_val('pa_contact1_phone'),
                                }
                            
                                if not gp_data['gp_name']:
                                    st.error("❌ GP Name nicht gefunden in Zeile 2!")
                                    st.stop()
                            
                                # Fund/Portfolio-Daten ab Zeile 4
                                fund_header = [str(h).strip() if pd.notna(h) else '' for h in raw_df.iloc[3]]
                            
                                # Fund-Spalten-Mapping
                                fund_col_map = {}
                                for i, header in enumerate(fund_header):
                                    header_lower = header.lower()
                                    if 'fund name' in header_lower or header_lower == 'fund':
                                        fund_col_map['fund_name'] = i
                                    elif 'stichtag' in header_lower or 'reporting date' in header_lower:
                                        fund_col_map['reporting_date'] = i
                                    elif 'vintage' in header_lower:
                                        fund_col_map['vintage_year'] = i
                                    elif 'fund size' in header_lower or 'size' in header_lower:
                                        fund_col_map['fund_size_m'] = i
                                    elif 'currency' in header_lower or 'währung' in header_lower:
                                        fund_col_map['currency'] = i
                                    elif 'geography' in header_lower or 'geograph' in header_lower:
                                        fund_col_map['geography'] = i
                                    elif 'portfolio company' in header_lower or 'company' in header_lower:
                                        fund_col_map['company_name'] = i
                                    elif 'investment date' in header_lower or 'investitionsdatum' in header_lower:
                                        fund_col_map['investment_date'] = i
                                    elif 'exit date' in header_lower or 'exitdatum' in header_lower:
                                        fund_col_map['exit_date'] = i
                                    elif 'ownership' in header_lower:
                                        fund_col_map['ownership'] = i
                                    elif 'investiert' in header_lower or 'invested' in header_lower:
                                        fund_col_map['invested_amount'] = i
                                    elif 'unrealisiert' in header_lower or 'unrealized' in header_lower:
                                        fund_col_map['unrealized_tvpi'] = i
                                    elif 'realisiert' in header_lower or 'realized' in header_lower:
                                        fund_col_map['realized_tvpi'] = i
                                    elif 'entry' in header_lower and ('multiple' in header_lower or 'ebitda' in header_lower):
                                        fund_col_map['entry_multiple'] = i
                                    elif 'gross irr' in header_lower or 'irr' in header_lower:
                                        fund_col_map['gross_irr'] = i
                            
                                # Hilfsfunktion für Datumsparsen
                                def parse_date(val):
                                    if pd.isna(val) or val == '' or val is None:
                                        return None
                                    try:
                                        if isinstance(val, (datetime, date)):
                                            return val.strftime('%Y-%m-%d')
                                        val_str = str(val).strip()
                                        if len(val_str) == 7:  # YYYY-MM
                                            return f"{val_str}-01"
                                        return pd.to_datetime(val_str).strftime('%Y-%m-%d')
                                    except:
                                        return None
                            
                                # Fund/Portfolio-Daten extrahieren
                                funds_data = {}
                            
                                for row_idx in range(4, len(raw_df)):
                                    row = raw_df.iloc[row_idx]
                                
                                    fund_name = None
                                    if 'fund_name' in fund_col_map:
                                        val = row.iloc[fund_col_map['fund_name']]
                                        if pd.notna(val) and str(val).strip():
                                            fund_name = str(val).strip()
                                
                                    if not fund_name:
                                        continue
                                
                                    if fund_name not in funds_data:
                                        funds_data[fund_name] = {
                                            'metadata': {},
                                            'companies': []
                                        }
                                
                                    for field in ['vintage_year', 'fund_size_m', 'currency', 'geography', 'reporting_date']:
                                        if field in fund_col_map:
                                            val = row.iloc[fund_col_map[field]]
                                            if pd.notna(val) and str(val).strip():
                                                if field == 'vintage_year':
                                                    try:
                                                        funds_data[fund_name]['metadata'][field] = int(float(val))
                                                    except:
                                                        pass
                                                elif field == 'fund_size_m':
                                                    try:
                                                        funds_data[fund_name]['metadata'][field] = float(val)
                                                    except:
                                                        pass
                                                elif field == 'reporting_date':
                                                    funds_data[fund_name]['metadata'][field] = parse_date(val)
                                                else:
                                                    funds_data[fund_name]['metadata'][field] = str(val).strip()
                                
                                    company_name = None
                                    if 'company_name' in fund_col_map:
                                        val = row.iloc[fund_col_map['company_name']]
                                        if pd.notna(val) and str(val).strip():
                                            company_name = str(val).strip()
                                
                                    if company_name:
                                        company_data = {'company_name': company_name}
                                    
                                        for field in ['investment_date', 'exit_date', 'ownership', 'invested_amount', 
                                                     'realized_tvpi', 'unrealized_tvpi', 'entry_multiple', 'gross_irr']:
                                            if field in fund_col_map:
                                                val = row.iloc[fund_col_map[field]]
                                                if pd.notna(val) and str(val).strip() != '':
                                                    if field in ['investment_date', 'exit_date']:
                                                        company_data[field] = parse_date(val)
                                                    elif field in ['ownership', 'gross_irr']:
                                                        try:
                                                            company_data[field] = float(val) * 100
                                                        except:
                                                            pass
                                                    elif field in ['invested_amount', 'realized_tvpi', 
                                                                  'unrealized_tvpi', 'entry_multiple']:
                                                        try:
                                                            company_data[field] = float(val)
                                                        except:
                                                            pass
                                    
                                        funds_data[fund_name]['companies'].append(company_data)
                            
                                # Änderungen ermitteln
                                changes = {'gp': [], 'funds': {}, 'companies': {}}
                            
                                with conn.cursor() as cursor:
                                    # GP-Änderungen prüfen
                                    cursor.execute("SELECT * FROM gps WHERE gp_name = %s", (gp_data['gp_name'],))
                                    existing_gp = cursor.fetchone()
                                    gp_columns = [desc[0] for desc in cursor.description] if existing_gp else []
                                
                                    if existing_gp:
                                        gp_dict = dict(zip(gp_columns, existing_gp))
                                        for field, new_val in gp_data.items():
                                            if new_val is not None and field in gp_dict:
                                                old_val = gp_dict.get(field)
                                                if old_val != new_val and str(old_val) != str(new_val):
                                                    changes['gp'].append({
                                                        'field': field,
                                                        'old': old_val,
                                                        'new': new_val
                                                    })
                                
                                    # Fund- und Company-Änderungen prüfen
                                    for fund_name, fund_info in funds_data.items():
                                        reporting_date = fund_info['metadata'].get('reporting_date')
                                        if not reporting_date:
                                            st.warning(f"⚠️ Kein Stichtag für Fund '{fund_name}' - wird übersprungen")
                                            continue
                                    
                                        cursor.execute("SELECT * FROM funds WHERE fund_name = %s", (fund_name,))
                                        existing_fund = cursor.fetchone()
                                        fund_columns = [desc[0] for desc in cursor.description] if existing_fund else []
                                    
                                        changes['funds'][fund_name] = []
                                        if existing_fund:
                                            fund_dict = dict(zip(fund_columns, existing_fund))
                                            if gp_data.get('strategy'):
                                                old_strategy = fund_dict.get('strategy')
                                                new_strategy = gp_data['strategy']
                                                if old_strategy != new_strategy and str(old_strategy) != str(new_strategy):
                                                    changes['funds'][fund_name].append({
                                                        'field': 'strategy',
                                                        'old': old_strategy,
                                                        'new': new_strategy
                                                    })
                                        
                                            for field in ['vintage_year', 'fund_size_m', 'currency', 'geography']:
                                                new_val = fund_info['metadata'].get(field)
                                                if new_val is not None:
                                                    old_val = fund_dict.get(field)
                                                    if old_val != new_val and str(old_val) != str(new_val):
                                                        changes['funds'][fund_name].append({
                                                            'field': field,
                                                            'old': old_val,
                                                            'new': new_val
                                                        })
                                    
                                        changes['companies'][fund_name] = {}
                                        for company in fund_info['companies']:
                                            company_name = company['company_name']
                                        
                                            cursor.execute("""
                                            SELECT * FROM portfolio_companies_history 
                                            WHERE fund_id = (SELECT fund_id FROM funds WHERE fund_name = %s)
                                            AND company_name = %s AND reporting_date = %s
                                            """, (fund_name, company_name, reporting_date))
                                            existing_company = cursor.fetchone()
                                            company_columns = [desc[0] for desc in cursor.description] if existing_company else []
                                        
                                            changes['companies'][fund_name][company_name] = []
                                            if existing_company:
                                                company_dict = dict(zip(company_columns, existing_company))
                                                for field in ['investment_date', 'exit_date', 'ownership', 'invested_amount',
                                                             'realized_tvpi', 'unrealized_tvpi', 'entry_multiple', 'gross_irr']:
                                                    new_val = company.get(field)
                                                    if new_val is not None:
                                                        old_val = company_dict.get(field)
                                                        if old_val != new_val and str(old_val) != str(new_val):
                                                            changes['companies'][fund_name][company_name].append({
                                                                'field': field,
                                                                'old': old_val,
                                                                'new': new_val
                                                            })
                            
                                st.session_state.import_data = {
                                    'gp_data': gp_data,
                                    'pa_data': pa_data,
                                    'funds_data': funds_data,
                                    'changes': changes
                                }
                                st.session_state.import_preview = True
                                st.rerun()
                            
                            except Exception as e:
                                st.error(f"❌ Fehler beim Lesen: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                        # Vorschau anzeigen
                        if st.session_state.import_preview and st.session_state.import_data:
                            data = st.session_state.import_data
                            changes = data['changes']
                        
                            st.markdown("---")
                            st.subheader("📋 Import-Vorschau")
                        
                            st.markdown(f"**GP:** {data['gp_data']['gp_name']}")
                            if data['pa_data'].get('pa_name'):
                                st.markdown(f"**Placement Agent:** {data['pa_data']['pa_name']}")
                        
                            fund_names = list(data['funds_data'].keys())
                            total_companies = sum(len(f['companies']) for f in data['funds_data'].values())
                            st.markdown(f"**Fonds:** {len(fund_names)} | **Portfolio Companies:** {total_companies}")
                        
                            if 'selected_changes' not in st.session_state:
                                st.session_state.selected_changes = {
                                    'gp': {},
                                    'funds': {},
                                    'companies': {}
                                }
                        
                            has_changes = False
                            change_count = 0
                        
                            if changes['gp']:
                                has_changes = True
                                st.markdown("#### 🔄 GP-Änderungen")
                                for idx, ch in enumerate(changes['gp']):
                                    key = f"gp_{ch['field']}"
                                    if key not in st.session_state.selected_changes['gp']:
                                        st.session_state.selected_changes['gp'][key] = True
                                
                                    col1, col2 = st.columns([0.05, 0.95])
                                    with col1:
                                        st.session_state.selected_changes['gp'][key] = st.checkbox(
                                            "", value=st.session_state.selected_changes['gp'][key],
                                            key=f"cb_gp_{idx}", label_visibility="collapsed"
                                        )
                                    with col2:
                                        status = "✅" if st.session_state.selected_changes['gp'][key] else "⏸️"
                                        st.markdown(f"{status} **{ch['field']}:** `{ch['old']}` → `{ch['new']}`")
                                    change_count += 1
                        
                            for fund_name, fund_changes in changes['funds'].items():
                                if fund_changes:
                                    has_changes = True
                                    st.markdown(f"#### 🔄 Fund '{fund_name}' - Änderungen")
                                
                                    if fund_name not in st.session_state.selected_changes['funds']:
                                        st.session_state.selected_changes['funds'][fund_name] = {}
                                
                                    for idx, ch in enumerate(fund_changes):
                                        key = f"{fund_name}_{ch['field']}"
                                        if key not in st.session_state.selected_changes['funds'][fund_name]:
                                            st.session_state.selected_changes['funds'][fund_name][key] = True
                                    
                                        col1, col2 = st.columns([0.05, 0.95])
                                        with col1:
                                            st.session_state.selected_changes['funds'][fund_name][key] = st.checkbox(
                                                "", value=st.session_state.selected_changes['funds'][fund_name][key],
                                                key=f"cb_fund_{fund_name}_{idx}", label_visibility="collapsed"
                                            )
                                        with col2:
                                            status = "✅" if st.session_state.selected_changes['funds'][fund_name][key] else "⏸️"
                                            st.markdown(f"{status} **{ch['field']}:** `{ch['old']}` → `{ch['new']}`")
                                        change_count += 1
                        
                            for fund_name, companies in changes['companies'].items():
                                for company_name, company_changes in companies.items():
                                    if company_changes:
                                        has_changes = True
                                        st.markdown(f"#### 🔄 '{company_name}' ({fund_name})")
                                    
                                        comp_key = f"{fund_name}_{company_name}"
                                        if comp_key not in st.session_state.selected_changes['companies']:
                                            st.session_state.selected_changes['companies'][comp_key] = {}
                                    
                                        for idx, ch in enumerate(company_changes):
                                            key = f"{comp_key}_{ch['field']}"
                                            if key not in st.session_state.selected_changes['companies'][comp_key]:
                                                st.session_state.selected_changes['companies'][comp_key][key] = True
                                        
                                            col1, col2 = st.columns([0.05, 0.95])
                                            with col1:
                                                st.session_state.selected_changes['companies'][comp_key][key] = st.checkbox(
                                                    "", value=st.session_state.selected_changes['companies'][comp_key][key],
                                                    key=f"cb_comp_{comp_key}_{idx}", label_visibility="collapsed"
                                                )
                                            with col2:
                                                status = "✅" if st.session_state.selected_changes['companies'][comp_key][key] else "⏸️"
                                                st.markdown(f"{status} **{ch['field']}:** `{ch['old']}` → `{ch['new']}`")
                                            change_count += 1
                        
                            if has_changes:
                                st.markdown("---")
                                col1, col2, col3 = st.columns([1, 1, 2])
                                with col1:
                                    if st.button("✅ Alle auswählen", key="select_all_changes"):
                                        for key in st.session_state.selected_changes['gp']:
                                            st.session_state.selected_changes['gp'][key] = True
                                        for fund in st.session_state.selected_changes['funds']:
                                            for key in st.session_state.selected_changes['funds'][fund]:
                                                st.session_state.selected_changes['funds'][fund][key] = True
                                        for comp in st.session_state.selected_changes['companies']:
                                            for key in st.session_state.selected_changes['companies'][comp]:
                                                st.session_state.selected_changes['companies'][comp][key] = True
                                        st.rerun()
                                with col2:
                                    if st.button("❌ Keine auswählen", key="select_none_changes"):
                                        for key in st.session_state.selected_changes['gp']:
                                            st.session_state.selected_changes['gp'][key] = False
                                        for fund in st.session_state.selected_changes['funds']:
                                            for key in st.session_state.selected_changes['funds'][fund]:
                                                st.session_state.selected_changes['funds'][fund][key] = False
                                        for comp in st.session_state.selected_changes['companies']:
                                            for key in st.session_state.selected_changes['companies'][comp]:
                                                st.session_state.selected_changes['companies'][comp][key] = False
                                        st.rerun()
                            
                                selected_count = 0
                                for key, val in st.session_state.selected_changes['gp'].items():
                                    if val:
                                        selected_count += 1
                                for fund in st.session_state.selected_changes['funds'].values():
                                    for val in fund.values():
                                        if val:
                                            selected_count += 1
                                for comp in st.session_state.selected_changes['companies'].values():
                                    for val in comp.values():
                                        if val:
                                            selected_count += 1
                            
                                st.info(f"📊 {selected_count} von {change_count} Änderungen ausgewählt")
                        
                            if not has_changes:
                                st.info("ℹ️ Keine Änderungen an bestehenden Daten. Nur neue Einträge werden hinzugefügt.")
                        
                            # Neue Einträge zählen
                            new_funds = []
                            new_companies = {}
                        
                            with conn.cursor() as cursor:
                                for fund_name, fund_info in data['funds_data'].items():
                                    cursor.execute("SELECT fund_id FROM funds WHERE fund_name = %s", (fund_name,))
                                    if not cursor.fetchone():
                                        new_funds.append(fund_name)
                                
                                    reporting_date = fund_info['metadata'].get('reporting_date')
                                    if reporting_date:
                                        new_companies[fund_name] = []
                                        for company in fund_info['companies']:
                                            cursor.execute("""
                                            SELECT history_id FROM portfolio_companies_history 
                                            WHERE fund_id = (SELECT fund_id FROM funds WHERE fund_name = %s)
                                            AND company_name = %s AND reporting_date = %s
                                            """, (fund_name, company['company_name'], reporting_date))
                                            if not cursor.fetchone():
                                                new_companies[fund_name].append(company['company_name'])
                        
                            if new_funds:
                                st.markdown("#### ➕ Neue Fonds")
                                for f in new_funds:
                                    st.markdown(f"- {f}")
                        
                            for fund_name, companies in new_companies.items():
                                if companies:
                                    st.markdown(f"#### ➕ Neue Companies in '{fund_name}'")
                                    for c in companies:
                                        st.markdown(f"- {c}")
                        
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("❌ Abbrechen", key="cancel_import"):
                                    st.session_state.import_preview = None
                                    st.session_state.import_data = None
                                    st.session_state.selected_changes = {'gp': {}, 'funds': {}, 'companies': {}}
                                    st.rerun()
                        
                            with col2:
                                if st.button("✅ Bestätigen & Importieren", type="primary", key="confirm_import"):
                                    try:
                                        with conn.cursor() as cursor:
                                            gp_data = data['gp_data']
                                            pa_data = data['pa_data']
                                            funds_data = data['funds_data']
                                            selected = st.session_state.selected_changes
                                        
                                            # GP anlegen/aktualisieren
                                            cursor.execute("SELECT gp_id FROM gps WHERE gp_name = %s", (gp_data['gp_name'],))
                                            existing_gp = cursor.fetchone()
                                        
                                            if existing_gp:
                                                gp_id = existing_gp[0]
                                                update_fields = []
                                                update_values = []
                                                for field in ['rating', 'sector', 'headquarters', 'website',
                                                             'last_meeting', 'next_raise_estimate', 'contact1_name',
                                                             'contact1_function', 'contact1_email', 'contact1_phone',
                                                             'contact2_name', 'contact2_function', 'contact2_email', 'contact2_phone']:
                                                    if gp_data.get(field):
                                                        change_key = f"gp_{field}"
                                                        if change_key in selected['gp'] and not selected['gp'][change_key]:
                                                            continue
                                                        update_fields.append(f"{field} = %s")
                                                        update_values.append(gp_data[field])
                                            
                                                if update_fields:
                                                    update_values.append(gp_id)
                                                    cursor.execute(f"""
                                                    UPDATE gps SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                                                    WHERE gp_id = %s
                                                    """, update_values)
                                            else:
                                                cursor.execute("""
                                                INSERT INTO gps (gp_name, rating, sector, headquarters, website,
                                                                last_meeting, next_raise_estimate, contact1_name, contact1_function,
                                                                contact1_email, contact1_phone, contact2_name, contact2_function,
                                                                contact2_email, contact2_phone)
                                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                                RETURNING gp_id
                                                """, (gp_data['gp_name'], gp_data.get('rating'),
                                                     gp_data.get('sector'), gp_data.get('headquarters'), gp_data.get('website'),
                                                     gp_data.get('last_meeting'), gp_data.get('next_raise_estimate'),
                                                     gp_data.get('contact1_name'), gp_data.get('contact1_function'),
                                                     gp_data.get('contact1_email'), gp_data.get('contact1_phone'),
                                                     gp_data.get('contact2_name'), gp_data.get('contact2_function'),
                                                     gp_data.get('contact2_email'), gp_data.get('contact2_phone')))
                                                gp_id = cursor.fetchone()[0]
                                            
                                            # Placement Agent anlegen/aktualisieren (falls vorhanden)
                                            pa_id = None
                                            if pa_data.get('pa_name'):
                                                cursor.execute("SELECT pa_id FROM placement_agents WHERE pa_name = %s", (pa_data['pa_name'],))
                                                existing_pa = cursor.fetchone()
                                                
                                                if existing_pa:
                                                    pa_id = existing_pa[0]
                                                    update_fields = []
                                                    update_values = []
                                                    for field in ['rating', 'headquarters', 'website', 'last_meeting',
                                                                 'contact1_name', 'contact1_function', 'contact1_email', 'contact1_phone']:
                                                        if pa_data.get(field):
                                                            update_fields.append(f"{field} = %s")
                                                            update_values.append(pa_data[field])
                                                    
                                                    if update_fields:
                                                        update_values.append(pa_id)
                                                        cursor.execute(f"""
                                                        UPDATE placement_agents SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                                                        WHERE pa_id = %s
                                                        """, update_values)
                                                else:
                                                    cursor.execute("""
                                                    INSERT INTO placement_agents (pa_name, rating, headquarters, website, last_meeting,
                                                                                  contact1_name, contact1_function, contact1_email, contact1_phone)
                                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                                    RETURNING pa_id
                                                    """, (pa_data['pa_name'], pa_data.get('rating'),
                                                         pa_data.get('headquarters'), pa_data.get('website'), pa_data.get('last_meeting'),
                                                         pa_data.get('contact1_name'), pa_data.get('contact1_function'),
                                                         pa_data.get('contact1_email'), pa_data.get('contact1_phone')))
                                                    pa_id = cursor.fetchone()[0]
                                        
                                            imported_funds = 0
                                            imported_companies = 0
                                            updated_companies = 0
                                            skipped_changes = 0
                                        
                                            for fund_name, fund_info in funds_data.items():
                                                reporting_date = fund_info['metadata'].get('reporting_date')
                                                if not reporting_date:
                                                    continue
                                            
                                                cursor.execute("SELECT fund_id FROM funds WHERE fund_name = %s", (fund_name,))
                                                existing_fund = cursor.fetchone()
                                            
                                                if existing_fund:
                                                    fund_id = existing_fund[0]
                                                    update_fields = ['gp_id = %s']
                                                    update_values = [gp_id]
                                                    
                                                    # Placement Agent verknüpfen
                                                    if pa_id:
                                                        update_fields.append('placement_agent_id = %s')
                                                        update_values.append(pa_id)
                                                
                                                    if gp_data.get('strategy'):
                                                        change_key = f"{fund_name}_strategy"
                                                        fund_selected = selected['funds'].get(fund_name, {})
                                                        if change_key not in fund_selected or fund_selected[change_key]:
                                                            update_fields.append("strategy = %s")
                                                            update_values.append(gp_data['strategy'])
                                                        else:
                                                            skipped_changes += 1
                                                
                                                    for field in ['vintage_year', 'fund_size_m', 'currency', 'geography']:
                                                        if fund_info['metadata'].get(field):
                                                            change_key = f"{fund_name}_{field}"
                                                            fund_selected = selected['funds'].get(fund_name, {})
                                                            if change_key not in fund_selected or fund_selected[change_key]:
                                                                update_fields.append(f"{field} = %s")
                                                                update_values.append(fund_info['metadata'][field])
                                                            else:
                                                                skipped_changes += 1
                                                
                                                    update_values.append(fund_id)
                                                    cursor.execute(f"""
                                                    UPDATE funds SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                                                    WHERE fund_id = %s
                                                    """, update_values)
                                                else:
                                                    cursor.execute("""
                                                    INSERT INTO funds (fund_name, gp_id, placement_agent_id, strategy, vintage_year, fund_size_m, currency, geography)
                                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                                    RETURNING fund_id
                                                    """, (fund_name, gp_id, pa_id, gp_data.get('strategy'),
                                                         fund_info['metadata'].get('vintage_year'),
                                                         fund_info['metadata'].get('fund_size_m'), fund_info['metadata'].get('currency'),
                                                         fund_info['metadata'].get('geography')))
                                                    fund_id = cursor.fetchone()[0]
                                                    imported_funds += 1
                                            
                                                for company in fund_info['companies']:
                                                    company_name = company['company_name']
                                                
                                                    cursor.execute("""
                                                    SELECT history_id FROM portfolio_companies_history
                                                    WHERE fund_id = %s AND company_name = %s AND reporting_date = %s
                                                    """, (fund_id, company_name, reporting_date))
                                                    existing_company = cursor.fetchone()
                                                
                                                    if existing_company:
                                                        update_fields = []
                                                        update_values = []
                                                        comp_key = f"{fund_name}_{company_name}"
                                                        comp_selected = selected['companies'].get(comp_key, {})
                                                    
                                                        for field in ['investment_date', 'exit_date', 'ownership', 'invested_amount',
                                                                     'realized_tvpi', 'unrealized_tvpi', 'entry_multiple', 'gross_irr']:
                                                            if field in company and company[field] is not None:
                                                                change_key = f"{comp_key}_{field}"
                                                                if change_key not in comp_selected or comp_selected[change_key]:
                                                                    update_fields.append(f"{field} = %s")
                                                                    update_values.append(company[field])
                                                                else:
                                                                    skipped_changes += 1
                                                    
                                                        if update_fields:
                                                            update_values.extend([fund_id, company_name, reporting_date])
                                                            cursor.execute(f"""
                                                            UPDATE portfolio_companies_history
                                                            SET {', '.join(update_fields)}
                                                            WHERE fund_id = %s AND company_name = %s AND reporting_date = %s
                                                            """, update_values)
                                                            updated_companies += 1
                                                    else:
                                                        cursor.execute("""
                                                        INSERT INTO portfolio_companies_history
                                                            (fund_id, company_name, reporting_date, investment_date, exit_date,
                                                             ownership, invested_amount, realized_tvpi, unrealized_tvpi,
                                                             entry_multiple, gross_irr)
                                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                                        """, (fund_id, company_name, reporting_date,
                                                             company.get('investment_date'), company.get('exit_date'),
                                                             company.get('ownership'), company.get('invested_amount', 0),
                                                             company.get('realized_tvpi', 0), company.get('unrealized_tvpi', 0),
                                                             company.get('entry_multiple'), company.get('gross_irr')))
                                                        imported_companies += 1
                                            
                                                # Metriken berechnen
                                                cursor.execute("""
                                                SELECT company_name, invested_amount, realized_tvpi, unrealized_tvpi
                                                FROM portfolio_companies_history
                                                WHERE fund_id = %s AND reporting_date = %s
                                                """, (fund_id, reporting_date))
                                                investments = cursor.fetchall()
                                            
                                                if investments:
                                                    total_invested = sum(inv[1] for inv in investments if inv[1])
                                                    if total_invested > 0:
                                                        company_values = []
                                                        total_realized_ccy = 0
                                                        total_value_ccy = 0
                                                        loss_invested = 0
                                                    
                                                        for comp, invested, real_tvpi, unreal_tvpi in investments:
                                                            invested = invested or 0
                                                            real_tvpi = real_tvpi or 0
                                                            unreal_tvpi = unreal_tvpi or 0
                                                        
                                                            realized_ccy = real_tvpi * invested
                                                            unrealized_ccy = unreal_tvpi * invested
                                                            total_ccy = realized_ccy + unrealized_ccy
                                                        
                                                            company_values.append((comp, invested, total_ccy))
                                                            total_realized_ccy += realized_ccy
                                                            total_value_ccy += total_ccy
                                                        
                                                            if (real_tvpi + unreal_tvpi) < 1.0:
                                                                loss_invested += invested
                                                    
                                                        company_values.sort(key=lambda x: x[2], reverse=True)
                                                    
                                                        top5_value = sum(cv[2] for cv in company_values[:5])
                                                        top5_capital = sum(cv[1] for cv in company_values[:5])
                                                    
                                                        top5_value_pct = (top5_value / total_value_ccy * 100) if total_value_ccy > 0 else 0
                                                        top5_capital_pct = (top5_capital / total_invested * 100) if total_invested > 0 else 0
                                                    
                                                        calc_tvpi = total_value_ccy / total_invested if total_invested > 0 else 0
                                                        dpi = total_realized_ccy / total_invested if total_invested > 0 else 0
                                                    
                                                        realized_pct = (total_realized_ccy / total_value_ccy * 100) if total_value_ccy > 0 else 0
                                                        loss_ratio = (loss_invested / total_invested * 100) if total_invested > 0 else 0
                                                    
                                                        cursor.execute("""
                                                        INSERT INTO fund_metrics_history
                                                            (fund_id, reporting_date, total_tvpi, dpi, top5_value_concentration,
                                                             top5_capital_concentration, loss_ratio, realized_percentage, num_investments)
                                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                                        ON CONFLICT (fund_id, reporting_date) DO UPDATE SET
                                                            total_tvpi = EXCLUDED.total_tvpi,
                                                            dpi = EXCLUDED.dpi,
                                                            top5_value_concentration = EXCLUDED.top5_value_concentration,
                                                            top5_capital_concentration = EXCLUDED.top5_capital_concentration,
                                                            loss_ratio = EXCLUDED.loss_ratio,
                                                            realized_percentage = EXCLUDED.realized_percentage,
                                                            num_investments = EXCLUDED.num_investments
                                                        """, (fund_id, reporting_date, calc_tvpi, dpi, top5_value_pct,
                                                              top5_capital_pct, loss_ratio, realized_pct, len(investments)))
                                                    
                                                        cursor.execute("""
                                                        SELECT MAX(reporting_date) FROM portfolio_companies_history WHERE fund_id = %s
                                                        """, (fund_id,))
                                                        latest_date = cursor.fetchone()[0]
                                                        latest_date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
                                                    
                                                        if reporting_date == latest_date_str:
                                                            cursor.execute("DELETE FROM portfolio_companies WHERE fund_id = %s", (fund_id,))
                                                        
                                                            cursor.execute("""
                                                            INSERT INTO portfolio_companies 
                                                                (fund_id, company_name, invested_amount, realized_tvpi, unrealized_tvpi,
                                                                 investment_date, exit_date, entry_multiple, gross_irr, ownership)
                                                            SELECT fund_id, company_name, invested_amount, realized_tvpi, unrealized_tvpi,
                                                                   investment_date, exit_date, entry_multiple, gross_irr, ownership
                                                            FROM portfolio_companies_history
                                                            WHERE fund_id = %s AND reporting_date = %s
                                                            """, (fund_id, reporting_date))
                                                        
                                                            cursor.execute("""
                                                            INSERT INTO fund_metrics 
                                                                (fund_id, total_tvpi, dpi, top5_value_concentration,
                                                                 top5_capital_concentration, loss_ratio, realized_percentage, 
                                                                 num_investments, calculation_date)
                                                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                                            ON CONFLICT (fund_id) DO UPDATE SET
                                                                total_tvpi = EXCLUDED.total_tvpi,
                                                                dpi = EXCLUDED.dpi,
                                                                top5_value_concentration = EXCLUDED.top5_value_concentration,
                                                                top5_capital_concentration = EXCLUDED.top5_capital_concentration,
                                                                loss_ratio = EXCLUDED.loss_ratio,
                                                                realized_percentage = EXCLUDED.realized_percentage,
                                                                num_investments = EXCLUDED.num_investments,
                                                                calculation_date = EXCLUDED.calculation_date
                                                            """, (fund_id, calc_tvpi, dpi, top5_value_pct,
                                                                  top5_capital_pct, loss_ratio, realized_pct, 
                                                                  len(investments), reporting_date))
                                        
                                            conn.commit()
                                    
                                        clear_cache()
                                    
                                        st.session_state.import_preview = None
                                        st.session_state.import_data = None
                                        st.session_state.selected_changes = {'gp': {}, 'funds': {}, 'companies': {}}
                                    
                                        st.success(f"""✅ Import erfolgreich!
                                        - GP: {data['gp_data']['gp_name']}
                                        - Neue Fonds: {imported_funds}
                                        - Neue Companies: {imported_companies}
                                        - Aktualisierte Companies: {updated_companies}
                                        - Übersprungene Änderungen: {skipped_changes}
                                        """)
                                        st.session_state.filter_version += 1
                                        time.sleep(2)
                                        st.rerun()
                                    
                                    except Exception as e:
                                        conn.rollback()
                                        st.error(f"❌ Fehler: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())
                
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
                                
                                    # Daten der Company laden
                                    with conn.cursor() as cursor:
                                        cursor.execute("""
                                        SELECT company_name, invested_amount, realized_tvpi, unrealized_tvpi,
                                               investment_date, exit_date, entry_multiple, gross_irr, ownership
                                        FROM portfolio_companies_history
                                        WHERE fund_id = %s AND reporting_date = %s AND company_name = %s
                                        """, (selected_pc_fund_id, edit_pc_date, selected_company))
                                        pc_data = cursor.fetchone()
                                
                                    if pc_data:
                                        st.markdown("---")
                                        st.markdown(f"**Bearbeite:** {selected_company} | **Stichtag:** {format_quarter(edit_pc_date)}")
                                    
                                        with st.form(f"edit_pc_form_{selected_pc_fund_id}_{selected_company}"):
                                            col1, col2 = st.columns(2)
                                        
                                            with col1:
                                                st.markdown("##### Finanzdaten")
                                                new_invested = st.number_input(
                                                    "Investiert (Mio.)",
                                                    value=float(pc_data[1]) if pc_data[1] else 0.0,
                                                    min_value=0.0,
                                                    step=0.1,
                                                    format="%.2f"
                                                )
                                                new_realized_tvpi = st.number_input(
                                                    "Realized TVPI",
                                                    value=float(pc_data[2]) if pc_data[2] else 0.0,
                                                    min_value=0.0,
                                                    step=0.01,
                                                    format="%.2f"
                                                )
                                                new_unrealized_tvpi = st.number_input(
                                                    "Unrealized TVPI",
                                                    value=float(pc_data[3]) if pc_data[3] else 0.0,
                                                    min_value=0.0,
                                                    step=0.01,
                                                    format="%.2f"
                                                )
                                                new_entry_multiple = st.number_input(
                                                    "Entry Multiple",
                                                    value=float(pc_data[6]) if pc_data[6] else 0.0,
                                                    min_value=0.0,
                                                    step=0.1,
                                                    format="%.1f"
                                                )
                                                new_ownership = st.number_input(
                                                    "Ownership (%)",
                                                    value=float(pc_data[8]) if pc_data[8] else 0.0,
                                                    min_value=0.0,
                                                    max_value=100.0,
                                                    step=0.01,
                                                    format="%.2f"
                                                )
                                        
                                            with col2:
                                                st.markdown("##### Datums- und Renditedaten")
                                            
                                                # Investment Date - Monat/Jahr Auswahl
                                                if pc_data[4]:
                                                    try:
                                                        inv_date_parsed = pd.to_datetime(pc_data[4])
                                                        inv_month_default = inv_date_parsed.month
                                                        inv_year_default = inv_date_parsed.year
                                                    except:
                                                        inv_month_default = 1
                                                        inv_year_default = 2020
                                                else:
                                                    inv_month_default = 1
                                                    inv_year_default = 2020
                                            
                                                st.markdown("**Investitionsdatum**")
                                                inv_col1, inv_col2, inv_col3 = st.columns([2, 2, 1])
                                                with inv_col1:
                                                    inv_month = st.selectbox(
                                                        "Monat",
                                                        options=[0] + list(range(1, 13)),
                                                        index=inv_month_default if pc_data[4] else 0,
                                                        format_func=lambda x: '-' if x == 0 else ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'][x-1],
                                                        key=f"inv_month_{selected_company}"
                                                    )
                                                with inv_col2:
                                                    inv_year = st.selectbox(
                                                        "Jahr",
                                                        options=[0] + list(range(2000, 2031)),
                                                        index=(inv_year_default - 2000 + 1) if pc_data[4] and 2000 <= inv_year_default <= 2030 else 0,
                                                        format_func=lambda x: '-' if x == 0 else str(x),
                                                        key=f"inv_year_{selected_company}"
                                                    )
                                                with inv_col3:
                                                    if inv_month > 0 and inv_year > 0:
                                                        st.markdown(f"<br>✓", unsafe_allow_html=True)
                                                    else:
                                                        st.markdown(f"<br>", unsafe_allow_html=True)
                                            
                                                if inv_month > 0 and inv_year > 0:
                                                    new_investment_date = date(inv_year, inv_month, 1)
                                                else:
                                                    new_investment_date = None
                                            
                                                # Exit Date - Monat/Jahr Auswahl
                                                if pc_data[5]:
                                                    try:
                                                        exit_date_parsed = pd.to_datetime(pc_data[5])
                                                        exit_month_default = exit_date_parsed.month
                                                        exit_year_default = exit_date_parsed.year
                                                    except:
                                                        exit_month_default = 1
                                                        exit_year_default = 2020
                                                else:
                                                    exit_month_default = 1
                                                    exit_year_default = 2020
                                            
                                                st.markdown("**Exitdatum**")
                                                exit_col1, exit_col2, exit_col3 = st.columns([2, 2, 1])
                                                with exit_col1:
                                                    exit_month = st.selectbox(
                                                        "Monat",
                                                        options=[0] + list(range(1, 13)),
                                                        index=exit_month_default if pc_data[5] else 0,
                                                        format_func=lambda x: '-' if x == 0 else ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'][x-1],
                                                        key=f"exit_month_{selected_company}"
                                                    )
                                                with exit_col2:
                                                    exit_year = st.selectbox(
                                                        "Jahr",
                                                        options=[0] + list(range(2000, 2031)),
                                                        index=(exit_year_default - 2000 + 1) if pc_data[5] and 2000 <= exit_year_default <= 2030 else 0,
                                                        format_func=lambda x: '-' if x == 0 else str(x),
                                                        key=f"exit_year_{selected_company}"
                                                    )
                                                with exit_col3:
                                                    if exit_month > 0 and exit_year > 0:
                                                        st.markdown(f"<br>✓", unsafe_allow_html=True)
                                                    else:
                                                        st.markdown(f"<br>", unsafe_allow_html=True)
                                            
                                                if exit_month > 0 and exit_year > 0:
                                                    new_exit_date = date(exit_year, exit_month, 1)
                                                else:
                                                    new_exit_date = None
                                            
                                                new_gross_irr = st.number_input(
                                                    "Gross IRR (%)",
                                                    value=float(pc_data[7]) if pc_data[7] else 0.0,
                                                    step=0.1,
                                                    format="%.1f"
                                                )
                                        
                                            # Berechnung anzeigen
                                            total_tvpi = new_realized_tvpi + new_unrealized_tvpi
                                            total_value = total_tvpi * new_invested
                                            st.markdown(f"**Berechnete Werte:** Total TVPI: {total_tvpi:.2f}x | Gesamtwert: {total_value:,.2f} Mio.")
                                        
                                            submitted_pc = st.form_submit_button("💾 Portfolio Company speichern", type="primary")
                                    
                                        if submitted_pc:
                                            try:
                                                with conn.cursor() as cursor:
                                                    # History-Tabelle aktualisieren
                                                    cursor.execute("""
                                                    UPDATE portfolio_companies_history
                                                    SET invested_amount = %s, realized_tvpi = %s, unrealized_tvpi = %s,
                                                        investment_date = %s, exit_date = %s, entry_multiple = %s, gross_irr = %s, ownership = %s
                                                    WHERE fund_id = %s AND reporting_date = %s AND company_name = %s
                                                    """, (
                                                        new_invested, new_realized_tvpi, new_unrealized_tvpi,
                                                        new_investment_date, new_exit_date,
                                                        new_entry_multiple if new_entry_multiple > 0 else None,
                                                        new_gross_irr if new_gross_irr != 0 else None,
                                                        new_ownership if new_ownership > 0 else None,
                                                        selected_pc_fund_id, edit_pc_date, selected_company
                                                    ))
                                                
                                                    # Auch aktuelle Tabelle aktualisieren wenn es der neueste Stichtag ist
                                                    cursor.execute("""
                                                    SELECT MAX(reporting_date) FROM portfolio_companies_history WHERE fund_id = %s
                                                    """, (selected_pc_fund_id,))
                                                    latest_date_result = cursor.fetchone()
                                                    latest_date = latest_date_result[0].strftime('%Y-%m-%d') if latest_date_result and latest_date_result[0] else None
                                                
                                                    if edit_pc_date == latest_date:
                                                        cursor.execute("""
                                                        UPDATE portfolio_companies
                                                        SET invested_amount = %s, realized_tvpi = %s, unrealized_tvpi = %s,
                                                            investment_date = %s, exit_date = %s, entry_multiple = %s, gross_irr = %s, ownership = %s
                                                        WHERE fund_id = %s AND company_name = %s
                                                        """, (
                                                            new_invested, new_realized_tvpi, new_unrealized_tvpi,
                                                            new_investment_date, new_exit_date,
                                                            new_entry_multiple if new_entry_multiple > 0 else None,
                                                            new_gross_irr if new_gross_irr != 0 else None,
                                                            new_ownership if new_ownership > 0 else None,
                                                            selected_pc_fund_id, selected_company
                                                        ))
                                                
                                                    conn.commit()
                                                clear_cache()
                                            
                                                st.success(f"✅ '{selected_company}' für Stichtag {format_quarter(edit_pc_date)} aktualisiert!")
                                                time.sleep(1)
                                                st.rerun()
                                            
                                            except Exception as e:
                                                conn.rollback()
                                                st.error(f"❌ Fehler: {e}")
                
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
                                fund_data = pd.read_sql_query("SELECT fund_name, gp_id, placement_agent_id, vintage_year, strategy, geography, fund_size_m, currency, notes FROM funds WHERE fund_id = %s", conn, params=(edit_fund_id,))
                            
                                with conn.cursor() as cursor:
                                    cursor.execute("SELECT gp_id, gp_name FROM gps ORDER BY gp_name")
                                    gp_list = cursor.fetchall()
                                    cursor.execute("SELECT pa_id, pa_name FROM placement_agents ORDER BY pa_name")
                                    pa_list = cursor.fetchall()
                            
                                if not fund_data.empty:
                                    gp_dict = {gp[1]: gp[0] for gp in gp_list}
                                    gp_names = list(gp_dict.keys())
                                    current_gp_id = fund_data['gp_id'].iloc[0]
                                    current_gp_name = next((name for name, gid in gp_dict.items() if gid == current_gp_id), None)
                                    
                                    # Placement Agent Dictionary mit "(Kein PA)" Option
                                    pa_dict = {"(Kein PA)": None}
                                    pa_dict.update({pa[1]: pa[0] for pa in pa_list})
                                    pa_names = list(pa_dict.keys())
                                    current_pa_id = fund_data['placement_agent_id'].iloc[0]
                                    current_pa_name = next((name for name, pid in pa_dict.items() if pid == current_pa_id), "(Kein PA)")
                                
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
                                            # Placement Agent Dropdown
                                            pa_index = pa_names.index(current_pa_name) if current_pa_name in pa_names else 0
                                            selected_pa_name = st.selectbox("Placement Agent", options=pa_names, index=pa_index)
                                            new_pa_id = pa_dict[selected_pa_name]
                                            
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
                                                UPDATE funds SET fund_name=%s, gp_id=%s, placement_agent_id=%s, vintage_year=%s, strategy=%s, geography=%s, 
                                                fund_size_m=%s, currency=%s, notes=%s, updated_at=CURRENT_TIMESTAMP WHERE fund_id=%s
                                                """, (new_fund_name, new_gp_id, new_pa_id, new_vintage, new_strategy, new_geography, new_fund_size, new_currency, new_notes, edit_fund_id))
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
                                    
                                        st.markdown("#### Kontaktperson 1")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            new_c1_name = st.text_input("Name", value=gp_data['contact1_name'].iloc[0] or "", key=f"c1n_{edit_gp_id}")
                                            new_c1_func = st.text_input("Funktion", value=gp_data['contact1_function'].iloc[0] or "", key=f"c1f_{edit_gp_id}")
                                        with col2:
                                            new_c1_email = st.text_input("E-Mail", value=gp_data['contact1_email'].iloc[0] or "", key=f"c1e_{edit_gp_id}")
                                            new_c1_phone = st.text_input("Telefon", value=gp_data['contact1_phone'].iloc[0] or "", key=f"c1p_{edit_gp_id}")
                                    
                                        st.markdown("#### Kontaktperson 2")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            new_c2_name = st.text_input("Name", value=gp_data['contact2_name'].iloc[0] or "", key=f"c2n_{edit_gp_id}")
                                            new_c2_func = st.text_input("Funktion", value=gp_data['contact2_function'].iloc[0] or "", key=f"c2f_{edit_gp_id}")
                                        with col2:
                                            new_c2_email = st.text_input("E-Mail", value=gp_data['contact2_email'].iloc[0] or "", key=f"c2e_{edit_gp_id}")
                                            new_c2_phone = st.text_input("Telefon", value=gp_data['contact2_phone'].iloc[0] or "", key=f"c2p_{edit_gp_id}")
                                    
                                        new_gp_notes = st.text_area("Notizen", value=gp_data['notes'].iloc[0] or "")
                                    
                                        if st.form_submit_button("💾 GP Speichern", type="primary"):
                                            if new_gp_name.strip():
                                                with conn.cursor() as cursor:
                                                    cursor.execute("""
                                                    UPDATE gps SET gp_name=%s, sector=%s, headquarters=%s, website=%s, rating=%s, 
                                                    last_meeting=%s, next_raise_estimate=%s, notes=%s,
                                                    contact1_name=%s, contact1_function=%s, contact1_email=%s, contact1_phone=%s,
                                                    contact2_name=%s, contact2_function=%s, contact2_email=%s, contact2_phone=%s,
                                                    updated_at=CURRENT_TIMESTAMP 
                                                    WHERE gp_id=%s
                                                    """, (new_gp_name.strip(), new_sector or None, new_headquarters or None, 
                                                          new_website or None, new_rating or None, new_last_meeting, new_next_raise, 
                                                          new_gp_notes or None,
                                                          new_c1_name or None, new_c1_func or None, new_c1_email or None, new_c1_phone or None,
                                                          new_c2_name or None, new_c2_func or None, new_c2_email or None, new_c2_phone or None,
                                                          edit_gp_id))
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
                
                    # EDIT PLACEMENT AGENT
                    with admin_tab5:
                        st.subheader("🤝 Placement Agent bearbeiten")
                        
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT pa_id, pa_name FROM placement_agents ORDER BY pa_name")
                            pa_list = cursor.fetchall()
                        
                        # Immer "(Neu erstellen)" als Option anbieten
                        pa_dict = {pa[1]: pa[0] for pa in pa_list} if pa_list else {}
                        edit_pa_name = st.selectbox("Placement Agent auswählen", options=["(Neu erstellen)"] + list(pa_dict.keys()), key="edit_pa_select")
                        
                        if edit_pa_name != "(Neu erstellen)":
                            edit_pa_id = pa_dict[edit_pa_name]
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT * FROM placement_agents WHERE pa_id = %s", (edit_pa_id,))
                                pa_data_row = cursor.fetchone()
                                pa_columns = [desc[0] for desc in cursor.description]
                                pa_info = dict(zip(pa_columns, pa_data_row))
                        else:
                            edit_pa_id = None
                            pa_info = {}
                        
                        # Formular mit dynamischem Key basierend auf ausgewähltem PA
                        form_key = f"edit_pa_form_{edit_pa_id if edit_pa_id else 'new'}"
                        
                        with st.form(form_key):
                            col1, col2 = st.columns(2)
                            with col1:
                                new_pa_name = st.text_input("PA Name *", value=pa_info.get('pa_name', ''))
                                new_pa_rating = st.text_input("Rating", value=pa_info.get('rating') or '')
                                new_pa_hq = st.text_input("Headquarters", value=pa_info.get('headquarters') or '')
                                new_pa_website = st.text_input("Website", value=pa_info.get('website') or '')
                                pa_last_meeting = pa_info.get('last_meeting')
                                new_pa_last_meeting = st.date_input("Last Meeting", value=pa_last_meeting if pa_last_meeting else None)
                            with col2:
                                st.markdown("**Kontaktperson**")
                                new_pa_contact1_name = st.text_input("Kontakt Name", value=pa_info.get('contact1_name') or '')
                                new_pa_contact1_function = st.text_input("Kontakt Funktion", value=pa_info.get('contact1_function') or '')
                                new_pa_contact1_email = st.text_input("Kontakt E-Mail", value=pa_info.get('contact1_email') or '')
                                new_pa_contact1_phone = st.text_input("Kontakt Telefon", value=pa_info.get('contact1_phone') or '')
                            
                            if st.form_submit_button("💾 Placement Agent speichern", type="primary"):
                                if new_pa_name:
                                    with conn.cursor() as cursor:
                                        if edit_pa_id:
                                            cursor.execute("""
                                            UPDATE placement_agents SET pa_name = %s, rating = %s, headquarters = %s, website = %s, last_meeting = %s,
                                                   contact1_name = %s, contact1_function = %s, contact1_email = %s, contact1_phone = %s, updated_at = CURRENT_TIMESTAMP
                                            WHERE pa_id = %s
                                            """, (new_pa_name, new_pa_rating or None, new_pa_hq or None, new_pa_website or None, new_pa_last_meeting,
                                                 new_pa_contact1_name or None, new_pa_contact1_function or None, new_pa_contact1_email or None, new_pa_contact1_phone or None, edit_pa_id))
                                            conn.commit()
                                            clear_cache()
                                            st.success(f"✅ Placement Agent '{new_pa_name}' aktualisiert!")
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            cursor.execute("SELECT pa_id FROM placement_agents WHERE pa_name = %s", (new_pa_name,))
                                            if cursor.fetchone():
                                                st.error("Ein Placement Agent mit diesem Namen existiert bereits!")
                                            else:
                                                cursor.execute("""
                                                INSERT INTO placement_agents (pa_name, rating, headquarters, website, last_meeting,
                                                                              contact1_name, contact1_function, contact1_email, contact1_phone)
                                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                                """, (new_pa_name, new_pa_rating or None, new_pa_hq or None, new_pa_website or None, new_pa_last_meeting,
                                                     new_pa_contact1_name or None, new_pa_contact1_function or None, new_pa_contact1_email or None, new_pa_contact1_phone or None))
                                                conn.commit()
                                                clear_cache()
                                                st.success(f"✅ Placement Agent '{new_pa_name}' erstellt!")
                                                time.sleep(1)
                                                st.rerun()
                                else:
                                    st.error("Bitte PA Namen eingeben!")
                
                    # DELETE FUND
                    with admin_tab6:
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
                    with admin_tab7:
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
                    
                    # DELETE PLACEMENT AGENT
                    with admin_tab8:
                        st.subheader("Placement Agent löschen")
                        st.warning("⚠️ Diese Aktion kann nicht rückgängig gemacht werden!")
                    
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT pa_id, pa_name FROM placement_agents ORDER BY pa_name")
                            delete_pa_list = cursor.fetchall()
                    
                        if delete_pa_list:
                            delete_pa_dict = {pa[1]: pa[0] for pa in delete_pa_list}
                            delete_pa_name = st.selectbox("Placement Agent zum Löschen", options=list(delete_pa_dict.keys()), key="delete_pa_select")
                        
                            if delete_pa_name:
                                delete_pa_id = delete_pa_dict[delete_pa_name]
                            
                                with conn.cursor() as cursor:
                                    cursor.execute("SELECT COUNT(*) FROM funds WHERE placement_agent_id = %s", (delete_pa_id,))
                                    pa_fund_count = cursor.fetchone()[0]
                            
                                if pa_fund_count > 0:
                                    st.error(f"❌ Dieser Placement Agent hat noch {pa_fund_count} zugeordnete Fonds! Bitte erst die Fonds einem anderen PA zuordnen oder die Zuordnung entfernen.")
                                else:
                                    confirm_delete_pa = st.checkbox(f"Ich bestätige, dass ich '{delete_pa_name}' löschen möchte", key="confirm_delete_pa")
                                
                                    if confirm_delete_pa and st.button("🗑️ Placement Agent LÖSCHEN", type="primary", key="delete_pa_btn"):
                                        with conn.cursor() as cursor:
                                            cursor.execute("DELETE FROM placement_agents WHERE pa_id = %s", (delete_pa_id,))
                                            conn.commit()
                                        clear_cache()
                                        st.success(f"✅ Placement Agent '{delete_pa_name}' gelöscht!")
                                        time.sleep(1)
                                        st.rerun()
                        else:
                            st.info("Keine Placement Agents vorhanden.")

    except psycopg2.Error as e:
        st.error(f"❌ Datenbankfehler: {e}")
        st.info("💡 Bitte prüfen Sie die PostgreSQL-Verbindungseinstellungen.")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PE Fund Analyzer v4.2**")
    st.sidebar.markdown("🔐 Mit Supabase Auth & Rollen")


# === APP ENTRY POINT ===

init_auth_state()

if st.session_state.authenticated:
    show_main_app()
else:
    show_login_page()
