from sqlalchemy.engine import Engine


def ensure_user_preferred_language(engine: Engine):
    """Add preferred_language column to users table if it does not exist (SQLite only)."""
    if not engine.url.get_backend_name().startswith("sqlite"):
        return
    with engine.connect() as conn:
        res = conn.exec_driver_sql("PRAGMA table_info(users);")
        cols = [r[1] for r in res.fetchall()]
        if "preferred_language" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN preferred_language VARCHAR(10) NOT NULL DEFAULT 'en';"
            )


def ensure_user_is_active(engine: Engine):
    if not engine.url.get_backend_name().startswith("sqlite"):
        return
    with engine.connect() as conn:
        res = conn.exec_driver_sql("PRAGMA table_info(users);")
        cols = [r[1] for r in res.fetchall()]
        if "is_active" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1;"
            )

def run_startup_migrations(engine: Engine):
    ensure_user_preferred_language(engine)
    ensure_user_is_active(engine)
