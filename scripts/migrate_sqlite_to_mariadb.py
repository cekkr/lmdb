#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from db_slm.settings import load_settings


@dataclass
class TableDump:
    name: str
    columns: List[str]
    rows: List[Sequence]


@dataclass
class MigrationBundle:
    create_statements: List[str]
    index_statements: List[str]
    tables: List[TableDump]


def convert_sqlite_create(sql: str) -> str:
    converted = sql.replace('"', "`")
    conversions = [
        (r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT", "BIGINT AUTO_INCREMENT PRIMARY KEY"),
        (r"INTEGER\s+PRIMARY\s+KEY", "BIGINT PRIMARY KEY"),
        (r"REAL", "DOUBLE"),
        (r"BLOB", "LONGBLOB"),
    ]
    for pattern, replacement in conversions:
        converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
    converted = converted.replace("WITHOUT ROWID", "")
    converted = converted.rstrip().rstrip(";")
    return f"{converted} ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"


def convert_sqlite_index(sql: str) -> str:
    converted = sql.replace('"', "`").rstrip()
    if not converted.endswith(";"):
        converted += ";"
    return converted


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        blob = bytes(value)
        return f"0x{blob.hex()}"
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def build_bundle(sqlite_path: Path) -> MigrationBundle:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    create_statements: list[str] = []
    tables: list[TableDump] = []
    cursor = conn.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    for row in cursor.fetchall():
        table_name = row["name"]
        sql = row["sql"]
        if not sql:
            continue
        create_statements.append(convert_sqlite_create(sql))
        col_rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = [col_row["name"] for col_row in col_rows]
        data_rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
        formatted = []
        for data in data_rows:
            formatted.append(tuple(data[col] for col in columns))
        tables.append(TableDump(name=table_name, columns=columns, rows=formatted))

    index_statements: list[str] = []
    idx_cursor = conn.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type='index'
          AND sql IS NOT NULL
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    for idx_row in idx_cursor.fetchall():
        index_sql = idx_row["sql"]
        if index_sql:
            index_statements.append(convert_sqlite_index(index_sql))

    conn.close()
    return MigrationBundle(create_statements=create_statements, index_statements=index_statements, tables=tables)


def write_sql(bundle: MigrationBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("-- Auto-generated MariaDB migration for DB-SLM.\n")
        handle.write("SET FOREIGN_KEY_CHECKS=0;\n\n")
        for statement in bundle.create_statements:
            handle.write(f"{statement}\n\n")
        for index in bundle.index_statements:
            handle.write(f"{index}\n")
        handle.write("\n")
        for table in bundle.tables:
            if not table.rows:
                continue
            column_list = ", ".join(f"`{col}`" for col in table.columns)
            for group in chunks(table.rows, 250):
                values = ", ".join(
                    "(" + ", ".join(sql_literal(value) for value in row) + ")"
                    for row in group
                )
                handle.write(f"INSERT INTO `{table.name}` ({column_list}) VALUES {values};\n")
        handle.write("\nSET FOREIGN_KEY_CHECKS=1;\n")
    print(f"[migrate] Wrote MariaDB SQL script to {output_path}")


def apply_bundle(bundle: MigrationBundle, drop_existing: bool, env_file: str | Path) -> None:
    try:
        import mysql.connector  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "mysql-connector-python is required for --apply. Install it via 'pip install mysql-connector-python'."
        ) from exc

    settings = load_settings(env_file)
    conn = mysql.connector.connect(
        host=settings.mariadb_host,
        port=settings.mariadb_port,
        user=settings.mariadb_user,
        password=settings.mariadb_password,
        database=settings.mariadb_database,
    )
    cursor = conn.cursor()
    cursor.execute("SET FOREIGN_KEY_CHECKS=0")
    if drop_existing:
        for table in reversed([table.name for table in bundle.tables]):
            cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
    for statement in bundle.create_statements:
        cursor.execute(statement)
    for index in bundle.index_statements:
        cursor.execute(index)
    for table in bundle.tables:
        if not table.rows:
            continue
        placeholders = ", ".join(["%s"] * len(table.columns))
        column_list = ", ".join(f"`{col}`" for col in table.columns)
        cursor.executemany(
            f"INSERT INTO `{table.name}` ({column_list}) VALUES ({placeholders})",
            table.rows,
        )
        conn.commit()
    cursor.execute("SET FOREIGN_KEY_CHECKS=1")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[migrate] Applied migration bundle to MariaDB database '{settings.mariadb_database}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the DB-SLM SQLite store into MariaDB-compatible SQL."
    )
    parser.add_argument(
        "--sqlite",
        default="var/db_slm.sqlite3",
        help="Path to the validated SQLite database (default: %(default)s).",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Environment file to load MariaDB credentials from (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="var/mariadb-migration.sql",
        help="File that will store the generated SQL script (default: %(default)s).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the generated bundle directly to MariaDB using mysql-connector-python.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop destination tables before applying --apply migrations (use with caution).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sqlite_path = Path(args.sqlite).expanduser()
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite database not found: {sqlite_path}")
    bundle = build_bundle(sqlite_path)
    output_path = Path(args.output).expanduser()
    write_sql(bundle, output_path)
    if args.apply:
        apply_bundle(bundle, drop_existing=args.drop_existing, env_file=args.env)
    else:
        print("[migrate] Skipped --apply (script written only).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
