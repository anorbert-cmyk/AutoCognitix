# Wave2 R3 — Data

## A) Migration SHARE + downgrade
- severity: MEDIUM
- finding: A `print(f"... rows: {purged}")` (sor 63) az ÖSSZES orphan UUID-t egyetlen stringben dump-olja — 10000 orphan ≈ 370 KB stdout (UUID 36 char + `, '` overhead). Railway / GitHub Actions log soronkénti limit (~4-64 KB jellemző) miatt truncate-elődhet, pont az audit-trail veszik el. Javaslat: chunked print 100/500-as batchekben, vagy `print(f"... count={len(purged)} first={purged[:50]} last={purged[-50:]}")` korlátozott mintával, a teljes lista pedig külön audit táblába `INSERT`-tel (a SHARE lock alatt amúgy is biztonságos).
- finding: A downgrade `op.drop_constraint(...)` (sor 96-100) NEM idempotens — ha a downgrade kétszer fut, vagy ha az upgrade félúton failelt (constraint sosem jött létre), `UndefinedObject` exception. Alembic tényleg nem támogatja `if_exists`-et constraint-en (op API limitáció), de a `op.execute(sa.text("ALTER TABLE diagnosis_archive DROP CONSTRAINT IF EXISTS diagnosis_archive_user_id_fkey"))` workaround konzisztens lenne az indexek `if_exists=True` mintázatával (sor 101, 103). A docstring említi a downgrade→re-upgrade ciklust mint kommunikációs note, de a tényleges idempotencia a constraint drop-nál hiányzik.

## B) bind vs op execute
- severity: LOW
- finding: A `bind = op.get_bind(); result = bind.execute(...)` (sor 41, 52) helyes mintázat amikor `CursorResult.fetchall()` kell (RETURNING). A többi DDL (`op.execute(sa.text("LOCK TABLE ..."))`, sor 46) tudatosan op.execute-ot használ mert nincs visszatérési érték. Az inkonzisztencia funkcionálisan indokolt, nem hiba. Ha mégis egységesíteni akarjátok: minden helyen `bind.execute(...)` használata működne (op.execute is bind-et használ a háttérben), de stilisztikai változás, nincs szemantikai különbség. Megtartás javasolt.
