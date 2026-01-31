# src/furniture_matcher.py
# -*- coding: utf-8 -*-
import argparse, difflib, json, os, re, sqlite3
from pathlib import Path

DEFAULT_ROOTS = ["data/sourse/imodern", "sourse/imodern", "data/sourse", "sourse"]

def norm(t: str) -> str:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^a-zа-я0-9_ +\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def toks(t: str): return [w for w in re.split(r"[\\/_\-\s]+", norm(t)) if w]

TYPE_PATTERNS = [
    ({"кровать","кроват","двуспаль","односпаль","полутор","трехспаль","трёхспаль"}, "bed"),
    ({"тв","тумба","тумбочка","комод"}, "tv_stand"),
    ({"стул","табурет"}, "chair"),
    ({"кресло"}, "armchair"),
    ({"диван","реклайнер"}, "sofa"),
    ({"стол","письменный","журнальный","обеденный","раскладной"}, "table"),
    ({"шкаф","гардероб","стеллаж"}, "wardrobe"),
    ({"консоль"}, "console"),
]

DEFAULT_SIZES_MM = {
    "bed":      ([1400,2000,350], [2000,2200,1100]),
    "tv_stand": ([1200,360,350],  [2400,500,650]),
    "chair":    ([420,420,800],   [520,520,1000]),
    "armchair": ([650,650,800],   [900,900,1100]),
    "sofa":     ([1600,800,800],  [2600,1100,1100]),
    "table":    ([800,800,720],   [2400,1100,760]),
    "wardrobe": ([800,500,1800],  [2400,800,2600]),
    "console":  ([900,300,700],   [1800,450,850]),
}

def guess_type(tokens):
    s = set(tokens)
    for keys, t in TYPE_PATTERNS:
        if s & keys: return t
    return None

def adjust_bed(tokens, mn, mx):
    s = " ".join(tokens)
    if "односпаль" in s: return [900,2000,mn[2]],  [1100,2200,mx[2]]
    if "полутор"   in s: return [1200,2000,mn[2]], [1400,2200,mx[2]]
    if "двуспаль"  in s: return [1400,2000,mn[2]], [1800,2200,mx[2]]
    if "трехспаль" in s or "трёхспаль" in s or "king" in s: 
        return [1800,2000,mn[2]], [2200,2200,mx[2]]
    return mn, mx

def sim(q, c):
    tq, tc = toks(q), toks(c)
    if not tq or not tc: return 0.0
    jacc = len(set(tq)&set(tc))/max(1,len(set(tq)|set(tc)))
    seq  = difflib.SequenceMatcher(None, norm(q), norm(c)).ratio()
    return 0.6*seq + 0.4*jacc

def scan_dirs(roots):
    out=[]
    for root in roots:
        p=Path(root)
        if not p.exists(): continue
        for dp,_,files in os.walk(p):
            files=set(f.lower() for f in files)
            mesh=None
            for f in files:
                if f.endswith(".obj"): mesh=os.path.join(dp,f); break
            if not mesh:
                for f in files:
                    if f.endswith(".glb") or f.endswith(".gltf"):
                        mesh=os.path.join(dp,f); break
            if not mesh: continue
            name=os.path.basename(dp)
            out.append({"name": name, "mesh_path": mesh})
    return out

def from_sqlite(db_path, table="items", name_col="name", mesh_col="mesh_path"):
    rows=[]
    con=sqlite3.connect(db_path)
    con.row_factory=sqlite3.Row
    cur=con.cursor()
    cur.execute(f"SELECT * FROM {table} LIMIT 1")
    cols={c["name"] for c in cur.description}
    if name_col not in cols:
        # пытаемся угадать
        for k in ("title","item_name","full_name"): 
            if k in cols: name_col=k; break
    if mesh_col not in cols:
        for k in ("obj_path","glb_path","path","file","filepath"):
            if k in cols: mesh_col=k; break
    cur.execute(f"SELECT {name_col} as name, {mesh_col} as mesh_path FROM {table}")
    for r in cur.fetchall():
        if r["name"] and r["mesh_path"]:
            rows.append({"name": r["name"], "mesh_path": r["mesh_path"]})
    con.close()
    return rows

def choose_best(query, catalog):
    best,score=None,0.0
    for it in catalog:
        sc=sim(query, it["name"])
        if sc>score: best,score=it,sc
    return best

def build_item(query, mesh_path):
    tk=toks(query)
    typ=guess_type(tk) or "tv_stand"
    mn,mx=DEFAULT_SIZES_MM[typ]
    if typ=="bed": mn,mx=adjust_bed(tk,mn,mx)
    cons={"mount_type":"floor"}
    if typ in {"tv_stand","wardrobe"}: cons["touch_wall"]={"side":"back"}
    return {
        "name": query,
        "min_size_mm": mn,
        "max_size_mm": mx,
        "color": [0.75,0.75,0.75],
        "constraints": cons,
        "mesh_path": mesh_path,
        "mesh_fit_mode": "stretch"
    }

def main():
    ap=argparse.ArgumentParser(description="Подбор мебели из .db или каталогов с .obj/.glb")
    ap.add_argument("queries", nargs="+", help="Напр.: 'трехспальная кровать' 'ТВ тумба Stanford'")
    ap.add_argument("--db", help="Путь к SQLite (.db)")
    ap.add_argument("--table", default="items")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--mesh-col", default="mesh_path")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--out", default="src/data/input/objects.json")
    ap.add_argument("--seed", type=int, default=None)
    args=ap.parse_args()

    catalog=[]
    if args.db and os.path.isfile(args.db):
        try:
            catalog=from_sqlite(args.db, args.table, args.name_col, args.mesh_col)
        except Exception as e:
            print("⚠️ Не удалось прочитать .db, переключаюсь на сканирование каталогов:", e)
    if not catalog:
        catalog=scan_dirs(args.roots)
    if not catalog:
        raise SystemExit("Не найдено ни одной модели")

    items=[]
    for q in args.queries:
        best=choose_best(q, catalog)
        if not best:
            raise SystemExit(f"Нет кандидатов для: {q}")
        items.append(build_item(q, best["mesh_path"]))

    out={"seed": args.seed, "items": items}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(out,f,indent=2,ensure_ascii=False)
    print(f"✅ {args.out} создан: {len(items)} шт.")

if __name__=="__main__":
    main()