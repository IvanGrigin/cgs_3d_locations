# src/tools/blender_obj_preview.py
import sys
import argparse
import bpy


def _parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="Path to .obj")
    ap.add_argument("--mtl", default=None, help="Path to .mtl (optional; usually referenced from obj)")
    return ap.parse_args(argv)


def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_obj(path: str):
    # Blender 4+/5 uses wm.obj_import
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        # fallback for older versions
        bpy.ops.import_scene.obj(filepath=path)


def _frame_view():
    # Подгоняем вид во всех 3D Viewport-окнах
    for area in bpy.context.window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        region = None
        for r in area.regions:
            if r.type == "WINDOW":
                region = r
                break
        if region is None:
            continue
        override = bpy.context.copy()
        override["area"] = area
        override["region"] = region
        try:
            bpy.ops.view3d.view_all(override, center=True)
        except Exception:
            pass


def main():
    args = _parse_argv()
    _clear_scene()
    _import_obj(args.obj)
    _frame_view()
    # Ничего не рендерим и не закрываем Blender: окно остаётся, пока ты его не закроешь.


if __name__ == "__main__":
    main()