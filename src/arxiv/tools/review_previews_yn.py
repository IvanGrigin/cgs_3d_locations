#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
import tkinter as tk


class Reviewer:
    def __init__(self, previews_dir: Path, assets_root: Path, dry_run: bool):
        self.previews_dir = previews_dir
        self.assets_root = assets_root
        self.dry_run = dry_run

        self.pngs = sorted(self.previews_dir.glob("*.png"))
        if not self.pngs:
            raise SystemExit(f"Нет PNG в: {self.previews_dir}")

        self.idx = 0
        self.current_png: Path | None = None

        self.root = tk.Tk()
        self.root.title("Asset review: y=keep, n=delete")
        self.root.configure(bg="black")

        # UI
        self.info = tk.Label(
            self.root, text="", fg="white", bg="black",
            font=("Menlo", 14), justify="left", anchor="w"
        )
        self.info.pack(fill="x", padx=10, pady=(10, 6))

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.hint = tk.Label(
            self.root,
            text="Keys: y = keep, n = delete asset folder, q = quit",
            fg="gray", bg="black", font=("Menlo", 12)
        )
        self.hint.pack(fill="x", padx=10, pady=(6, 10))

        # Key bindings
        self.root.bind("<KeyPress-y>", self.on_yes)
        self.root.bind("<KeyPress-n>", self.on_no)
        self.root.bind("<KeyPress-q>", self.on_quit)
        self.root.bind("<Escape>", self.on_quit)

        # Also accept uppercase
        self.root.bind("<KeyPress-Y>", self.on_yes)
        self.root.bind("<KeyPress-N>", self.on_no)
        self.root.bind("<KeyPress-Q>", self.on_quit)

        # Resize handling (re-render current image)
        self.root.bind("<Configure>", self.on_resize)

        # Tk image holder to prevent GC
        self._tk_img = None
        self._canvas_img_id = None

        self.load_current()

    def asset_dir_for(self, png: Path) -> Path:
        return self.assets_root / png.stem

    def load_current(self):
        if self.idx >= len(self.pngs):
            self.finish()
            return

        self.current_png = self.pngs[self.idx]
        asset_dir = self.asset_dir_for(self.current_png)
        exists = asset_dir.exists()

        self.info.config(
            text=(
                f"[{self.idx + 1}/{len(self.pngs)}] {self.current_png.name}\n"
                f"asset dir: {asset_dir} {'(MISSING)' if not exists else ''}\n"
                f"dry-run: {self.dry_run}"
            )
        )

        self.render_image()

    def render_image(self):
        # Canvas size
        w = max(200, self.canvas.winfo_width())
        h = max(200, self.canvas.winfo_height())

        png = self.current_png
        if png is None:
            return

        # PhotoImage can load PNG directly
        try:
            img = tk.PhotoImage(file=str(png))
        except tk.TclError as e:
            # If file unreadable, show error and allow skip via y (keep)
            self.canvas.delete("all")
            self.canvas.create_text(
                10, 10, anchor="nw",
                fill="red", font=("Menlo", 14),
                text=f"Cannot open image:\n{png}\n\n{e}"
            )
            self._tk_img = None
            return

        # Fit-to-window by integer subsample/zoom only (tkinter limitation)
        iw, ih = img.width(), img.height()
        if iw <= 0 or ih <= 0:
            return

        # Scale down if necessary using subsample (integer factor)
        scale = max(iw / w, ih / h, 1.0)
        if scale > 1.0:
            factor = int(scale) + 1
            img = img.subsample(factor, factor)

        # Center on canvas
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        x = cw // 2
        y = ch // 2

        self._tk_img = img
        self._canvas_img_id = self.canvas.create_image(x, y, image=self._tk_img, anchor="center")

    def step_next(self):
        self.idx += 1
        self.load_current()

    def on_yes(self, _event=None):
        # keep
        self.step_next()

    def on_no(self, _event=None):
        # delete asset folder
        if self.current_png is None:
            return
        asset_dir = self.asset_dir_for(self.current_png)
        if asset_dir.exists():
            if self.dry_run:
                print(f"DRY delete: {asset_dir}")
            else:
                shutil.rmtree(asset_dir)
                print(f"deleted: {asset_dir}")
        else:
            print(f"skip delete (missing): {asset_dir}")
        self.step_next()

    def on_quit(self, _event=None):
        self.root.destroy()

    def on_resize(self, _event=None):
        # re-render current image when window changes
        if self.current_png is not None:
            # debounce via after to avoid flooding
            if hasattr(self, "_resize_job") and self._resize_job is not None:
                try:
                    self.root.after_cancel(self._resize_job)
                except Exception:
                    pass
            self._resize_job = self.root.after(120, self.render_image)

    def finish(self):
        self.canvas.delete("all")
        self.info.config(text="DONE")
        self.hint.config(text="Press q or Esc to quit")
        print("DONE")

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--previews", default="data/output/asset_previews", help="папка с PNG превью")
    ap.add_argument("--root", default="data/sourse/imodern", help="корень папок ассетов (удаляем отсюда)")
    ap.add_argument("--dry-run", action="store_true", help="не удалять, только печатать действия")
    args = ap.parse_args()

    previews = Path(args.previews).resolve()
    assets_root = Path(args.root).resolve()

    Reviewer(previews, assets_root, args.dry_run).run()


if __name__ == "__main__":
    main()