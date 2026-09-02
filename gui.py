#!/usr/bin/env python3
"""
Tkinter preview UI for the offline transcription pipeline.

- Drag-and-drop / multi-file selection into Sources
- Per-file language after files are added
- Transcribe button with live Markdown preview
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from transcribe import (
    DEFAULT_LANGUAGE_LABEL,
    DEFAULT_MODEL,
    LANGUAGE_PRESETS,
    MODEL_CHOICES,
    OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
    ensure_output_dir,
    format_timestamp,
    load_model,
    transcript_path,
    transcribe_file_by_label,
)

FILE_FILTER = (
    ("Media files", " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))),
    ("All files", "*.*"),
)

# Prefer tkinterdnd2 for Windows drag-and-drop into the Sources list.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _TkBase = TkinterDnD.Tk
    _HAS_DND = True
except ImportError:
    _TkBase = tk.Tk
    _HAS_DND = False
    DND_FILES = None  # type: ignore


class TranscriptorApp(_TkBase):
    def __init__(self) -> None:
        super().__init__()
        self.title("Transcriptor — Offline Whisper Preview")
        self.geometry("980x720")
        self.minsize(780, 560)

        self._model = None
        self._model_name = ""
        self._busy = False
        self._results: dict[str, str] = {}  # path str -> markdown
        self._msg_queue: queue.Queue = queue.Queue()
        self._lang_vars: dict[str, tk.StringVar] = {}
        self._updating_selected_lang = False
        self._partial_preview: list[str] = []

        self._build_ui()
        self._setup_drag_drop()
        self.after(120, self._poll_queue)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar = ttk.Frame(root)
        toolbar.pack(fill=tk.X, **pad)

        ttk.Button(toolbar, text="Add Files…", command=self._add_files).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Add Folder…", command=self._add_folder).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Remove Selected", command=self._remove_selected).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Clear All", command=self._clear_all).pack(
            side=tk.LEFT, padx=(0, 12)
        )

        ttk.Label(toolbar, text="Default for new files:").pack(side=tk.LEFT)
        self.default_lang = tk.StringVar(value=DEFAULT_LANGUAGE_LABEL)
        self.default_lang_combo = ttk.Combobox(
            toolbar,
            textvariable=self.default_lang,
            values=list(LANGUAGE_PRESETS.keys()),
            state="readonly",
            width=22,
        )
        self.default_lang_combo.pack(side=tk.LEFT, padx=(4, 6))
        ttk.Button(
            toolbar, text="Apply Default to All", command=self._apply_lang_to_all
        ).pack(side=tk.LEFT)

        # Model / speed row
        model_row = ttk.Frame(root)
        model_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        ttk.Label(model_row, text="Model (CPU speed):").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        model_labels = {
            "small": "small — fast (recommended on CPU)",
            "medium": "medium — balanced quality / speed",
            "turbo": "turbo — best quality, slowest on CPU",
        }
        self._model_display = {
            model_labels.get(m, m): m for m in MODEL_CHOICES
        }
        default_display = model_labels.get(DEFAULT_MODEL, DEFAULT_MODEL)
        self.model_display_var = tk.StringVar(value=default_display)
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_display_var,
            values=list(self._model_display.keys()),
            state="readonly",
            width=40,
        )
        self.model_combo.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            model_row,
            text="  small ≈ fastest · medium ≈ better accuracy · turbo ≈ slowest",
            foreground="#555",
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Sources list (drop target)
        drop_hint = (
            "Sources — drag & drop media files or folders here"
            if _HAS_DND
            else "Sources (install tkinterdnd2 for drag-and-drop)"
        )
        self.list_frame = ttk.LabelFrame(root, text=drop_hint, padding=8)
        self.list_frame.pack(fill=tk.BOTH, expand=False, **pad)

        columns = ("file", "language", "status")
        self.tree = ttk.Treeview(
            self.list_frame,
            columns=columns,
            show="headings",
            height=9,
            selectmode="extended",
        )
        self.tree.heading("file", text="File")
        self.tree.heading("language", text="Meeting Language")
        self.tree.heading("status", text="Status")
        self.tree.column("file", width=460, anchor=tk.W)
        self.tree.column("language", width=200, anchor=tk.W)
        self.tree.column("status", width=140, anchor=tk.W)

        scroll_y = ttk.Scrollbar(
            self.list_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double_click_lang)

        # Per-file language editor (after files are in the list)
        lang_row = ttk.Frame(root)
        lang_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        ttk.Label(lang_row, text="Language for selected file(s):").pack(side=tk.LEFT)
        self.selected_lang = tk.StringVar(value=DEFAULT_LANGUAGE_LABEL)
        self.selected_lang_combo = ttk.Combobox(
            lang_row,
            textvariable=self.selected_lang,
            values=list(LANGUAGE_PRESETS.keys()),
            state="disabled",
            width=24,
        )
        self.selected_lang_combo.pack(side=tk.LEFT, padx=(8, 6))
        self.selected_lang_combo.bind(
            "<<ComboboxSelected>>", self._on_selected_lang_changed
        )

        ttk.Button(
            lang_row,
            text="Set Language",
            command=self._apply_lang_to_selected,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(
            lang_row,
            text="Select a row → choose language → Set Language  (or double-click the row)",
            foreground="#555",
        ).pack(side=tk.LEFT)

        # Action row
        action = ttk.Frame(root)
        action.pack(fill=tk.X, **pad)

        self.overwrite_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            action,
            text="Overwrite existing .md",
            variable=self.overwrite_var,
        ).pack(side=tk.LEFT)

        self.transcribe_btn = ttk.Button(
            action,
            text="Transcribe",
            command=self._start_transcribe,
        )
        self.transcribe_btn.pack(side=tk.RIGHT, padx=(6, 0))

        ttk.Button(
            action,
            text="Open Output Folder",
            command=self._open_output_folder,
        ).pack(side=tk.RIGHT)

        # Progress
        prog = ttk.Frame(root)
        prog.pack(fill=tk.X, **pad)
        ready = (
            "Ready — drag files into Sources, or use Add Files…"
            if _HAS_DND
            else "Ready — use Add Files… (drag-and-drop needs: pip install tkinterdnd2)"
        )
        self.status_var = tk.StringVar(value=ready)
        ttk.Label(prog, textvariable=self.status_var).pack(anchor=tk.W)
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(4, 0))

        # Preview
        preview = ttk.LabelFrame(root, text="Transcript Preview", padding=8)
        preview.pack(fill=tk.BOTH, expand=True, **pad)

        self.preview = tk.Text(
            preview,
            wrap=tk.WORD,
            font=("Consolas", 10),
            undo=False,
        )
        preview_scroll = ttk.Scrollbar(
            preview, orient=tk.VERTICAL, command=self.preview.yview
        )
        self.preview.configure(yscrollcommand=preview_scroll.set)
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview.insert(
            "1.0",
            "1) Add or drag media into Sources\n"
            "2) Select each file and set its meeting language\n"
            "3) Click Transcribe\n\n"
            f"Markdown transcripts are saved to:\n{OUTPUT_DIR}",
        )
        self.preview.configure(state=tk.DISABLED)

    # ---------------------------------------------------------- drag-drop
    def _setup_drag_drop(self) -> None:
        if not _HAS_DND:
            return
        # Register the whole Sources area + the tree + window
        for widget in (self, self.list_frame, self.tree):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
            except tk.TclError:
                pass

    def _on_drop(self, event) -> str:
        """Handle files/folders dropped onto the Sources section."""
        try:
            raw_paths = self.tk.splitlist(event.data)
        except tk.TclError:
            raw_paths = [event.data]

        paths: list[Path] = []
        for raw in raw_paths:
            text = raw.strip().strip("{}").strip('"')
            if not text:
                continue
            paths.append(Path(text))

        self._ingest_paths(paths, from_drop=True)
        return event.action if hasattr(event, "action") else "copy"

    # -------------------------------------------------------------- file ops
    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=FILE_FILTER,
        )
        self._ingest_paths([Path(p) for p in paths])

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder of recordings")
        if not folder:
            return
        self._ingest_paths([Path(folder)])

    def _expand_to_media(self, paths: list[Path]) -> list[Path]:
        """Resolve files and recurse into folders for supported media."""
        found: list[Path] = []
        for path in paths:
            try:
                path = path.expanduser().resolve()
            except OSError:
                continue
            if path.is_file():
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.append(path)
            elif path.is_dir():
                for ext in SUPPORTED_EXTENSIONS:
                    found.extend(path.rglob(f"*{ext}"))
                    found.extend(path.rglob(f"*{ext.upper()}"))
        # Deduplicate while preserving order
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in found:
            try:
                key = p.resolve()
            except OSError:
                continue
            if key not in seen and key.is_file():
                seen.add(key)
                unique.append(key)
        return unique

    def _ingest_paths(self, paths: list[Path], *, from_drop: bool = False) -> None:
        media = self._expand_to_media(paths)
        added = 0
        first_iid: Optional[str] = None
        for path in media:
            key = str(path)
            if self.tree.exists(key):
                continue
            lang = self.default_lang.get()
            self._lang_vars[key] = tk.StringVar(value=lang)
            self.tree.insert(
                "",
                tk.END,
                iid=key,
                values=(path.name, lang, "Queued"),
            )
            if first_iid is None:
                first_iid = key
            added += 1

        if added:
            verb = "Dropped" if from_drop else "Added"
            self.status_var.set(f"{verb} {added} file(s). {self._count()} total.")
            if first_iid:
                self.tree.selection_set(first_iid)
                self.tree.see(first_iid)
                self._on_select()
        elif paths:
            self.status_var.set(
                "No supported media found "
                f"({', '.join(sorted(SUPPORTED_EXTENSIONS))})."
            )

    def _count(self) -> int:
        return len(self.tree.get_children())

    def _remove_selected(self) -> None:
        for iid in self.tree.selection():
            self.tree.delete(iid)
            self._lang_vars.pop(iid, None)
            self._results.pop(iid, None)
        self._sync_selected_lang_combo()
        self.status_var.set(f"{self._count()} file(s) remaining.")

    def _clear_all(self) -> None:
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._lang_vars.clear()
        self._results.clear()
        self._sync_selected_lang_combo()
        self._set_preview(
            "1) Add or drag media into Sources\n"
            "2) Select each file and set its meeting language\n"
            "3) Click Transcribe"
        )
        self.status_var.set("List cleared.")

    def _apply_lang_to_all(self) -> None:
        lang = self.default_lang.get()
        for iid in self.tree.get_children():
            self._set_row_language(iid, lang)
        self._sync_selected_lang_combo()
        self.status_var.set(f"Applied “{lang}” to all sources.")

    def _set_row_language(self, iid: str, lang: str) -> None:
        if iid in self._lang_vars:
            self._lang_vars[iid].set(lang)
        vals = list(self.tree.item(iid, "values"))
        if len(vals) >= 2:
            vals[1] = lang
            self.tree.item(iid, values=vals)

    def _apply_lang_to_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo(
                "Set Language",
                "Select one or more files in Sources first, then choose a language.",
            )
            return
        lang = self.selected_lang.get()
        for iid in sel:
            self._set_row_language(iid, lang)
        names = ", ".join(Path(i).name for i in sel[:3])
        extra = f" (+{len(sel) - 3} more)" if len(sel) > 3 else ""
        self.status_var.set(f"Set “{lang}” on: {names}{extra}")

    def _on_selected_lang_changed(self, _event=None) -> None:
        """Changing the combo immediately updates the current selection."""
        if self._updating_selected_lang:
            return
        if self.tree.selection():
            self._apply_lang_to_selected()

    def _sync_selected_lang_combo(self) -> None:
        sel = self.tree.selection()
        self._updating_selected_lang = True
        try:
            if not sel:
                self.selected_lang_combo.configure(state="disabled")
                return
            self.selected_lang_combo.configure(state="readonly")
            # If multi-select has mixed languages, show the first; user can still set all
            first = sel[0]
            if first in self._lang_vars:
                self.selected_lang.set(self._lang_vars[first].get())
        finally:
            self._updating_selected_lang = False

    def _on_double_click_lang(self, event=None) -> None:
        # Only open editor when double-clicking the language column (or any row)
        sel = self.tree.selection()
        if not sel:
            # Identify row under cursor
            if event is not None:
                row = self.tree.identify_row(event.y)
                if row:
                    self.tree.selection_set(row)
                    sel = (row,)
        if not sel:
            return
        self._edit_language(sel[0])

    def _edit_language(self, iid: str) -> None:
        labels = list(LANGUAGE_PRESETS.keys())
        current = self._lang_vars[iid].get()

        pop = tk.Toplevel(self)
        pop.title("Meeting language")
        pop.transient(self)
        pop.grab_set()
        pop.resizable(False, False)

        ttk.Label(pop, text=Path(iid).name, padding=10).pack(anchor=tk.W)
        ttk.Label(pop, text="Language for this recording:", padding=(10, 0)).pack(
            anchor=tk.W
        )

        var = tk.StringVar(value=current)
        combo = ttk.Combobox(
            pop, textvariable=var, values=labels, state="readonly", width=28
        )
        combo.pack(padx=10, pady=8)
        combo.focus_set()

        def apply() -> None:
            chosen = var.get()
            self._set_row_language(iid, chosen)
            self.tree.selection_set(iid)
            self._sync_selected_lang_combo()
            pop.destroy()

        btn_row = ttk.Frame(pop, padding=10)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Cancel", command=pop.destroy).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="OK", command=apply).pack(side=tk.RIGHT, padx=(0, 6))
        pop.bind("<Return>", lambda _e: apply())
        pop.bind("<Escape>", lambda _e: pop.destroy())

        pop.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - pop.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - pop.winfo_height()) // 2
        pop.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------ preview
    def _set_preview(self, text: str) -> None:
        self.preview.configure(state=tk.NORMAL)
        self.preview.delete("1.0", tk.END)
        self.preview.insert("1.0", text)
        self.preview.configure(state=tk.DISABLED)

    def _on_select(self, _event=None) -> None:
        self._sync_selected_lang_combo()
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid in self._results:
            self._set_preview(self._results[iid])
            return
        md = transcript_path(Path(iid))
        if md.exists():
            try:
                self._set_preview(md.read_text(encoding="utf-8"))
            except OSError:
                self._set_preview(f"(Could not read existing transcript: {md})")
        else:
            self._set_preview(
                f"No transcript yet for:\n{Path(iid).name}\n\n"
                f"Language: {self._lang_vars[iid].get()}\n"
                f"Output folder: {OUTPUT_DIR}\n\n"
                "Change language above, then click Transcribe."
            )

    def _open_output_folder(self) -> None:
        folder = ensure_output_dir()
        try:
            import os

            os.startfile(folder)  # type: ignore[attr-defined]  # Windows
        except OSError as exc:
            messagebox.showerror("Open folder", str(exc))

    # --------------------------------------------------------- transcription
    def _set_row_status(self, iid: str, status: str) -> None:
        vals = list(self.tree.item(iid, "values"))
        vals[2] = status
        self.tree.item(iid, values=vals)

    def _start_transcribe(self) -> None:
        if self._busy:
            return
        items = list(self.tree.get_children())
        if not items:
            messagebox.showinfo("Transcribe", "Add at least one media file first.")
            return

        jobs: list[tuple[str, str]] = []
        for iid in items:
            md = transcript_path(Path(iid))
            if md.exists() and not self.overwrite_var.get():
                self._set_row_status(iid, "Skipped")
                try:
                    self._results[iid] = md.read_text(encoding="utf-8")
                except OSError:
                    pass
                continue
            jobs.append((iid, self._lang_vars[iid].get()))

        if not jobs:
            self.status_var.set("Nothing to do — all transcripts already exist.")
            if items:
                self.tree.selection_set(items[0])
                self._on_select()
            return

        self._busy = True
        self.transcribe_btn.configure(state=tk.DISABLED)
        self.progress.configure(mode="determinate", maximum=1000, value=0)
        chosen = self._model_display.get(
            self.model_display_var.get(), DEFAULT_MODEL
        )
        self.status_var.set(
            f"Loading model “{chosen}” (first use may download weights)…"
        )

        thread = threading.Thread(
            target=self._worker, args=(jobs, chosen), daemon=True
        )
        thread.start()

    def _worker(self, jobs: list[tuple[str, str]], model_name: str) -> None:
        try:
            # Reload if model choice changed
            if self._model is None or self._model_name != model_name:
                self._msg_queue.put(("status", f"Loading model “{model_name}”…"))
                model, name = load_model(model_name)
                self._model = model
                self._model_name = name
                self._msg_queue.put(("status", f"Model ready: {name}"))

            ok = fail = 0
            for i, (iid, lang_label) in enumerate(jobs, start=1):
                self._msg_queue.put(("row", iid, "Running…"))
                self._msg_queue.put(("partial_reset",))
                self._msg_queue.put(
                    (
                        "status",
                        f"[{i}/{len(jobs)}] {Path(iid).name} — {lang_label} "
                        f"(model={self._model_name})",
                    )
                )

                def on_progress(
                    current: float,
                    duration: float,
                    seg_i: int,
                    text: str,
                    _iid=iid,
                    _i=i,
                    _n=len(jobs),
                ) -> None:
                    self._msg_queue.put(
                        ("seg_progress", _iid, _i, _n, current, duration, seg_i, text)
                    )

                result = transcribe_file_by_label(
                    self._model,
                    Path(iid),
                    lang_label,
                    progress_cb=on_progress,
                )
                if result.success:
                    ok += 1
                    self._msg_queue.put(
                        ("done", iid, result.markdown, f"Done ({result.detected_lang})")
                    )
                else:
                    fail += 1
                    self._msg_queue.put(("fail", iid, result.error))

            self._msg_queue.put(
                ("finished", ok, fail, f"Finished. Succeeded: {ok} | Failed: {fail}")
            )
        except Exception as exc:  # noqa: BLE001
            self._msg_queue.put(("fatal", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                kind = msg[0]
                if kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "partial_reset":
                    self._partial_preview = [
                        "# Live transcript (in progress…)",
                        "",
                        "---",
                        "",
                    ]
                    self._set_preview("\n".join(self._partial_preview))
                elif kind == "seg_progress":
                    (
                        iid,
                        file_i,
                        file_n,
                        current,
                        duration,
                        seg_i,
                        text,
                    ) = msg[1:]
                    # Overall bar: file index + within-file fraction
                    if duration > 0:
                        frac = min(1.0, max(0.0, current / duration))
                    else:
                        frac = 0.0
                    overall = ((file_i - 1) + frac) / max(1, file_n)
                    self.progress.configure(value=int(overall * 1000))
                    self._set_row_status(
                        iid,
                        f"{format_timestamp(current).strip('[]')}"
                        + (
                            f" / {format_timestamp(duration).strip('[]')}"
                            if duration > 0
                            else ""
                        ),
                    )
                    self.status_var.set(
                        f"[{file_i}/{file_n}] {Path(iid).name} — "
                        f"segment {seg_i} @ {format_timestamp(current)}"
                        + (
                            f" / {format_timestamp(duration)}"
                            if duration > 0
                            else ""
                        )
                    )
                    if text:
                        self._partial_preview.append(
                            f"**{format_timestamp(current)}**: {text}"
                        )
                        self._partial_preview.append("")
                        # Refresh preview every few segments to keep UI light
                        if seg_i == 1 or seg_i % 3 == 0:
                            self._set_preview("\n".join(self._partial_preview))
                elif kind == "row":
                    self._set_row_status(msg[1], msg[2])
                elif kind == "done":
                    iid, markdown, status = msg[1], msg[2], msg[3]
                    self._results[iid] = markdown
                    self._set_row_status(iid, status)
                    self.tree.selection_set(iid)
                    self.tree.see(iid)
                    self._set_preview(markdown)
                    self._sync_selected_lang_combo()
                elif kind == "fail":
                    iid, err = msg[1], msg[2]
                    self._set_row_status(iid, "Failed")
                    self._results[iid] = f"# Transcription failed\n\n`{err}`"
                    self.tree.selection_set(iid)
                    self._set_preview(self._results[iid])
                elif kind == "finished":
                    _ok, _fail, text = msg[1], msg[2], msg[3]
                    self.status_var.set(text)
                    self.progress.configure(value=1000)
                    self._busy = False
                    self.transcribe_btn.configure(state=tk.NORMAL)
                elif kind == "fatal":
                    self.status_var.set(f"Error: {msg[1]}")
                    messagebox.showerror("Transcription error", msg[1])
                    self._busy = False
                    self.transcribe_btn.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)


def run_gui(initial_paths: Optional[list[Path]] = None) -> None:
    app = TranscriptorApp()
    if initial_paths:
        app._ingest_paths(initial_paths)
    app.mainloop()


if __name__ == "__main__":
    run_gui()
