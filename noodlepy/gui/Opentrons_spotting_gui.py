import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import datetime
import subprocess
import os
import json

# Try to import PDF library
try:
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import letter, landscape, portrait
    from reportlab.lib import colors as pdf_colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- CONFIGURATION ---
VALID_WELLS = [
    "B3", "C3", "D3", "G3", "H3", "I3", "J3", "M3", "N3", "O3",
    "A4", "A5", "A10", "A11", "A16", "A17", "A22", "A23",
    "B4", "B5", "B6", "B9", "B10", "B11", "B12", "B15", "B16", "B17", "B18", "B21", "B22", "B23", "B24",
    "C4", "C5", "C6", "C9", "C10", "C11", "C12", "C15", "C16", "C17", "C18", "C21", "C22", "C23", "C24",
    "D4", "D5", "D6", "D9", "D10", "D11", "D12", "D15", "D16", "D17", "D18", "D21", "D22", "D23", "D24",
    "E4", "E5", "E10", "E11", "E16", "E17", "E22", "E23",
    "G4", "G5", "G6", "G9", "G10", "G11", "G12", "G15", "G16", "G17", "G18", "G21", "G22", "G23", "G24",
    "H4", "H5", "H6", "H9", "H10", "H11", "H12", "H15", "H16", "H17", "H18", "H21", "H22", "H23", "H24",
    "I4", "I5", "I6", "I9", "I10", "I11", "I12", "I15", "I16", "I17", "I18", "I21", "I22", "I23", "I24",
    "J4", "J5", "J6", "J9", "J10", "J11", "J12", "J15", "J16", "J17", "J18", "J21", "J22", "J23", "J24",
    "L4", "L5", "L10", "L11", "L16", "L17", "L22", "L23",
    "M4", "M5", "M6", "M9", "M10", "M11", "M12", "M15", "M16", "M17", "M18", "M21", "M22", "M23", "M24",
    "N4", "N5", "N6", "N9", "N10", "N11", "N12", "N15", "N16", "N17", "N18", "N21", "N22", "N23", "N24",
    "O4", "O5", "O6", "O9", "O10", "O11", "O12", "O15", "O16", "O17", "O18", "O21", "O22", "O23", "O24",
    "P4", "P5", "P10", "P11", "P16", "P17", "P22", "P23"
]

COLORS_HEX = ["#ff5252", "#69f0ae", "#ffd740", "#448aff", "#ffab40", "#e040fb", "#18ffff", "#ff4081", "#c6ff00", "#ff6e40"]

# --- THEME COLORS (Dark Mode) ---
BG_COLOR = "#2e2e2e"
PANEL_BG = "#333333"
TEXT_COLOR = "#ffffff"
ENTRY_BG = "#505050"
ENTRY_FG = "#ffffff"
CANVAS_BG = "#404040"

def get_well_sort_key(well_name):
    row_char = well_name[0]
    col_num = int(well_name[1:])
    if 3 <= col_num <= 6: col_block = 0
    elif 9 <= col_num <= 12: col_block = 1
    elif 15 <= col_num <= 18: col_block = 2
    elif 21 <= col_num <= 24: col_block = 3
    else: col_block = 99
    row_idx = ord(row_char) - ord('A')
    if row_idx <= 4: row_block = 0       
    elif 6 <= row_idx <= 9: row_block = 1
    elif 11 <= row_idx <= 15: row_block = 2
    else: row_block = 99
    raw_group_num = (row_block * 4) + col_block + 1
    snake_order_map = {4: 1, 3: 2, 2: 3, 1: 4, 5: 5, 6: 6, 7: 7, 8: 8, 12: 9, 11: 10, 10: 11, 9: 12}
    return (snake_order_map.get(raw_group_num, 99), row_idx, -col_num)

class SpottingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Opentrons Spotting Manager v33.0")
        self.root.geometry("1450x950")
        self.root.configure(bg=BG_COLOR)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background="#444", foreground="white", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#666")], foreground=[("selected", "white")])
        style.configure("TFrame", background=BG_COLOR)

        # Variables
        date_prefix = datetime.datetime.now().strftime("%Y%m%d")
        self.protocol_name = tk.StringVar(value=f"{date_prefix}_OT2")
        
        self.num_samples = tk.IntVar(value=1)
        self.mix_volume = tk.DoubleVar(value=10.0)
        self.mix_cycles = tk.IntVar(value=3)
        self.droplets_per_sample = tk.IntVar(value=1)
        self.droplet_volume = tk.DoubleVar(value=1.0)
        
        self.asp_rate = tk.DoubleVar(value=5.0)
        self.disp_rate = tk.DoubleVar(value=5.0)
        
        # Generator Checkboxes
        self.gen_mix = tk.BooleanVar(value=False)
        self.gen_spot = tk.BooleanVar(value=False)
        self.gen_combined = tk.BooleanVar(value=True)
        
        # Blowout Controls
        self.do_blowout = tk.BooleanVar(value=False)
        self.blowout_height = tk.DoubleVar(value=1.0)
        self.blowout_rate = tk.DoubleVar(value=5.0)

        self.start_sample_idx = tk.IntVar(value=1)
        self.start_tip = tk.StringVar(value="A1")
        self.sample_entries = [] 
        self.last_path = ""
        self.overflow_strategy = "none"

        # --- Layout ---
        main = tk.Frame(root, bg=BG_COLOR)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left_panel = tk.Frame(main, width=320, bg=BG_COLOR)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        mid_panel = tk.Frame(main, width=300, bg=PANEL_BG, relief="flat", borderwidth=1)
        mid_panel.pack(side="left", fill="both", padx=10)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(side="right", fill="both", expand=True)
        self.tab_deck = tk.Frame(self.notebook, bg=BG_COLOR)
        self.tab_plate = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self.tab_deck, text="Source Layout (Deck)")
        self.notebook.add(self.tab_plate, text="Destination Map")

        # --- Left Panel ---
        tk.Label(left_panel, text="Run Actions (Check to Generate):", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=(10,0), anchor="w")
        cb_frame = tk.Frame(left_panel, bg=BG_COLOR)
        cb_frame.pack(fill="x", pady=5)
        tk.Checkbutton(cb_frame, text="Mix", variable=self.gen_mix, bg=BG_COLOR, fg="white", selectcolor=ENTRY_BG, activebackground=BG_COLOR, activeforeground="white").pack(side="left", padx=(0, 5))
        tk.Checkbutton(cb_frame, text="Spot", variable=self.gen_spot, bg=BG_COLOR, fg="white", selectcolor=ENTRY_BG, activebackground=BG_COLOR, activeforeground="white").pack(side="left", padx=5)
        tk.Checkbutton(cb_frame, text="Combined", variable=self.gen_combined, bg=BG_COLOR, fg="white", selectcolor=ENTRY_BG, activebackground=BG_COLOR, activeforeground="white").pack(side="left", padx=5)

        tk.Label(left_panel, text="\nGlobal Settings:", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
        self.create_input(left_panel, "Total Samples:", self.num_samples)
        
        tk.Label(left_panel, text="\nMixing Settings:", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
        self.create_input(left_panel, "Mix Vol (1-20 µL):", self.mix_volume)
        self.create_input(left_panel, "Mix Cycles:", self.mix_cycles)

        tk.Label(left_panel, text="\nSpotting Settings:", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
        self.create_input(left_panel, "Droplets/Sample:", self.droplets_per_sample)
        self.create_input(left_panel, "Dispense Vol (µL):", self.droplet_volume)

        tk.Label(left_panel, text="\nPhysics Config:", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
        self.create_input(left_panel, "Asp Rate (0.1-25):", self.asp_rate)
        self.create_input(left_panel, "Disp Rate (0.1-25):", self.disp_rate)
        tk.Label(left_panel, text="*Use slower disp rates for more viscous samples", font=("Segoe UI", 8, "italic"), bg=BG_COLOR, fg="#aaaaaa").pack(anchor="w", padx=5, pady=(0, 5))
        
        tk.Label(left_panel, text="\nBlowout Config:", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w")
        tk.Checkbutton(left_panel, text="Enable Blowout", variable=self.do_blowout, bg=BG_COLOR, fg="#ffab40", selectcolor=ENTRY_BG, activebackground=BG_COLOR, activeforeground="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,2))
        tk.Label(left_panel, text="*Warning, can introduce bubbles to the sample!", font=("Segoe UI", 8, "italic"), bg=BG_COLOR, fg="#ff5252").pack(anchor="w", padx=5, pady=(0, 5))
        self.create_input(left_panel, "Blowout Ht (mm):", self.blowout_height)
        self.create_input(left_panel, "Blowout Rate:", self.blowout_rate)

        rec_frame = tk.LabelFrame(left_panel, text="Crash Recovery", bg=BG_COLOR, fg="#ffab40", font=("Segoe UI", 10, "bold"), bd=1)
        rec_frame.pack(fill="x", pady=15, ipady=5)
        self.create_input(rec_frame, "Start Sample #:", self.start_sample_idx)
        self.create_input(rec_frame, "First Tip (e.g. A1):", self.start_tip)

        self.info_label = tk.Label(left_panel, text="", justify="left", fg="#4fc3f7", bg=BG_COLOR)
        self.info_label.pack(pady=10, anchor="w")

        tk.Button(left_panel, text="1. Generate Protocols", command=self.check_and_generate, bg="#388e3c", fg="white", font=("Segoe UI", 11, "bold"), relief="flat").pack(fill="x", pady=5)
        if REPORTLAB_AVAILABLE:
            tk.Button(left_panel, text="2. Export PDF Report", command=self.export_pdf, bg="#7b1fa2", fg="white", font=("Segoe UI", 11), relief="flat").pack(fill="x", pady=5)
        
        # --- Middle Panel ---
        tk.Label(mid_panel, text="Protocol Base Name:", font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg="#69f0ae").pack(pady=(10,2), anchor="w", padx=10)
        tk.Entry(mid_panel, textvariable=self.protocol_name, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground="white", font=("Segoe UI", 10)).pack(fill="x", padx=10, pady=(0,10))

        tk.Label(mid_panel, text="Sample Names:", font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg=TEXT_COLOR).pack(pady=2, anchor="w", padx=10)
        tk.Button(mid_panel, text="Paste List (Excel)", command=self.open_paste_window, bg="#00796b", fg="white", relief="flat").pack(fill="x", padx=10, pady=5)

        canvas_scroll = tk.Canvas(mid_panel, bg=PANEL_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(mid_panel, orient="vertical", command=canvas_scroll.yview)
        self.sample_list_frame = tk.Frame(canvas_scroll, bg=PANEL_BG)
        self.sample_list_frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=self.sample_list_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Right Panel ---
        self.canvas_plate = tk.Canvas(self.tab_plate, bg=CANVAS_BG, width=600, height=800, highlightthickness=0)
        plate_scroll = ttk.Scrollbar(self.tab_plate, orient="vertical", command=self.canvas_plate.yview)
        plate_scroll.pack(side="right", fill="y")
        self.canvas_plate.pack(side="left", fill="both", expand=True)
        self.canvas_plate.config(yscrollcommand=plate_scroll.set)
        
        self.canvas_deck = tk.Canvas(self.tab_deck, bg=CANVAS_BG, width=600, height=800, scrollregion=(0,0,600,1000), highlightthickness=0)
        deck_scroll = ttk.Scrollbar(self.tab_deck, orient="vertical", command=self.canvas_deck.yview)
        deck_scroll.pack(side="right", fill="y")
        self.canvas_deck.pack(side="left", fill="both", expand=True)
        self.canvas_deck.config(yscrollcommand=deck_scroll.set)

        self.num_samples.trace_add("write", self.update_all)
        self.droplets_per_sample.trace_add("write", self.update_visual_only)
        self.start_sample_idx.trace_add("write", self.update_visual_only)
        self.gen_mix.trace_add("write", self.update_visual_only)
        self.gen_spot.trace_add("write", self.update_visual_only)
        self.gen_combined.trace_add("write", self.update_visual_only)
        self.sorted_wells_cache = sorted(VALID_WELLS, key=get_well_sort_key)
        self.update_all()

    def create_input(self, parent, label, var):
        f = tk.Frame(parent, bg=parent['bg'])
        f.pack(fill="x", pady=5)
        tk.Label(f, text=label, width=15, anchor="w", bg=parent['bg'], fg=TEXT_COLOR).pack(side="left")
        tk.Entry(f, textvariable=var, width=8, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground="white").pack(side="right")

    def open_paste_window(self):
        top = tk.Toplevel(self.root)
        top.title("Paste Sample Names")
        top.geometry("400x500")
        top.configure(bg=BG_COLOR)
        tk.Label(top, text="Paste column of names:", bg=BG_COLOR, fg="white").pack(pady=10)
        txt = tk.Text(top, height=20, width=40, bg=ENTRY_BG, fg="white", insertbackground="white")
        txt.pack(padx=10, pady=5)
        def process():
            c = txt.get("1.0", tk.END).strip()
            if not c: return
            lines = [l.strip() for l in c.split('\n') if l.strip()]
            self.num_samples.set(len(lines))
            self.update_all()
            for i, name in enumerate(lines):
                if i < len(self.sample_entries):
                    self.sample_entries[i].delete(0, tk.END)
                    self.sample_entries[i].insert(0, name)
            top.destroy()
        tk.Button(top, text="Apply", command=process, bg="#388e3c", fg="white").pack(pady=10)

    def update_all(self, *args):
        try: n = self.num_samples.get()
        except: n = 0
        current = len(self.sample_entries)
        if n > current:
            for i in range(current, n):
                f = tk.Frame(self.sample_list_frame, bg=PANEL_BG)
                f.pack(fill="x", pady=2, padx=10)
                tk.Label(f, text=f"{i+1}:", width=3, anchor="w", bg=PANEL_BG, fg=TEXT_COLOR).pack(side="left")
                e = tk.Entry(f, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground="white")
                e.pack(side="right", fill="x", expand=True)
                self.sample_entries.append(e)
        elif n < current:
            for i in range(current - 1, n - 1, -1):
                self.sample_entries[i].master.destroy()
                self.sample_entries.pop()
        self.update_visual_only()

    def update_visual_only(self, *args):
        try:
            s, d = self.num_samples.get(), self.droplets_per_sample.get()
            start_s = self.start_sample_idx.get()
            
            max_tips = 0
            if self.gen_combined.get():
                max_tips = s + (s * d)
            elif self.gen_spot.get():
                max_tips = s * d
            elif self.gen_mix.get():
                max_tips = s
                
            tip_racks = math.ceil(max_tips / 96) if max_tips > 0 else 0
            
            plate_cap = len(VALID_WELLS)
            total_spots = s * d
            if total_spots <= plate_cap:
                self.overflow_strategy = "none"
            elif self.overflow_strategy == "none":
                self.overflow_strategy = "swap" 

            self.info_label.config(text=f"Est. Max Tips Needed: {max_tips}\nMax Tip Racks: {tip_racks}\nTube Racks: {math.ceil(s/24)}\nRecovery: Starting @ Sample {start_s}")
                
            self.draw_plate_canvas(s, d, start_s)
            self.draw_deck_canvas(s, start_s)
        except: pass

    def get_well_status(self, samples, droplets, start_s):
        status = {}
        if not self.gen_spot.get() and not self.gen_combined.get(): return status
        
        plate_capacity = len(self.sorted_wells_cache)
        global_cursor = 0
        
        for s in range(samples):
            skipped = (s + 1) < start_s
            color = COLORS_HEX[s % len(COLORS_HEX)] if not skipped else "#444"
            for _ in range(droplets):
                plate_num = global_cursor // plate_capacity
                well_idx = global_cursor % plate_capacity
                w_name = self.sorted_wells_cache[well_idx]
                
                status[(plate_num, w_name)] = (s, color, skipped)
                global_cursor += 1
        return status

    def draw_plate_canvas(self, samples, droplets, start_s):
        self.canvas_plate.delete("all")
        status = self.get_well_status(samples, droplets, start_s)
        if not status: return
        
        num_plates = max([p for p, w in status.keys()]) + 1
        self.canvas_plate.config(scrollregion=(0, 0, 600, num_plates * 400))
        
        cell = 18
        pad_x = 30
        
        for p in range(num_plates):
            pad_y = 50 + (p * 380)
            title = "Main Destination (Slot 3)" if p == 0 else f"Overflow Destination Locations (Plate {p+1})"
            self.canvas_plate.create_text(300, pad_y - 20, text=title, fill="white", font=("Arial", 12, "bold"))
            
            for r in range(16):
                for c in range(24):
                    w_name = f"{chr(65+r)}{c+1}"
                    x, y = pad_x + c*cell, pad_y + r*cell
                    
                    if (p, w_name) in status:
                        s_idx, fill, skipped = status[(p, w_name)]
                        outline = "white" if not skipped else "#555"
                        text_col = "black" if not skipped else "white"
                        self.canvas_plate.create_oval(x, y, x+14, y+14, fill=fill, outline=outline)
                        self.canvas_plate.create_text(x+7, y+7, text=str(s_idx+1), fill=text_col, font=("Arial", 6, "bold"))
                    elif w_name in VALID_WELLS:
                        self.canvas_plate.create_oval(x, y, x+14, y+14, fill="#555", outline="#777")
                    else:
                        self.canvas_plate.create_oval(x, y, x+14, y+14, outline="#4a4a4a")

    def draw_deck_canvas(self, samples, start_s):
        self.canvas_deck.delete("all")
        rack_capacity = 24
        req_racks = math.ceil(samples / rack_capacity)
        active_slots = [1, 4, 7, 10][:min(req_racks, 4)]
        display_order = active_slots[::-1] 
        start_x, start_y = 100, 50
        for visual_idx, slot in enumerate(display_order):
            visual_y = start_y + (visual_idx * 230)
            self.canvas_deck.create_rectangle(start_x, visual_y, start_x+360, visual_y+200, fill="#333", outline="white", width=2)
            self.canvas_deck.create_text(start_x-40, visual_y+100, text=f"Slot {slot}", font=("Arial", 16, "bold"), fill="white")
            rack_index = {1:0, 4:1, 7:2, 10:3}.get(slot, 0)
            for t_idx in range(24):
                global_sample_idx = (rack_index * 24) + t_idx
                if global_sample_idx >= samples: break
                row = t_idx % 4 
                col = t_idx // 4
                tx = start_x + 40 + (col * 50)
                ty = visual_y + 30 + (row * 40)
                skipped = (global_sample_idx + 1) < start_s
                color = COLORS_HEX[global_sample_idx % len(COLORS_HEX)] if not skipped else "#555"
                outline = "white" if not skipped else "#777"
                text_col = "black" if not skipped else "white"
                
                self.canvas_deck.create_oval(tx, ty, tx+25, ty+25, fill=color, outline=outline)
                self.canvas_deck.create_text(tx+12, ty+12, text=str(global_sample_idx+1), fill=text_col, font=("Arial", 8, "bold"))

    def export_pdf(self):
        if not REPORTLAB_AVAILABLE: 
            messagebox.showerror("Error", "ReportLab not installed")
            return
        try:
            s, d = self.num_samples.get(), self.droplets_per_sample.get()
            vol, asp, disp = self.droplet_volume.get(), self.asp_rate.get(), self.disp_rate.get()
            b_ht, b_rt = self.blowout_height.get(), self.blowout_rate.get()
            start_s, start_tip = self.start_sample_idx.get(), self.start_tip.get()
            base_name = self.protocol_name.get().strip() or "Protocol_Report"
            names = [e.get() if e.get() else f"Sample {i+1}" for i, e in enumerate(self.sample_entries)]
        except: return

        filename = filedialog.asksaveasfilename(initialfile=base_name, defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not filename: return
        c = pdf_canvas.Canvas(filename, pagesize=landscape(letter))
        w, h = landscape(letter)
        
        well_status = self.get_well_status(s, d, start_s)
        num_plates = max([p for p, w in well_status.keys()]) + 1 if well_status else 0

        for p in range(num_plates):
            c.setFont("Helvetica-Bold", 18)
            title = f"Opentrons Report: {base_name}" if p == 0 else f"Overflow Destination Locations (Plate {p+1})"
            c.drawString(30, h - 40, title)
            
            c.setFont("Helvetica", 10)
            if p == 0:
                c.drawString(30, h - 60, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
                c.drawString(30, h - 75, f"Config: {s} Samples | {d} Drops | {vol} µL")
                
                blowout_str = f"ON (Ht: {b_ht} mm, Rt: {b_rt} µL/s)" if self.do_blowout.get() else "OFF"
                c.drawString(30, h - 90, f"Physics: Asp {asp} | Disp {disp} | Blowout: {blowout_str}")
                c.drawString(30, h - 105, f"Recovery: Start Sample {start_s} | First Tip {start_tip}")

            cell_size = 20
            start_x = (w - (24 * cell_size)) / 2 
            start_y = h - 150
            
            for r in range(16):
                row_char = chr(ord('A') + r)
                y_pos = start_y - (r * cell_size)
                for col in range(24):
                    col_num = col + 1
                    w_name = f"{row_char}{col_num}"
                    x_pos = start_x + (col * cell_size)
                    
                    if (p, w_name) in well_status:
                        s_idx, hex_c, skipped = well_status[(p, w_name)]
                        c.setFillColor(pdf_colors.HexColor(hex_c))
                        c.circle(x_pos, y_pos, 7, stroke=1, fill=1)
                        c.setFillColor(pdf_colors.black)
                        c.setFont("Helvetica-Bold", 6)
                        c.drawCentredString(x_pos, y_pos - 2, str(s_idx + 1))
                    elif w_name in VALID_WELLS:
                        c.setFillColor(pdf_colors.lightgrey)
                        c.circle(x_pos, y_pos, 7, stroke=1, fill=1)
                    else:
                        c.setFillColor(pdf_colors.whitesmoke)
                        c.circle(x_pos, y_pos, 2, stroke=0, fill=1)
            
            c.setFillColor(pdf_colors.black)
            c.setFont("Helvetica", 6)
            for col in range(24): c.drawCentredString(start_x + (col * cell_size), start_y + 15, str(col+1))
            for r in range(16): c.drawString(start_x - 20, start_y - (r * cell_size) - 3, chr(ord('A')+r))
            c.showPage() 

        c.setPageSize(portrait(letter))
        w, h = portrait(letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(30, h - 40, "Opentrons Spotting Report - Source Deck Layout")
        rack_capacity = 24
        req_racks = math.ceil(s / rack_capacity)
        active_slots = [1, 4, 7, 10][:min(req_racks, 4)]
        display_order = active_slots[::-1] 
        rack_x = 100
        start_y = h - 100
        
        for visual_idx, slot in enumerate(display_order):
            visual_y = start_y - (visual_idx * 200)
            c.setStrokeColor(pdf_colors.black)
            c.rect(rack_x, visual_y - 150, 300, 150, fill=0)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(rack_x - 60, visual_y - 20, f"Slot {slot}")
            rack_index = {1:0, 4:1, 7:2, 10:3}.get(slot, 0)
            
            for t_idx in range(24):
                global_sample_idx = (rack_index * 24) + t_idx
                if global_sample_idx >= s: break
                row = t_idx % 4 
                col = t_idx // 4
                tx = rack_x + 30 + (col * 40)
                ty = visual_y - 30 - (row * 30)
                
                skipped = (global_sample_idx + 1) < start_s
                hex_c = COLORS_HEX[global_sample_idx % len(COLORS_HEX)] if not skipped else "#999999"
                c.setFillColor(pdf_colors.HexColor(hex_c))
                c.circle(tx, ty, 10, stroke=1, fill=1)
                
                c.setFillColor(pdf_colors.black if not skipped else pdf_colors.white)
                c.setFont("Helvetica-Bold", 8)
                c.drawCentredString(tx, ty - 3, str(global_sample_idx + 1))
        c.showPage()

        c.setPageSize(portrait(letter))
        w, h = portrait(letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(30, h - 40, "Sample Key")
        
        c.setFont("Helvetica", 10)
        key_start_x = 40
        key_start_y = h - 80
        col_width = 180
        line_height = 15
        max_y = 50
        
        current_x = key_start_x
        current_y = key_start_y
        
        for i, name in enumerate(names):
            col = pdf_colors.HexColor(COLORS_HEX[i % len(COLORS_HEX)])
            c.setFillColor(col)
            c.circle(current_x, current_y + 3, 5, fill=1)
            c.setFillColor(pdf_colors.black)
            c.drawString(current_x + 10, current_y, f"{i+1}: {name}")
            
            current_y -= line_height
            
            if current_y < max_y:
                current_y = key_start_y
                current_x += col_width
                
                if current_x + col_width > w:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 18)
                    c.drawString(30, h - 40, "Sample Key (Continued)")
                    c.setFont("Helvetica", 10)
                    current_x = key_start_x
                    current_y = key_start_y
                    
        c.save()
        messagebox.showinfo("Exported", f"Report saved to:\n{filename}")

    def ask_overflow_strategy(self):
        top = tk.Toplevel(self.root)
        top.title("Capacity Exceeded")
        top.geometry("450x220")
        top.configure(bg=BG_COLOR)
        top.transient(self.root)
        top.grab_set()

        self.strat_choice = tk.StringVar(value="cancel")
        
        tk.Label(top, text="Required spots exceed the capacity of a single plate.", bg=BG_COLOR, fg="white", font=("Segoe UI", 11, "bold")).pack(pady=10)
        
        def set_strat(s):
            self.strat_choice.set(s)
            top.destroy()

        tk.Button(top, text="Use Overflow Slot 11 (Dual Plate)", command=lambda: set_strat("slot11"), bg="#388e3c", fg="white", font=("Arial", 10)).pack(pady=5, fill="x", padx=50)
        tk.Button(top, text="Pause & Swap Slot 3 (Single Plate)", command=lambda: set_strat("swap"), bg="#f57c00", fg="white", font=("Arial", 10)).pack(pady=5, fill="x", padx=50)
        tk.Button(top, text="Cancel (Adjust Settings)", command=lambda: set_strat("cancel"), bg="#d32f2f", fg="white", font=("Arial", 10)).pack(pady=5, fill="x", padx=50)
        
        self.root.wait_window(top)
        return self.strat_choice.get()

    def check_and_generate(self):
        modes_to_gen = []
        if self.gen_mix.get(): modes_to_gen.append("Mix")
        if self.gen_spot.get(): modes_to_gen.append("Spot")
        if self.gen_combined.get(): modes_to_gen.append("Combined")

        if not modes_to_gen:
            messagebox.showwarning("Warning", "Please select at least one action (Mix, Spot, Combined) to generate.")
            return

        try:
            s, d = self.num_samples.get(), self.droplets_per_sample.get()
            vol = self.droplet_volume.get()
            asp, disp = self.asp_rate.get(), self.disp_rate.get()
            m_vol = self.mix_volume.get()
            
            b_ht, b_rt = self.blowout_height.get(), self.blowout_rate.get()
            
            if not (0.2 <= vol <= 3.0): raise ValueError("Dispense Volume error")
            if not (0.1 <= asp <= 25.0): raise ValueError("Physics error (Asp Rate)")
            if not (0.1 <= disp <= 25.0): raise ValueError("Physics error (Disp Rate)")
            if not (1.0 <= m_vol <= 20.0): raise ValueError("Mix Volume must be between 1 and 20 µL")
            if not (0.1 <= b_ht <= 10.0): raise ValueError("Blowout Height error")
            if not (0.1 <= b_rt <= 25.0): raise ValueError("Blowout Rate error")
            
            needs_spotting = "Spot" in modes_to_gen or "Combined" in modes_to_gen
            if needs_spotting:
                total_spots = s * d
                plate_capacity = len(VALID_WELLS)
                if total_spots > plate_capacity:
                    choice = self.ask_overflow_strategy()
                    if choice == "cancel":
                        return
                    else:
                        self.overflow_strategy = choice
                        self.generate_files(modes_to_gen)
                else:
                    self.overflow_strategy = "none"
                    self.generate_files(modes_to_gen)
            else:
                self.overflow_strategy = "none"
                self.generate_files(modes_to_gen)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_files(self, modes):
        names = [e.get() if e.get() else f"Sample {i+1}" for i, e in enumerate(self.sample_entries)]
        base_name = self.protocol_name.get().strip() or "Protocol"
        
        path = filedialog.asksaveasfilename(initialfile=base_name, defaultextension=".py", filetypes=[("Python", "*.py")])
        if not path: return
        
        base_dir = os.path.dirname(path)
        chosen_filename = os.path.basename(path).replace(".py", "")
        
        generated_files = []
        for mode in modes:
            mode_filename = f"{chosen_filename}_{mode}"
                
            full_path = os.path.join(base_dir, f"{mode_filename}.py")
            code = self.get_protocol_code(names, mode, mode_filename)
            
            with open(full_path, "w") as f:
                f.write(code)
                
            if mode in ["Spot", "Combined"]:
                self.generate_json_manifest(full_path, names, mode)
                
            generated_files.append(mode_filename + ".py")
            self.last_path = full_path 
            
        messagebox.showinfo("Success", "Generated Protocols:\n" + "\n".join(generated_files))

    def generate_json_manifest(self, filepath, names, mode):
        s, d = self.num_samples.get(), self.droplets_per_sample.get()
        start_s = self.start_sample_idx.get()
        manifest = []
        ordered_wells = sorted(VALID_WELLS, key=get_well_sort_key)
        cursor = 0
        plate_offset = 0
        total_capacity = len(ordered_wells) * 2 if self.overflow_strategy == "slot11" else len(ordered_wells)
        
        for i in range(s):
            if (i+1) < start_s:
                cursor += d
                while cursor >= total_capacity:
                    cursor -= total_capacity
                    plate_offset += (2 if self.overflow_strategy == "slot11" else 1)
                continue
                
            sample_name = names[i]
            for drop_idx in range(d):
                if cursor >= total_capacity:
                    cursor = 0
                    plate_offset += (2 if self.overflow_strategy == "slot11" else 1)
                    
                if self.overflow_strategy == "slot11":
                    if cursor < len(ordered_wells):
                        well = ordered_wells[cursor]
                        plate = 1 + plate_offset
                    else:
                        well = ordered_wells[cursor - len(ordered_wells)]
                        plate = 2 + plate_offset
                else:
                    well = ordered_wells[cursor]
                    plate = 1 + plate_offset
                    
                entry = {
                    "sample_index": i + 1,
                    "sample_name": sample_name,
                    "droplet_number": drop_idx + 1,
                    "destination_plate": plate,
                    "destination_well": well,
                    "protocol_type": mode
                }
                manifest.append(entry)
                cursor += 1
                
        json_path = filepath.replace(".py", ".json")
        with open(json_path, "w") as jf:
            json.dump(manifest, jf, indent=4)

    def simulate(self):
        if not self.last_path: return
        try:
            res = subprocess.run(['opentrons_simulate', self.last_path], capture_output=True, text=True)
            top = tk.Toplevel(self.root)
            top.configure(bg=BG_COLOR)
            top.title("Simulation Output")
            txt = tk.Text(top, width=90, height=30, bg=ENTRY_BG, fg="white", insertbackground="white")
            txt.pack(padx=10, pady=10)
            txt.insert("1.0", res.stdout if res.returncode == 0 else res.stderr)
        except: messagebox.showerror("Error", "Simulate failed.")

    def get_protocol_code(self, sample_names, mode, final_protocol_name):
        s, d, v = self.num_samples.get(), self.droplets_per_sample.get(), self.droplet_volume.get()
        asp, disp = self.asp_rate.get(), self.disp_rate.get()
        m_vol, m_cyc = self.mix_volume.get(), self.mix_cycles.get()
        start_s, start_tip = self.start_sample_idx.get(), self.start_tip.get()
        
        b_ht, b_rt = self.blowout_height.get(), self.blowout_rate.get()
        
        do_mix = mode in ["Mix", "Combined"]
        do_spot = mode in ["Spot", "Combined"]
        do_blowout = self.do_blowout.get()
            
        colors_py = str(COLORS_HEX)
        
        if self.overflow_strategy == "slot11":
            tip_slots = [2, 5, 8, 9] 
        else:
            tip_slots = [2, 5, 8, 11]

        return f"""
from opentrons import protocol_api
import math

metadata = {{'protocolName': '{final_protocol_name}', 'apiLevel': '2.14'}}

SAMPLE_NAMES = {sample_names}
NUM_SAMPLES = {s}
DROPLETS = {d}
VOL = {v}
VALID_WELLS = {VALID_WELLS}
COLORS = {colors_py}
STRATEGY = '{self.overflow_strategy}'
START_SAMPLE_INDEX = {start_s}
START_TIP_WELL = '{start_tip}'
ENABLE_MIX = {do_mix}
ENABLE_SPOT = {do_spot}
MIX_VOL = {m_vol}
MIX_CYCLES = {m_cyc}

# Blowout Configuration
ENABLE_BLOWOUT = {do_blowout}
BLOWOUT_HEIGHT = {b_ht}
BLOWOUT_RATE = {b_rt}

def run(protocol: protocol_api.ProtocolContext):
    plate1 = protocol.load_labware('spotting_384_wellplate_10ul', 3) if ENABLE_SPOT else None
    plate2 = protocol.load_labware('spotting_384_wellplate_10ul', 11) if (STRATEGY == "slot11" and ENABLE_SPOT) else None
    
    total_tips = 0
    if ENABLE_MIX: total_tips += NUM_SAMPLES
    if ENABLE_SPOT: total_tips += (NUM_SAMPLES * DROPLETS)
    
    tip_slots = {tip_slots}
    req_tip_racks = math.ceil(total_tips / 96)
    tips_to_load = []
    for i in range(min(req_tip_racks, len(tip_slots))):
        tips_to_load.append(protocol.load_labware('opentrons_96_tiprack_20ul', tip_slots[i]))
        
    p20 = protocol.load_instrument('p20_single_gen2', 'left', tip_racks=tips_to_load)
    if tips_to_load and START_TIP_WELL:
        p20.starting_tip = tips_to_load[0].wells_by_name()[START_TIP_WELL]
    
    src_slots = [1, 4, 7, 10]
    req_src_racks = math.ceil(NUM_SAMPLES / 24)
    sources = [protocol.load_labware('opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap', src_slots[i]) for i in range(min(req_src_racks, 4))]
    
    for i in range(NUM_SAMPLES):
        rack_idx = i // 24
        tube_idx = i % 24
        well = sources[rack_idx].wells()[tube_idx]
        name = SAMPLE_NAMES[i]
        color = COLORS[i % len(COLORS)]
        liq = protocol.define_liquid(name=name, description=f"Sample {{i+1}}", display_color=color)
        well.load_liquid(liquid=liq, volume=500)

    p20.flow_rate.aspirate = {asp}
    p20.flow_rate.dispense = {disp}
    p20.flow_rate.blow_out = BLOWOUT_RATE

    def sort_key(w):
        r, c = w[0], int(w[1:])
        if 3<=c<=6: cb=0
        elif 9<=c<=12: cb=1
        elif 15<=c<=18: cb=2
        elif 21<=c<=24: cb=3
        else: cb=99
        ridx = ord(r) - ord('A')
        if ridx<=4: rb=0
        elif 6<=ridx<=9: rb=1
        elif 11<=ridx<=15: rb=2
        else: rb=99
        grp = (rb*4) + cb + 1
        snake = {{4:1,3:2,2:3,1:4, 5:5,6:6,7:7,8:8, 12:9,11:10,10:11,9:12}}.get(grp, 99)
        return (snake, ridx, -c)

    ordered = sorted(VALID_WELLS, key=sort_key)
    cursor = 0
    total_capacity = len(ordered) * 2 if STRATEGY == "slot11" else len(ordered)
    
    for i in range(NUM_SAMPLES):
        if (i + 1) < START_SAMPLE_INDEX:
            if ENABLE_SPOT:
                cursor += DROPLETS
                while cursor >= total_capacity:
                    cursor -= total_capacity
            continue
            
        rack_idx = i // 24
        tube_idx = i % 24
        src = sources[rack_idx].wells()[tube_idx]
        
        protocol.comment(f"Processing {{SAMPLE_NAMES[i]}} (Sample {{i+1}})")
        
        if ENABLE_MIX:
            p20.pick_up_tip()
            p20.mix(MIX_CYCLES, MIX_VOL, src.bottom(2))
            p20.drop_tip()
            
        if ENABLE_SPOT:
            for _ in range(DROPLETS):
                p20.pick_up_tip()
                if cursor >= total_capacity:
                    swap_msg = "Plate(s) full! Please replace plate in Slot 3"
                    if STRATEGY == "slot11":
                        swap_msg += " and Slot 11"
                    swap_msg += " with fresh empty plates, then click Resume."
                    protocol.pause(swap_msg)
                    cursor = 0
                    
                current_plate = plate1
                well_target_name = ""
                if STRATEGY == "slot11":
                    if cursor < len(ordered):
                        current_plate = plate1
                        well_target_name = ordered[cursor]
                    else:
                        current_plate = plate2
                        well_target_name = ordered[cursor - len(ordered)]
                else:
                    current_plate = plate1
                    well_target_name = ordered[cursor]

                dest = current_plate.wells_by_name()[well_target_name]
                
                # Standard Aspiration
                p20.move_to(src.bottom(1), speed=125)
                protocol.delay(seconds=0)
                p20.aspirate(VOL, src.bottom(1))
                protocol.delay(seconds=1) 
                p20.move_to(src.top(), speed=125)
                
                # Dispense
                p20.move_to(dest.bottom(0.2), speed=125)
                protocol.delay(seconds=0)
                p20.dispense(VOL, dest.bottom(0.2))
                
                if ENABLE_BLOWOUT:
                    protocol.delay(seconds=1)
                    p20.move_to(dest.bottom(BLOWOUT_HEIGHT), speed=10)
                    p20.blow_out()
                
                protocol.delay(seconds=1)
                p20.move_to(dest.bottom(3.0), speed=10)
                
                p20.drop_tip()
                cursor += 1
"""

if __name__ == "__main__":
    root = tk.Tk()
    app = SpottingApp(root)
    root.mainloop()