import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk

class SampleDropPanel:
    def __init__(self, master, stage_control_module, circle_radius=15, spacing=40):
        self.master = master
        self.stage_control_module = stage_control_module  # Reference to CameraControl
        self.circle_radius = circle_radius
        self.spacing = spacing
        self.current_position = None  # To track the current position

        self.grid_pattern = [3, 5, 5, 3]  # Column-wise circle counts for symmetry

        max_rows = max(self.grid_pattern)
        total_width = len(self.grid_pattern) * spacing
        total_height = max_rows * spacing

        self.canvas = tk.Canvas(master, width=total_width, height=total_height, bg='white')
        self.canvas.pack()

        self.tooltip = tk.Label(master, text="", bg="yellow", relief=tk.SOLID, bd=1)
        self.create_circle_grid()

    def create_circle_grid(self):
        for col_index, num_circles in enumerate(self.grid_pattern):
            vertical_offset = (max(self.grid_pattern) - num_circles) * self.spacing // 2
            for row_index in range(num_circles):
                x = col_index * self.spacing + self.spacing // 2
                y = vertical_offset + row_index * self.spacing + self.spacing // 2
                circle = self.canvas.create_oval(
                    x - self.circle_radius, y - self.circle_radius,
                    x + self.circle_radius, y + self.circle_radius,
                    fill='skyblue', outline='black', tags=f"circle_{row_index}_{col_index}"
                )
                # Bind click event to each circle
                self.canvas.tag_bind(circle, '<Button-1>', lambda event, row=row_index, col=col_index: self.move_stage(row, col))
                # Bind mouse over and leave events for hover effect and tooltip
                self.canvas.tag_bind(circle, '<Enter>', lambda event, row=row_index, col=col_index: self.on_hover(event, row, col))
                self.canvas.tag_bind(circle, '<Leave>', self.on_leave)

    def move_stage(self, row, col):
        # Reset the previous circle color
        if self.current_position:
            prev_row, prev_col = self.current_position
            self.canvas.itemconfig(f"circle_{prev_row}_{prev_col}", fill='skyblue')

        # Highlight the selected circle
        self.canvas.itemconfig(f"circle_{row}_{col}", fill='green')
        self.current_position = (row, col)
        print(f"Moving XY stage to position: Row {row}, Column {col}")

        # Trigger CameraControl to take a picture
        # self.stage_control_module.capture_image(row, col)

    def on_hover(self, event, row, col):
        # Prevent changing color if hovering over the current position
        if self.current_position != (row, col):
            self.canvas.itemconfig(f"circle_{row}_{col}", fill='lightgreen')
        self.tooltip.config(text="Move to this drop")
        self.tooltip.place(x=event.x_root - self.master.winfo_rootx() + 10, y=event.y_root - self.master.winfo_rooty() + 10)

    def on_leave(self, event):
        for col_index, num_circles in enumerate(self.grid_pattern):
            for row_index in range(num_circles):
                if self.current_position != (row_index, col_index):
                    self.canvas.itemconfig(f"circle_{row_index}_{col_index}", fill='skyblue')
        self.tooltip.place_forget()


class MockStageControl:
    def __init__(self, master):
        self.master = master
        self.frame = tb.Frame(master, padding=10)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.label = tb.Label(self.frame, text="Camera Control Module")
        self.label.pack()

    def mock_move_stage(self, row, col):
        print(f"Moving stage to position: Row {row}, Column {col}")


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Control System")

        # Create frames for each module
        self.xy_frame = tb.Frame(root, padding=10)
        self.xy_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.mock_stage_control_frame = tb.Frame(root, padding=10)
        self.mock_stage_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Initialize modules
        self.mock_stage_control_module = MockStageControl(self.mock_stage_control_frame)
        self.sample_drop_panel = SampleDropPanel(self.xy_frame, self.mock_stage_control_module)


def main():
    root = tb.Window(themename="cosmo")
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
