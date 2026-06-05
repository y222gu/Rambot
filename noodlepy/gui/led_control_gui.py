from PIL import Image, ImageTk
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import logging
from led_controller import LEDController, led_on, led_off
from publisher_subscriber import Publisher, Subscriber
import os


logger = logging.getLogger(__name__)


class LEDControlGUI(ttk.Frame, Publisher, Subscriber):

    def __init__(self, parent, led: LEDController):
        ttk.Frame.__init__(self, parent)
        Publisher.__init__(self, ['led_on',
                                  'led_off',
                                  'led_brightness_change'])
        Subscriber.__init__(self)
        self.name = 'LEDControlGUI'
        self.parent = parent
        self.led = led

        self._is_on = False

        # Load and resize lightbulb image
        self._bulb_img = Image.open(os.path.join(os.getcwd(), "noodlepy","assets","lightbulb.png")).resize((20, 20))

        self.create_widgets()

    def create_widgets(self):

        main_frame = tk.LabelFrame(self, text='LED Control', padx=8, pady=8)
        main_frame.grid(row=0, column=0)
        main_frame.columnconfigure(2, weight=1, minsize=300)

        # Toggle button
        self._bulb_photo = ImageTk.PhotoImage(self._bulb_img)
        self._led_button = ttk.Button(
            main_frame,
            image=self._bulb_photo,
            command=self._toggle,
            width=4,
            bootstyle='secondary'
        )
        self._led_button.grid(row=0, column=0, padx=(4, 8), pady=4)

        # 0% label 
        ttk.Label(main_frame, text='0%').grid(row=0, column=1, padx=(0, 4))

    # 100% label
        ttk.Label(main_frame, text='100%').grid(row=0, column=3, padx=(4, 0))

        # Slider 
        self._slider = ttk.Scale(
            main_frame,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            command=self._on_slider_change,
            bootstyle='info'
        )
        self._slider.grid(row=0, column=2, sticky='ew', padx=4, pady=4)

        self._slider.bind('<ButtonRelease-1>', self._on_slider_change)
      
    # Logic

    def _toggle(self):
        if self._is_on:
            self._is_on = False
            self._led_button.configure(bootstyle='secondary')  # back to blue
            self.led.turn_off()
            self.dispatch('led_off')
            print("LED turned off")
        else:
            self._is_on = True
            self._led_button.configure(bootstyle='info')  # gray when on
            brightness = int(self._slider.get())
            self.led.turn_on(brightness or 255)
            self.dispatch('led_on')
            print(f"LED turned on at brightness {brightness}")

    def _on_slider_change(self, value):
        if self._is_on:
            brightness = int(self._slider.get())
            self.led.set_brightness(brightness)
            self.dispatch('led_brightness_change', brightness)
        
    # Called by automated system

    def turn_on_auto(self):
        self._is_on = True
        self._led_button.configure(bootstyle='secondary')
        self.dispatch('led_on')

    def turn_off_auto(self):
        self._is_on = False
        self._led_button.configure(bootstyle='info')
        self.dispatch('led_off')

    def toggle_mode(self, mode):
        """Enable or disable the button (called by scan workflow)."""
        state = NORMAL if mode == 'on' else DISABLED
        self._led_button.configure(state=state)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    root = ttk.Window(themename="darkly")
    root.title("LED Control Test")

    led = LEDController()
    app = LEDControlGUI(root, led=led)
    app.pack(padx=10, pady=10)

    root.mainloop()