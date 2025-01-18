from diffusers import DiffusionPipeline
import torch
from utils.imutil import mitsua_credit
import threading
import asyncio
from PIL import Image
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GLib, GdkPixbuf

from utils.template import TemplateX

@TemplateX("components/window.uix")
class Window(Gtk.ApplicationWindow):
    __gtype_name__ = "Window"
    generate_button = Gtk.Template.Child()
    prompt = Gtk.Template.Child()
    progress = Gtk.Template.Child()
    gacha_result = Gtk.Template.Child()

    async def _initialize_model(self, device, dtype):
        return DiffusionPipeline.from_pretrained("Mitsua/mitsua-likes", trust_remote_code=True).to(device, dtype=dtype)

    def initialize_model(self):
        self.generate_button.set_sensitive(False)
        self.generate_button.set_label("Loading model...")
        fut = asyncio.run_coroutine_threadsafe(
            self._initialize_model(self.device, self.dtype),
            self.bg_loop
        )
        def resolve():
            self.pipe = fut.result()
            self.generate_button.set_sensitive(True)
            self.generate_button.set_label("Generate")
        fut.add_done_callback(lambda _: GLib.idle_add(resolve))

    async def _process_pipe(self, **pipe_kwargs) -> Image.Image:
        ret = self.pipe(**pipe_kwargs)
        image = ret.images[0]
        image = mitsua_credit(image)
        image.save("output.png")
        return image

    def process_pipe(self, **pipe_kwargs):
        self.generate_button.set_sensitive(False)
        self.generate_button.set_label("Generating...")

        def _on_step_end(pipe, i, t, kwargs):
            fraction = (i+1) / pipe._num_timesteps
            GLib.idle_add(lambda: self.progress.set_fraction(fraction))
            return {}
        fut = asyncio.run_coroutine_threadsafe(
            self._process_pipe(
                width=self.image_width,
                height=self.image_height,
                callback_on_step_end=_on_step_end,
                **pipe_kwargs),
            self.bg_loop
        )
        def resolve():
            image = fut.result()
            data = image.tobytes()
            w, h = image.size
            data = GLib.Bytes.new(data)
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
            self.gacha_result.set_from_pixbuf(pixbuf.copy())
            self.generate_button.set_sensitive(True)
            self.generate_button.set_label("Generate")
        fut.add_done_callback(lambda _: GLib.idle_add(resolve))

    def __init__(self):
        super().__init__()
        self.bg_loop = asyncio.new_event_loop()
        self.bg_thread = threading.Thread(target=self.bg_loop.run_forever, daemon=True)
        self.bg_thread.start()
        if torch.cuda.is_available():
            print("CUDA is available")
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = torch.float16
        self.initialize_model()
        self.image_width = 640
        self.image_height = 640

    @Gtk.Template.Callback()
    def on_generate_button_clicked(self, button):
        kwargs = self.prompt.get_generation_config()
        self.process_pipe(**kwargs)
