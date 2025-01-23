from diffusers import DiffusionPipeline
import torch
from utils.imutil import mitsua_credit, save_image_with_metadata, load_image_with_metadata
import threading
import asyncio
from PIL import Image
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf, Gio
from pathlib import Path
import datetime
import copy

from components.change_canvas_size_modal import ChangeCanvasSizeModal
from utils.template import TemplateX
from utils.pipes import CanvasMemory, text_to_image, image_to_image, decode_latent

@TemplateX("components/window.uix")
class Window(Gtk.ApplicationWindow):
    __gtype_name__ = "Window"
    generate_button = Gtk.Template.Child()
    prompt = Gtk.Template.Child()
    additional = Gtk.Template.Child()
    progress = Gtk.Template.Child()
    gacha_result = Gtk.Template.Child()

    async def _initialize_model(self, device, dtype):
        pipe = DiffusionPipeline.from_pretrained("Mitsua/mitsua-likes", trust_remote_code=True).to(device, dtype=dtype)
        # Workaround because https://huggingface.co/Mitsua/mitsua-likes/blob/main/pipeline_likes_base_unet.py#L1035 is broken.
        pipe.run_character_detector = lambda *args: (None, None)
        return pipe

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

    async def _process_pipe(self, task_name, **pipe_kwargs) -> CanvasMemory:
        if task_name == "t2i":
            memory = text_to_image(self.pipe, **pipe_kwargs)
        elif task_name == "i2i":
            memory = image_to_image(self.pipe, **pipe_kwargs)
        return memory

    def process_pipe(self, task_name, **pipe_kwargs):
        self.generate_button.set_sensitive(False)
        self.generate_button.set_label("Generating...")
        def _on_step_end(pipe, i, t, kwargs):
            fraction = (i+1) / pipe._num_timesteps
            GLib.idle_add(lambda: self.progress.set_fraction(fraction))
            return {}
        fut = asyncio.run_coroutine_threadsafe(
            self._process_pipe(
                task_name,
                callback_on_step_end=_on_step_end,
                **pipe_kwargs),
            self.bg_loop
        )
        def resolve():
            self.generate_button.set_sensitive(True)
            self.generate_button.set_label("Generate")
            self.memory = fut.result()
            self.save_memory()
            self.visualize_result()
        fut.add_done_callback(lambda _: GLib.idle_add(resolve))
    
    def on_draw(self, widget, cr, width, height, data):
        cr.set_source_rgb(0, 0, 0)  # black
        cr.paint()
        if self.pixbuf:
            buf_width = self.pixbuf.get_width()
            buf_height = self.pixbuf.get_height()
            scale_on_fitting_height = height / buf_height
            scale_on_fitting_width = width / buf_width
            if scale_on_fitting_height < scale_on_fitting_width:
                # fit height and centering width
                scale = scale_on_fitting_height
                cr.scale(scale, scale)
                cr.translate((width - buf_width * scale) / 2, 0)
            else:
                # fit width and centering height
                scale = scale_on_fitting_width
                cr.scale(scale, scale)
                cr.translate(0, (height - buf_height * scale) / 2)
            Gdk.cairo_set_source_pixbuf(cr, self.pixbuf, 0, 0)
            cr.paint()

    def visualize_result(self):
        image = self.memory.image
        if image is None:
            return
        if self.show_credits.get_state():
            image = mitsua_credit(image)
        data = image.tobytes()
        w, h = image.size
        data = GLib.Bytes.new(data)
        self.pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
        self.gacha_result.queue_draw()

    def __init__(self, output_folder: Path):
        super().__init__()
        self.output_folder = output_folder
        self.pipe = None
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
        self.set_title(f"DiffuseGacha - {self.image_width}x{self.image_height}")

        # change canvas size
        change_canvas_size_action = Gio.SimpleAction.new(name="change_canvas_size")
        change_canvas_size_action.connect("activate", self.open_canvas_size_dialog)
        self.add_action(change_canvas_size_action)
        # show credits
        self.show_credits = Gio.SimpleAction.new_stateful("show_credits", None, GLib.Variant.new_boolean(True))
        self.show_credits.connect("change-state", self.update_show_credits)
        self.add_action(self.show_credits)
        # open file
        open_file_action = Gio.SimpleAction.new(name="open")
        open_file_action.connect("activate", self.show_open_dialog_native)
        self.add_action(open_file_action)
        # draw area
        self.gacha_result.set_draw_func(self.on_draw, None)
        # drag and drop to open image
        drop = Gtk.DropTargetAsync(
            actions=Gdk.DragAction.COPY,
            formats=Gdk.ContentFormats.new(["text/uri-list"])  # URL to the image
        )
        self.gacha_result.add_controller(drop)
        drop.connect("drop", self.on_drop_image)

        self.memory: CanvasMemory = CanvasMemory(None, None, None)
        self.pixbuf = None

        self.additional.i2i.set_request_memory_callback(self.request_memory)
    
    def request_memory(self):
        return copy.deepcopy(self.memory)

    # def show_open_dialog(self, action, _):
    #     open_dialog = Gtk.FileDialog()
    #     open_dialog.set_title("Select a File")
    #     f = Gtk.FileFilter()
    #     f.set_name("Image files")
    #     f.add_mime_type("image/png")
    #     filters = Gio.ListStore.new(Gtk.FileFilter)
    #     filters.append(f)
    #     open_dialog.set_filters(filters)  # Set the filters for the open dialog
    #     open_dialog.set_default_filter(f)
    #     open_dialog.open(self, None, self.open_dialog_open_callback)

    def show_open_dialog_native(self, action, _):
        self.open_dialog = Gtk.FileChooserNative.new(title="Open File", parent=self, action=Gtk.FileChooserAction.OPEN)
        self.open_dialog.set_modal(True)
        self.open_dialog.set_transient_for(self)
        self.open_dialog.connect("response", self.open_dialog_native_callback)
        self.open_dialog.show()
        
    # def open_dialog_open_callback(self, dialog, result):
    #     try:
    #         file = dialog.open_finish(result)
    #         if file is not None:
    #             print(f"File path is {file.get_path()}")
    #             _, latent, config = load_image_with_metadata(file.get_path())
    #             latent = latent.to(self.pipe._execution_device)
    #             image = mitsua_decode(self.pipe, latent)[0]
    #             self.memory = CanvasMemory(image, latent, config)
    #             self.change_canvas_size(image.width, image.height)
    #             self.visualize_result()
    #     except GLib.Error as error:
    #         print(error)
    #         # Gtk.AlertDialog(message=f"Error opening file", detail=error.message).show(self)
    #     except Exception as e:
    #         print(e)
    #         Gtk.AlertDialog(message=f"Error opening file", detail="This file is not supported.").show(self)

    def open_dialog_native_callback(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            try:
                file = self.open_dialog.get_file()
                if file is not None:
                    print(f"File path is {file.get_path()}")
                    self.load_memory(file.get_path())
            except GLib.Error as error:
                print(error)
                # Gtk.AlertDialog(message=f"Error opening file", detail=error.message).show(self)
            except Exception as e:
                print(e)
                Gtk.AlertDialog(message=f"Error opening file", detail="This file is not supported.").show(self)

    def change_canvas_size(self, width: int, height: int):
        self.image_width = width
        self.image_height = height
        self.set_title(f"DiffuseGacha - {self.image_width}x{self.image_height}")
        self.gacha_result.set_content_width(self.image_width)
        self.gacha_result.set_content_height(self.image_height)

    def open_canvas_size_dialog(self, action, _):
        self.modal = ChangeCanvasSizeModal(self.image_width, self.image_height, self.change_canvas_size)
        self.modal.show()

    def update_show_credits(self, action, value):
        self.show_credits.set_state(value)
        print("Show credits:", value)
        self.visualize_result()

    @Gtk.Template.Callback()
    def on_generate_button_clicked(self, button):
        kwargs = {}
        kwargs.update(self.prompt.get_config())
        kwargs.update(self.additional.get_config())
        kwargs["width"] = self.image_width
        kwargs["height"] = self.image_height

        if self.additional.get_task_name() == "i2i":
            latent = self.additional.i2i.get_current_latent()
            if latent is None:
                Gtk.AlertDialog(message=f"Invalid I2I Request", detail="Target image is not loaded.").show(self)
                return
            kwargs["latent"] = latent
        self.process_pipe(self.additional.get_task_name(), **kwargs)

    def save_memory(self):
        fname = self.output_folder / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")
        image = mitsua_credit(self.memory.image)
        save_image_with_metadata(image, fname, self.memory.latent, self.memory.generation_config)

    def load_memory(self, path: str):
        if self.pipe is None:
            Gtk.AlertDialog(message=f"Error opening file", detail="To open an image, the diffusion model must be loaded. Please try agin later.").show(self)
            return
        _, latent, config = load_image_with_metadata(path)
        latent = latent.to(self.pipe._execution_device)
        image = decode_latent(self.pipe, latent)[0]
        self.memory = CanvasMemory(image, latent, config)
        self.change_canvas_size(image.width, image.height)
        self.visualize_result()

    def on_drop_image(self, widget, drop, x, y):
        drop.read_value_async(
            Gdk.FileList,
            GLib.PRIORITY_DEFAULT,
            None,
            self.drop_image_callback,
        )
        drop.finish(Gdk.DragAction.COPY)

    def drop_image_callback(self, drop, result):
        files = drop.read_value_finish(result).get_files()
        if len(files) > 1:
            Gtk.AlertDialog(message=f"Notice", detail="Only one file will be opened.").show(self)
        file = files[0]
        self.load_memory(file.get_path())
