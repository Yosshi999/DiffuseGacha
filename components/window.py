from dataclasses import dataclass
from typing import Optional
from diffusers import DiffusionPipeline
import torch
from utils.imutil import mitsua_credit, save_image_with_metadata, load_image_with_metadata
import threading
import asyncio
from PIL import Image
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GLib, GdkPixbuf, Gio

from components.change_canvas_size_modal import ChangeCanvasSizeModal
from utils.template import TemplateX

@dataclass
class CanvasMemory:
    image: Optional[Image.Image]
    latent: Optional[torch.Tensor]  # tensor of shape (1, 8, height, width)
    generation_config: Optional[dict]

@torch.no_grad()
def mitsua_decode(self, latents) -> Image.Image:
    """Decode Latents to Image.
    Derived from https://huggingface.co/Mitsua/mitsua-likes/blob/main/pipeline_likes_base_unet.py#L994
    """
    # make sure the VAE is in float32 mode, as it overflows in float16
    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

    if needs_upcasting:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != self.vae.dtype:
        if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            self.vae = self.vae.to(latents.dtype)

    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
    has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.latent_channels, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.latent_channels, 1, 1).to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / self.vae.config.scaling_factor

    image = self.vae.decode(latents, return_dict=False)[0]

    # cast back to fp16 if needed
    if needs_upcasting:
        self.vae.to(dtype=torch.float16)
    
    image = self.image_processor.postprocess(image, output_type="pil")
    return image

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
        # image = mitsua_decode(self.pipe, self.memory.latent)[0]  # alternative to the above line
        self.memory.image = image
        image_cred = mitsua_credit(image)
        save_image_with_metadata(image_cred, "output.png", self.memory.latent, self.memory.generation_config)
        return image

    def process_pipe(self, **pipe_kwargs):
        self.generate_button.set_sensitive(False)
        self.generate_button.set_label("Generating...")
        self.memory.generation_config = {
            "width": self.image_width,
            "height": self.image_height,
            **pipe_kwargs
        }

        def _on_step_end(pipe, i, t, kwargs):
            self.memory.latent = kwargs["latents"]
            fraction = (i+1) / pipe._num_timesteps
            GLib.idle_add(lambda: self.progress.set_fraction(fraction))
            return {}
        fut = asyncio.run_coroutine_threadsafe(
            self._process_pipe(
                callback_on_step_end=_on_step_end,
                **self.memory.generation_config),
            self.bg_loop
        )
        def resolve():
            self.visualize_result()
            self.generate_button.set_sensitive(True)
            self.generate_button.set_label("Generate")
        fut.add_done_callback(lambda _: GLib.idle_add(resolve))

    def visualize_result(self):
        image = self.memory.image
        if image is None:
            return
        if self.show_credits.get_state():
            image = mitsua_credit(image)
        data = image.tobytes()
        w, h = image.size
        data = GLib.Bytes.new(data)
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
        self.gacha_result.set_from_pixbuf(pixbuf.copy())

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
        open_file_action.connect("activate", self.show_open_dialog)
        self.add_action(open_file_action)

        self.memory: CanvasMemory = CanvasMemory(None, None, None)

    def show_open_dialog(self, action, _):
        open_dialog = Gtk.FileDialog()
        open_dialog.set_title("Select a File")
        f = Gtk.FileFilter()
        f.set_name("Image files")
        f.add_mime_type("image/png")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        open_dialog.set_filters(filters)  # Set the filters for the open dialog
        open_dialog.set_default_filter(f)
        open_dialog.open(self, None, self.open_dialog_open_callback)
        
    def open_dialog_open_callback(self, dialog, result):
        try:
            file = dialog.open_finish(result)
            if file is not None:
                print(f"File path is {file.get_path()}")
                _, latent, config = load_image_with_metadata(file.get_path())
                latent = latent.to(self.pipe._execution_device)
                image = mitsua_decode(self.pipe, latent)[0]
                self.memory = CanvasMemory(image, latent, config)
                self.change_canvas_size(image.width, image.height)
                self.visualize_result()
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
        self.gacha_result.set_pixel_size(max(self.image_width, self.image_height))

    def open_canvas_size_dialog(self, action, _):
        self.modal = ChangeCanvasSizeModal(self.image_width, self.image_height, self.change_canvas_size)
        self.modal.show()

    def update_show_credits(self, action, value):
        self.show_credits.set_state(value)
        print("Show credits:", value)
        self.visualize_result()

    @Gtk.Template.Callback()
    def on_generate_button_clicked(self, button):
        kwargs = self.prompt.get_generation_config()
        self.process_pipe(**kwargs)
