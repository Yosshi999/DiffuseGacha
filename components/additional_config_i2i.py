import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib, GdkPixbuf, Gdk
from PIL import Image

from utils.template import TemplateX
from utils.pipes import CanvasMemory, decode_latent
from utils.imutil import load_image_with_metadata

@TemplateX("components/additional_config_i2i.uix")
class AdditionalConfigI2I(Gtk.Box):
    __gtype_name__ = "AdditionalConfigI2I"
    guidance_scale = Gtk.Template.Child()
    guidance_rescale = Gtk.Template.Child()
    num_inference_steps = Gtk.Template.Child()
    denoising_strength = Gtk.Template.Child()
    target_image = Gtk.Template.Child()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = CanvasMemory(None, None, None)
        self.size = self.target_image.get_pixel_size()
        self.request_memory = None
        # drag and drop to open image
        drop = Gtk.DropTargetAsync(
            actions=Gdk.DragAction.COPY,
            formats=Gdk.ContentFormats.new(["text/uri-list"])  # URL to the image
        )
        self.target_image.add_controller(drop)
        drop.connect("drop", self.on_drop_image)
        self.visualize()
    
    def visualize(self):
        image = self.memory.image
        if image is None:
            # set black image
            image = Image.new("RGB", (self.size, self.size))
        image.thumbnail(size=(self.size, self.size))
        thumb = Image.new("RGB", (self.size, self.size))
        thumb.paste(image, (int((self.size - image.width) / 2), int((self.size - image.height) / 2)))

        w, h = thumb.size
        data = thumb.tobytes()
        data = GLib.Bytes.new(data)
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
        self.target_image.set_from_pixbuf(pixbuf.copy())
    
    @Gtk.Template.Callback()
    def on_set_target_button_clicked(self, button):
        self.memory = self.request_memory()
        self.visualize()

    @Gtk.Template.Callback()
    def on_open_button_clicked(self, button):
        self.request_openwindow(self.set_memory)

    def set_memory(self, mem):
        if mem is not None:
            self.memory = mem
            self.visualize()
    
    def set_request_memory_callback(self, func):
        self.request_memory = func
    
    def set_request_openwindow_callback(self, func):
        self.request_openwindow = func
    
    def set_request_open_callback(self, func):
        self.request_open = func
    
    def get_current_latent(self):
        return self.memory.latent.clone()
    
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
        file = files[0]
        mem = self.request_open(file.get_path())
        self.set_memory(mem)

    def get_config(self):
        return {
            "guidance_scale": self.guidance_scale.get_value(),
            "guidance_rescale": self.guidance_rescale.get_value(),
            "num_inference_steps": self.num_inference_steps.get_value_as_int(),
            "strength": self.denoising_strength.get_value(),
        }