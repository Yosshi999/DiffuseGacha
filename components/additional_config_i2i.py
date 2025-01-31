from dataclasses import dataclass
import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib, GdkPixbuf, Gdk
from PIL import Image
import numpy as np

from utils.template import TemplateX
from utils.pipes import CanvasMemory, decode_latent
from utils.imutil import load_image_with_metadata
from components.mask_editor_window import MaskEditorWindow
import gstate

@TemplateX("components/additional_config_i2i.uix")
class AdditionalConfigI2I(Gtk.Box):
    __gtype_name__ = "AdditionalConfigI2I"
    guidance_scale = Gtk.Template.Child()
    guidance_rescale = Gtk.Template.Child()
    num_inference_steps = Gtk.Template.Child()
    denoising_strength = Gtk.Template.Child()
    target_image = Gtk.Template.Child()
    inpaint_mode = Gtk.Template.Child()
    open_mask_button = Gtk.Template.Child()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = CanvasMemory(None, None, None)
        self.mask = None
        self.mask_window = None
        self.size = self.target_image.get_pixel_size()
        # drag and drop to open image
        drop = Gtk.DropTargetAsync(
            actions=Gdk.DragAction.COPY,
            formats=Gdk.ContentFormats.new(["text/uri-list"])  # URL to the image
        )
        self.target_image.add_controller(drop)
        drop.connect("drop", self.on_drop_image)
        # inpaint mode toggle
        self.inpaint_mode.connect("toggled", lambda button: self.open_mask_button.set_sensitive(button.get_active()))
        self.visualize()
    
    @property
    def window(self):
        return self.get_ancestor(Gtk.ApplicationWindow)
    
    def visualize(self):
        image = self.memory.image
        if self.mask_window is not None:
            self.mask_window.set_background(image)
        if image is None:
            # set black image
            image = Image.new("RGB", (self.size, self.size))
        image = image.copy()
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
        self.memory.copy_from(self.window.request_memory())
        self.visualize()

    @Gtk.Template.Callback()
    def on_open_button_clicked(self, button):
        gstate.load_memory_with_dialog(self.window, self.set_memory)
    
    @Gtk.Template.Callback()
    def on_open_mask_button_clicked(self, button):
        if self.mask_window is None:
            self.mask_window = MaskEditorWindow()
            self.mask_window.set_hide_on_close(True)
        self.mask_window.present()  # make this window active
        self.mask_window.set_background(self.memory.image)

    def set_memory(self, mem):
        self.memory.copy_from(mem)
        self.visualize()
    
    def get_current_latent(self):
        if self.memory.latent is None:
            return None
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
        if len(files) > 1:
            Gtk.AlertDialog(
                message=f"Notice",
                detail="Only one file will be opened."
            ).show(self.window)
        file = files[0]
        try:
            mem = gstate.load_memory(file.get_path())
            self.set_memory(mem)
        except gstate.ModelNotInitializedError:
            Gtk.AlertDialog(
                message=f"Error opening file",
                detail="To open an image, the diffusion model must be loaded. Please try agin later."
            ).show(self.window)

    def get_config(self):
        if self.inpaint_mode.get_active() and self.mask_window is not None:
            mask_image = self.mask_window.get_mask()
        else:
            mask_image = None

        return {
            "guidance_scale": self.guidance_scale.get_value(),
            "guidance_rescale": self.guidance_rescale.get_value(),
            "num_inference_steps": self.num_inference_steps.get_value_as_int(),
            "strength": self.denoising_strength.get_value(),
            "ddim_inversion": False,
            "mask_image": mask_image,
        }