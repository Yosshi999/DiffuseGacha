import torch
from utils.imutil import mitsua_credit
from PIL import Image
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf, Gio
from typing import Optional
from pathlib import Path
import datetime
import copy
from functools import partial

from components.change_canvas_size_modal import ChangeCanvasSizeModal
from utils.template import TemplateX
import gstate
from gstate import CanvasMemory

@TemplateX("components/window.uix")
class Window(Gtk.ApplicationWindow):
    __gtype_name__ = "Window"
    generate_button = Gtk.Template.Child()
    prompt = Gtk.Template.Child()
    additional = Gtk.Template.Child()
    progress = Gtk.Template.Child()
    gacha_result = Gtk.Template.Child()

    def lock_ui(self, button_label: str):
        self.generate_button.set_sensitive(False)
        self.generate_button.set_label(button_label)

    def unlock_ui(self):
        self.generate_button.set_sensitive(True)
        self.generate_button.set_label("Generate")
    
    def on_draw(self, widget, cr, width, height, data):
        cr.set_source_rgba(0, 0, 0, 0)  # transparent color
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
        gstate.initialize()
        self.lock_ui("Loading model...")
        gstate.initialize_model_async(self.unlock_ui)
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
        def open_resolve(mem: Optional[CanvasMemory]):
            if mem is not None:
                self.memory = mem
                self.change_canvas_size(mem.image.width, mem.image.height)
                self.visualize_result()
        open_file_action.connect("activate", partial(self.show_open_dialog_native, open_resolve))
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
        self.additional.i2i.set_request_openwindow_callback(self.show_open_dialog_native)
        self.additional.i2i.set_request_open_callback(self.load_memory)
    
    def request_memory(self):
        return copy.deepcopy(self.memory)

    def show_open_dialog_native(self, then, action=None, _=None):
        self.open_dialog = Gtk.FileChooserNative.new(title="Open File", parent=self, action=Gtk.FileChooserAction.OPEN)
        self.open_dialog.set_modal(True)
        self.open_dialog.set_transient_for(self)
        self.open_dialog.connect("response", partial(self.open_dialog_native_callback, then))
        self.open_dialog.show()
        
    def open_dialog_native_callback(self, then, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            try:
                file = self.open_dialog.get_file()
                if file is not None:
                    print(f"File path is {file.get_path()}")
                    then(self.load_memory(file.get_path()))
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
        
        def resolve(memory: CanvasMemory):
            self.unlock_ui()
            self.progress.set_fraction(0)
            self.memory = memory
            self.save_memory()
            self.visualize_result()

        self.lock_ui("Generating...")
        gstate.process_pipe_async(
            self.additional.get_task_name(),
            kwargs,
            self.progress.set_fraction,
            resolve
        )

    def save_memory(self):
        fname = self.output_folder / (datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")
        gstate.save_memory(self.memory, fname)

    def load_memory(self, path: str) -> Optional[CanvasMemory]:
        try:
            return gstate.load_memory(path)
        except gstate.ModelNotInitializedError:
            Gtk.AlertDialog(message=f"Error opening file", detail="To open an image, the diffusion model must be loaded. Please try agin later.").show(self)
            return None

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
        mem = self.load_memory(file.get_path())
        if mem is not None:
            self.memory = mem
            self.change_canvas_size(mem.image.width, mem.image.height)
            self.visualize_result()
