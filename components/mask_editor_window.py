from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import math
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf, Gio
import cairo
import numpy as np
from PIL import Image

from utils.template import TemplateX


def fit_scale_and_offset(canvas_width, canvas_height, image_width, image_height):
    scale_on_fitting_height = canvas_height / image_height
    scale_on_fitting_width = canvas_width / image_width
    if scale_on_fitting_height < scale_on_fitting_width:
        # fit height and centering width
        scale = scale_on_fitting_height
        offset = ((canvas_width - image_width * scale) / 2, 0)
    else:
        # fit width and centering height
        scale = scale_on_fitting_width
        offset = (0, (canvas_height - image_height * scale) / 2)
    return scale, offset

class Blob:
    def __init__(self, initial_x: float, initial_y: float, radius: float):
        self.points: List[Tuple[float, float]] = [(initial_x, initial_y)]
        self.radius = radius
    def update_offset(self, x: float, y: float):
        self.points.append((self.points[0][0] + x, self.points[0][1] + y))
    def draw(self, cr, color):
        cr.set_source_rgb(*color)
        cr.arc(self.points[0][0], self.points[0][1], self.radius, 0, 2 * math.pi)
        cr.fill()
        cr.set_line_width(self.radius * 2)
        for i in range(len(self.points)-1):
            cr.move_to(self.points[i][0], self.points[i][1])
            cr.line_to(self.points[i+1][0], self.points[i+1][1])
            cr.stroke()
            cr.arc(self.points[i+1][0], self.points[i+1][1], self.radius, 0, 2 * math.pi)
            cr.fill()

@dataclass
class UndoableAction:
    pass

@dataclass
class DrawAction(UndoableAction):
    blob: Blob

@dataclass
class EraseAction(UndoableAction):
    blob: Blob

@dataclass
class ClearAction(UndoableAction):
    pass

@TemplateX("components/mask_editor_window.uix")
class MaskEditorWindow(Gtk.Window):
    __gtype_name__ = "MaskEditorWindow"
    canvas = Gtk.Template.Child()
    brush_size = Gtk.Template.Child()

    def __init__(self):
        super().__init__()
        self.history: List[UndoableAction] = []
        self.pixbuf = None
        self.canvas.set_draw_func(self.on_draw, None)
        self.radius = self.brush_size.get_value()
        # mouse move
        self.cursor_point = (0, 0)
        evk = Gtk.EventControllerMotion.new()
        evk.connect("motion", self.on_mouse_motion)
        self.canvas.add_controller(evk)
        # mouse drag event
        evk = Gtk.GestureDrag.new()
        evk.connect("drag-begin", self.on_drag_begin)
        evk.connect("drag-update", self.on_drag_update)
        evk.connect("drag-end", self.on_drag_end)
        self.canvas.add_controller(evk)
        self.top_blob: Optional[Blob] = None
        self.blobs: List[Blob] = []
    
    @Gtk.Template.Callback()
    def on_clear_clicked(self, button):
        self.top_blob = None
        self.blobs.clear()
        self.history.append(ClearAction())
        self.canvas.queue_draw()
    
    def on_mouse_motion(self, motion, x, y):
        self.cursor_point = (x, y)
        self.canvas.queue_draw()

    @Gtk.Template.Callback()
    def on_change_brush_size(self, widget):
        self.radius = self.brush_size.get_value()
        center_x = self.canvas.get_width() / 2
        center_y = self.canvas.get_height() / 2
        self.cursor_point = (center_x, center_y)
        self.canvas.queue_draw()
        
    def on_drag_begin(self, drag, x, y):
        if self.pixbuf is not None:
            # has background
            self.top_blob = Blob(x, y, self.radius)
            self.canvas.queue_draw()
    
    def on_drag_update(self, drag, x, y):
        if self.top_blob is not None:
            self.top_blob.update_offset(x, y)
            self.canvas.queue_draw()
    
    def on_drag_end(self, drag, x, y):
        # commit action
        if self.top_blob is not None:
            self.blobs.append(self.top_blob)
            self.history.append(DrawAction(self.top_blob))
            self.canvas.queue_draw()
        self.top_blob = None
    
    @Gtk.Template.Callback()
    def on_save_clicked(self, button):
        mask = self.get_mask()
        if mask is not None:
            mask.save("mask.png")

    def get_mask(self) -> Optional[Image.Image]:
        if self.pixbuf:
            width = int(self.canvas.get_width())
            height = int(self.canvas.get_height())
            # draw background
            buf_width = self.pixbuf.get_width()
            buf_height = self.pixbuf.get_height()
            scale, offset = fit_scale_and_offset(width, height, buf_width, buf_height)
            with cairo.ImageSurface(cairo.FORMAT_RGB24, buf_width, buf_height) as surface:
                cr = cairo.Context(surface)
                cr.translate(-offset[0], -offset[1])
                cr.scale(1/scale, 1/scale)
                for blob in self.blobs:
                    blob.draw(cr, (1,1,1))
                mask_arr = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape(buf_height, buf_width, 4)[:,:,1:].copy()
            return Image.fromarray(mask_arr)
        else:
            return None

    def set_background(self, image: Optional[Image.Image]):
        if image is None:
            self.pixbuf = None
        else:
            data = image.tobytes()
            w, h = image.size
            data = GLib.Bytes.new(data)
            self.pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
        self.canvas.queue_draw()
    
    def on_draw(self, widget, cr, width, height, data):
        cr.set_source_rgba(0, 0, 0, 0)  # transparent color
        cr.paint()
        if self.pixbuf:
            # draw background
            buf_width = self.pixbuf.get_width()
            buf_height = self.pixbuf.get_height()
            scale, offset = fit_scale_and_offset(width, height, buf_width, buf_height)
            cr.scale(scale, scale)
            cr.translate(offset[0], offset[1])
            Gdk.cairo_set_source_pixbuf(cr, self.pixbuf, 0, 0)
            cr.paint()
            # draw mask
            cr.identity_matrix()
            for blob in self.blobs:
                blob.draw(cr, (0,0,0))
            if self.top_blob is not None:
                self.top_blob.draw(cr, (1,0,0))
            # cursor
            cr.set_source_rgb(0, 0, 1)
            cr.set_line_width(2)
            cr.arc(self.cursor_point[0], self.cursor_point[1], self.radius, 0, 2*math.pi)
            cr.stroke()

