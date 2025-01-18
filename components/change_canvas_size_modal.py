import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from utils.template import TemplateX

@TemplateX("components/change_canvas_size_modal.uix")
class ChangeCanvasSizeModal(Gtk.Window):
    __gtype_name__ = "ChangeCanvasSizeModal"
    width_entry = Gtk.Template.Child()
    height_entry = Gtk.Template.Child()
    lock_aspect_ratio = Gtk.Template.Child()

    def __init__(self, width, height, callback):
        super().__init__()
        self.width_entry.set_value(width)
        self.height_entry.set_value(height)
        self.width = width
        self.height = height
        self.callback = callback

    @Gtk.Template.Callback("on_cancel")
    def on_cancel(self, widget):
        self.close()
    
    @Gtk.Template.Callback("on_confirm")
    def on_confirm(self, widget):
        self.close()
        self.callback(self.width, self.height)
    
    @Gtk.Template.Callback("on_change")
    def on_change(self, widget):
        width = self.width_entry.get_value_as_int()
        height = self.height_entry.get_value_as_int()
        if self.lock_aspect_ratio.get_active():
            if width != self.width:
                # width changed. adjust height
                height = width / self.width * self.height
            else:
                # height changed. adjust width
                width = height / self.height * self.width
        width = int(width)
        height = int(height)
        width -= width % 8
        height -= height % 8
        width = max(min(width, 1024), 8)
        height = max(min(height, 1024), 8)
        self.width = width
        self.height = height
        self.width_entry.set_value(width)
        self.height_entry.set_value(height)