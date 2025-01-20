import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk
from pathlib import Path

@Gtk.Template(string=Path("components/additional_configs.ui").read_text())
class AdditionalConfigs(Gtk.Notebook):
    __gtype_name__ = "AdditionalConfigs"
    t2i = Gtk.Template.Child()
    i2i = Gtk.Template.Child()

    def __init__(self):
        super().__init__()
        self.i2i.set_sensitive(False)

    def get_config(self):
        return self.t2i.get_config()