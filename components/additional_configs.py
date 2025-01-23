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
        self.children = [self.t2i, self.i2i]
        self.task_name = ["t2i", "i2i"]

    def get_config(self):
        index = self.get_current_page()
        return self.children[index].get_config()
    
    def get_task_name(self):
        index = self.get_current_page()
        return self.task_name[index]
