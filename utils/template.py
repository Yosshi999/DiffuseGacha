from pathlib import Path
from gi.overrides import Gtk
from .transpile import transpile

class TemplateX(Gtk.Template):
    def __init__(self, resource_path: str):
        string = Path(resource_path).read_text()
        super().__init__(string=transpile(string))