import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk
from pathlib import Path

from components.window import Window

class MyApp(Gtk.Application):
    def __init__(self, output_folder, **kwargs):
        super().__init__(**kwargs)
        self.output_folder = output_folder
        self.connect('activate', self.on_activate)
        self.connect('startup', self.on_startup)

    def on_activate(self, app):
        provider = Gtk.CssProvider.new()
        provider.load_from_path("style.css")
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.win = Window(self.output_folder)
        self.win.set_application(self)
        self.win.set_show_menubar(True)
        self.win.present()

    def on_startup(self, app):
        menu = Gtk.Builder.new_from_file("components/menu.ui").get_object("menu")
        self.set_menubar(menu)

output_folder = Path("outputs")
if output_folder.exists():
    assert output_folder.is_dir(), f"{str(output_folder.absolute())} must be folder."
if not output_folder.exists():
    output_folder.mkdir()

app = MyApp(output_folder=output_folder, application_id='org.gtk.Example')
app.run(None)