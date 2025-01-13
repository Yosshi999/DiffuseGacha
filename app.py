import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk
import threading
import asyncio

from components.main_window import MainWindow

class MyApp(Gtk.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect('activate', self.on_activate)
        self.bg_loop = asyncio.new_event_loop()

    def on_activate(self, app):
        provider = Gtk.CssProvider.new()
        provider.load_from_path("style.css")
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.bg_thread = threading.Thread(target=self.bg_loop.run_forever, daemon=True)
        self.bg_thread.start()
        self.win = MainWindow(self.bg_loop).render()
        self.win.set_application(self)
        self.win.present()

app = MyApp(application_id='org.gtk.Example')
app.run(None)