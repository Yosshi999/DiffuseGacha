import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from .transpile import transpile

class BaseComponent:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def prepare_xml(self) -> str:
        raise NotImplementedError

    def on_render(self) -> None:
        pass

    def render(self) -> Gtk.Widget:
        xml = self.prepare_xml()
        transpiled, handlers = transpile(xml)
        self.builder = Gtk.Builder.new_from_string(transpiled, -1)
        for handler in handlers:
            obj = self.builder.get_object(handler.id)
            obj.connect(handler.signal, getattr(self, handler.handler))
        self.on_render()
        return self.builder.get_object(self.name)

    def get_element_by_id(self, id: str) -> Gtk.Widget:
        return self.builder.get_object(id)
