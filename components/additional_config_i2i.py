import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from utils.template import TemplateX

@TemplateX("components/additional_config_i2i.uix")
class AdditionalConfigI2I(Gtk.Box):
    __gtype_name__ = "AdditionalConfigI2I"
    denoising_strength = Gtk.Template.Child()

    def get_config(self):
        return {
            "strength": self.denoising_strength.get_value(),
        }