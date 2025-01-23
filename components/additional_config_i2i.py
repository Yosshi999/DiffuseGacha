import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from utils.template import TemplateX

@TemplateX("components/additional_config_i2i.uix")
class AdditionalConfigI2I(Gtk.Box):
    __gtype_name__ = "AdditionalConfigI2I"
    guidance_scale = Gtk.Template.Child()
    guidance_rescale = Gtk.Template.Child()
    num_inference_steps = Gtk.Template.Child()
    denoising_strength = Gtk.Template.Child()

    def get_config(self):
        return {
            "guidance_scale": self.guidance_scale.get_value(),
            "guidance_rescale": self.guidance_rescale.get_value(),
            "num_inference_steps": self.num_inference_steps.get_value_as_int(),
            "strength": self.denoising_strength.get_value(),
        }