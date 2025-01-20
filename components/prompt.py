import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from utils.template import TemplateX

@TemplateX("components/prompt.uix")
class Prompt(Gtk.Box):
    __gtype_name__ = "Prompt"
    positive_prompt = Gtk.Template.Child()
    negative_prompt = Gtk.Template.Child()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_prompt.get_buffer().set_text("夜空、クジラ、星, Van Gogh style, oil painting")
        self.negative_prompt.get_buffer().set_text("elan doodle, lowres")
    
    def get_config(self):
        positive_buf = self.positive_prompt.get_buffer()
        negative_buf = self.negative_prompt.get_buffer()
        return {
            "prompt": positive_buf.get_text(positive_buf.get_start_iter(), positive_buf.get_end_iter(), True),
            "negative_prompt": negative_buf.get_text(negative_buf.get_start_iter(), negative_buf.get_end_iter(), True),
        }