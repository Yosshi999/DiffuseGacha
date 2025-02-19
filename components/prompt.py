import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk
import random

from utils.template import TemplateX

@TemplateX("components/prompt.uix")
class Prompt(Gtk.Box):
    __gtype_name__ = "Prompt"
    positive_prompt = Gtk.Template.Child()
    negative_prompt = Gtk.Template.Child()
    seed: Gtk.Entry = Gtk.Template.Child()
    random_generation: Gtk.CheckButton = Gtk.Template.Child()

    @Gtk.Template.Callback()
    def on_rand_toggled(self, button):
        if self.random_generation.get_active():
            self.seed.set_sensitive(False)
            self.seed.get_buffer().delete_text(0, -1)
        else:
            self.seed.set_sensitive(True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_prompt.get_buffer().set_text("夜空、クジラ、星, Van Gogh style, oil painting")
        self.negative_prompt.get_buffer().set_text("elan doodle,lowres,3d,3d cg,vroid,ugly,cropped,jpeg artifacts,blurry")
        self.seed.set_sensitive(False)
    
    def get_config(self):
        positive_buf = self.positive_prompt.get_buffer()
        negative_buf = self.negative_prompt.get_buffer()
        if self.random_generation.get_active():
            self.seed.get_buffer().set_text(str(random.randint(0, 2**31)), -1)
        seed_buf = self.seed.get_buffer()
        seed = seed_buf.get_text()
        try:
            seed = int(seed)
        except ValueError:
            seed = hash(seed)
        return {
            "prompt": positive_buf.get_text(positive_buf.get_start_iter(), positive_buf.get_end_iter(), True),
            "negative_prompt": negative_buf.get_text(negative_buf.get_start_iter(), negative_buf.get_end_iter(), True),
            "seed": seed,
        }