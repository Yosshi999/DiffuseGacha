from diffusers import DiffusionPipeline
import torch
from utils import mitsua_credit
import asyncio
from PIL import Image
from .base_component import BaseComponent

import gi
gi.require_version("GdkPixbuf", "2.0")

from gi.repository import GdkPixbuf, GLib

class MainWindow(BaseComponent):
    def __init__(self, bg_loop):
        if torch.cuda.is_available():
            # You may need pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
            print("CUDA is available")
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = torch.float16
        self.pipe = DiffusionPipeline.from_pretrained("Mitsua/mitsua-likes", trust_remote_code=True).to(self.device, dtype=self.dtype)
        self.bg_loop = bg_loop
    
    @property
    def name(self) -> str:
        return "main_window"
    
    def on_render(self):
        self.get_element_by_id("positive_prompt").get_buffer().set_text("夜空、クジラ、星, Van Gogh style, oil painting")
        self.get_element_by_id("negative_prompt").get_buffer().set_text("elan doodle, lowres")
        # pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale("output.png", width=640, height=640, preserve_aspect_ratio=False)
        # self.get_element_by_id("gacha_result").set_from_pixbuf(pixbuf)

    async def process_pipe(self, **pipe_kwargs) -> Image.Image:
        ret = self.pipe(**pipe_kwargs)
        image = ret.images[0]
        image = mitsua_credit(image)
        image.save("output.png")
        return image
    
    def on_generate_button_clicked(self, button) -> None:
        self.get_element_by_id("progress").set_fraction(0)
        self.get_element_by_id("gacha_result").clear()
        positive_buf = self.get_element_by_id("positive_prompt").get_buffer()
        positive = positive_buf.get_text(positive_buf.get_start_iter(), positive_buf.get_end_iter(), True)
        negative_buf = self.get_element_by_id("negative_prompt").get_buffer()
        negative = negative_buf.get_text(negative_buf.get_start_iter(), negative_buf.get_end_iter(), True)

        def on_step_end(pipe, i, t, kwargs):
            fraction = (i+1) / pipe._num_timesteps
            GLib.idle_add(lambda: self.get_element_by_id("progress").set_fraction(fraction))
            return {}

        fut = asyncio.run_coroutine_threadsafe(
            self.process_pipe(
                prompt=positive,
                negative_prompt=negative,
                guidance_scale=5.0,
                guidance_rescale=0.7,
                width=640,
                height=640,
                num_inference_steps=40,
                callback_on_step_end=on_step_end),
            self.bg_loop)
        def resolve():
            image = fut.result()
            data = image.tobytes()
            w, h = image.size
            data = GLib.Bytes.new(data)
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)
            self.get_element_by_id("gacha_result").set_from_pixbuf(pixbuf.copy())
        fut.add_done_callback(lambda _: GLib.idle_add(resolve))

    def prepare_xml(self) -> str:
        return """
<ApplicationWindow
    id="main_window"
    defaultHeight="400"
    defaultWidth="600"
    title="DiffuseGacha"
>
    <Box orientation="vertical">
        <Box orientation="horizontal">
            <Box orientation="vertical">
                <Box orientation="horizontal">
                    <ScrolledWindow hexpand="true" className="m-2">
                        <TextView
                            id="positive_prompt"
                            hexpand="true"
                            heightRequest="50"
                            wrapMode="char" />
                    </ScrolledWindow>
                    <Button
                        id="generate_button"
                        label="Generate"
                        className="suggested-action"
                        _clicked="on_generate_button_clicked"
                    />
                </Box>
                <Expander label="Negative Prompt" className="m-2">
                    <ScrolledWindow hexpand="true">
                        <TextView
                            id="negative_prompt"
                            hexpand="true"
                            heightRequest="50"
                            wrapMode="char" />
                    </ScrolledWindow>
                </Expander>
            </Box>
        </Box>
        <ProgressBar id="progress" hexpand="true" />
        <Box orientation="vertical">
            <Image id="gacha_result" pixelSize="640" />
        </Box>
    </Box>
</ApplicationWindow>
"""