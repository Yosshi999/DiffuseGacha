from diffusers import DiffusionPipeline
from PIL import Image
import gi
from gi.repository import GLib, Gtk
import torch
from diffusers.utils.torch_utils import randn_tensor
from dataclasses import dataclass
from typing import Optional, List, Any
import asyncio
from collections.abc import Callable
import threading
from functools import partial

from utils.pipes import CanvasMemory, text_to_image, image_to_image, decode_latent
from utils.imutil import save_image_with_metadata, load_image_with_metadata, mitsua_credit


class ModelNotInitializedError(Exception):
    pass

@dataclass
class GlobalState:
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    device: str
    dtype: torch.dtype
    _pipe: Optional[Any] = None

    @property
    def pipe(self):
        if self._pipe is None:
            raise ModelNotInitializedError()
        return self._pipe
    @pipe.setter
    def pipe(self, p):
        self._pipe = p


_gstate: Optional[GlobalState] = None

def initialize():
    global _gstate
    if torch.cuda.is_available():
        print("CUDA is available")
        device = "cuda"
    else:
        device = "cpu"
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    _gstate = GlobalState(loop=loop, thread=thread, device=device, dtype=torch.float16)

async def _initialize_model():
    pipe = DiffusionPipeline.from_pretrained("Mitsua/mitsua-likes", trust_remote_code=True).to(_gstate.device, dtype=_gstate.dtype)
    # Workaround because https://huggingface.co/Mitsua/mitsua-likes/blob/main/pipeline_likes_base_unet.py#L1035 is broken.
    pipe.run_character_detector = lambda *args: (None, None)
    _gstate.pipe = pipe

def initialize_model_async(resolve: Optional[Callable[[], None]] = None):
    fut = asyncio.run_coroutine_threadsafe(
        _initialize_model(),
        _gstate.loop
    )
    if resolve is not None:
        fut.add_done_callback(lambda _: GLib.idle_add(resolve))

async def _process_pipe(task_name: str, **pipe_kwargs) -> CanvasMemory:
    if task_name == "t2i":
        return text_to_image(_gstate.pipe, **pipe_kwargs)
    elif task_name == "i2i":
        return image_to_image(_gstate.pipe, **pipe_kwargs)
    else:
        raise NotImplementedError()

def process_pipe_async(task_name: str, pipe_kwargs: dict, progress: Callable[[float], None], resolve: Callable[[CanvasMemory], None]):
    def on_step_end(pipe, i, t, kwargs):
        GLib.idle_add(lambda: progress((i + 1) / _gstate.pipe._num_timesteps))
        return {}
    fut = asyncio.run_coroutine_threadsafe(
        _process_pipe(
            task_name,
            callback_on_step_end=on_step_end,
            **pipe_kwargs
        ),
        _gstate.loop
    )
    fut.add_done_callback(lambda m: GLib.idle_add(resolve, m.result()))

def load_memory(path: str) -> CanvasMemory:
    _, latent, config = load_image_with_metadata(path)
    latent = latent.to(_gstate.pipe._execution_device)
    image = decode_latent(_gstate.pipe, latent)[0]
    return CanvasMemory(image, latent, config)

def save_memory(memory: CanvasMemory, path: str):
    image = mitsua_credit(memory.image)
    save_image_with_metadata(image, path, memory.latent, memory.generation_config)

def load_memory_with_dialog(window: Gtk.ApplicationWindow, resolve: Callable[[CanvasMemory], None]):
    def callback(resolve, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            try:
                file = window.open_dialog.get_file()
                if file is not None:
                    print(f"File path is {file.get_path()}")
                    resolve(load_memory(file.get_path()))
            except GLib.Error as error:
                print(error)
                Gtk.AlertDialog(message=f"Error opening file", detail=error.message).show(window)
            except Exception as e:
                print(e)
                Gtk.AlertDialog(message=f"Error opening file", detail="This file is not supported.").show(window)

    window.open_dialog = Gtk.FileChooserNative.new(title="Open File", parent=window, action=Gtk.FileChooserAction.OPEN)
    window.open_dialog.set_modal(True)
    window.open_dialog.set_transient_for(window)
    window.open_dialog.connect("response", partial(callback, resolve))
    window.open_dialog.show()