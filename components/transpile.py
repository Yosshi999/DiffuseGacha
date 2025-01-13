from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET
import re
from uuid import uuid4
from typing import List, Tuple
from collections import namedtuple

Handler = namedtuple("Handler", ["id", "signal", "handler"])

def _transpile(from_element: ET.Element, to_element: ET.Element, handlers: List[Handler]) -> None:
    class_name = "Gtk" + from_element.tag
    new_element = ET.SubElement(to_element, "object", {"class": class_name})
    for key, value in from_element.attrib.items():
        element_id = None
        if key == "id":
            new_element.attrib["id"] = from_element.attrib["id"]
            element_id = from_element.attrib["id"]
        elif key == "className":
            style = ET.SubElement(new_element, "style")
            for name in value.split(" "):
                ET.SubElement(style, "class", {"name": name})
        elif key.startswith("_"):
            # prepare event handler
            event_name = key[1:]
            if element_id is None:
                element_id = str(uuid4())
                new_element.attrib["id"] = element_id
            handlers.append(Handler(element_id, event_name, value))
        else:
            # rename camelCase to kebab-case
            key = re.sub(r'(?<!^)(?=[A-Z])', '-', key).lower()
            ET.SubElement(new_element, "property", {"name": key}).text = value
    for from_child in from_element:
        to_child = ET.SubElement(new_element, "child")
        _transpile(from_child, to_child, handlers)

def transpile(xmlstr: str, pprint: bool = False) -> Tuple[str, List[Handler]]:
    handlers: List[Handler] = []
    root = ET.fromstring(xmlstr)
    result_root = ET.Element("interface")
    ET.SubElement(result_root, "requires", {"lib": "gtk", "version": "4.0"})
    _transpile(root, result_root, handlers)
    if pprint:
        ET.indent(result_root)
    return ET.tostring(result_root, encoding="utf-8", xml_declaration=True).decode(), handlers


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", type=Path, help="Input folder containing XMLs")
    args = parser.parse_args()

    fnames = sorted(args.input.glob("*.xml"))
    print(f"Transpiling {len(fnames)} XML files...")

    for fn in fnames:
        print("Processing", fn.name)
        trans = transpile(fn.read_text())[0]
        fn.with_suffix(".ui").write_text(trans)