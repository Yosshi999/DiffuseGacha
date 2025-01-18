import xml.etree.ElementTree as ET
import re

def _get_class_name(tag: str) -> str:
    if tag.startswith("m"):
        # custom widget
        return tag[1:]
    else:
        return "Gtk" + tag

def _rename(key: str) -> str:
    """Rename camelCase to kebab-case."""
    return re.sub(r'(?<!^)(?=[A-Z])', '-', key).lower()

def _transpile(from_element: ET.Element, to_element: ET.Element) -> None:
    if from_element.tag == "template":
        new_element = ET.SubElement(to_element, "template", {
            "class": from_element.attrib.pop("class"),
            "parent": from_element.attrib.pop("parent"),
        })
    else:
        class_name = _get_class_name(from_element.tag)
        new_element = ET.SubElement(to_element, "object", {"class": class_name})

    layout_element = None
    for key, value in from_element.attrib.items():
        if key == "id":
            new_element.attrib["id"] = from_element.attrib["id"]
        elif key == "className":
            style = ET.SubElement(new_element, "style")
            for name in value.split(" "):
                ET.SubElement(style, "class", {"name": name})
        elif key.startswith("on_"):
            # prepare event handler
            event_name = _rename(key[3:])
            ET.SubElement(new_element, "signal", {"name": event_name, "handler": value})
        elif key.startswith("layout_"):
            if layout_element is None:
                layout_element = ET.SubElement(new_element, "layout")
            layout_name = _rename(key[7:])
            ET.SubElement(layout_element, "property", {"name": layout_name}).text = value
        else:
            key = _rename(key)
            ET.SubElement(new_element, "property", {"name": key}).text = value
    for from_child in from_element:
        to_child = ET.SubElement(new_element, "child")
        _transpile(from_child, to_child)

def transpile(xmlstr: str, pprint: bool = False) -> str:
    root = ET.fromstring(xmlstr)
    assert root.tag == "interface"

    result_root = ET.Element("interface")
    ET.SubElement(result_root, "requires", {"lib": "gtk", "version": "4.0"})

    for definition in root:
        _transpile(definition, result_root)

    if pprint:
        ET.indent(result_root)
    return ET.tostring(result_root, encoding="utf-8", xml_declaration=True).decode()
