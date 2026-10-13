"""Run with Blender's bundled Python; consumes generate.py --backend blender output."""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = args.dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["backend"] != "blender":
        raise ValueError("Prepare this dataset with generate.py --backend blender")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.cycles.seed = manifest["seed"]
    scene.render.resolution_x = manifest["width"]
    scene.render.resolution_y = manifest["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (0.4, 0.4, 0.4)
    camera = manifest["camera"]
    data = bpy.data.cameras.new("VortexCamera")
    obj = bpy.data.objects.new("VortexCamera", data)
    scene.collection.objects.link(obj)
    scene.camera = obj
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.lens = camera["fx"] * 36 / manifest["width"]
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = camera["fx"] / camera["fy"]
    data.shift_x = (manifest["width"] / 2 - camera["cx"]) / manifest["width"]
    data.shift_y = (camera["cy"] - manifest["height"] / 2) * (camera["fx"] / camera["fy"]) / manifest["width"]
    data.clip_start = 0.01
    data.clip_end = 100
    planes = []
    for tag in manifest["tags"]:
        size = camera["tag_size_m"] * 1.25
        # Local plane x is map y, plane y is map z, outward normal is map x.
        r = Matrix(tag["rotation"]) @ Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
        mesh = bpy.data.meshes.new(f"tag_{tag['id']}")
        s = size / 2
        mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
        plane = bpy.data.objects.new(mesh.name, mesh)
        scene.collection.objects.link(plane)
        plane.rotation_euler = r.to_euler()
        plane.location = tag["position"]
        uv = mesh.uv_layers.new()
        for item, coords in zip(uv.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            item.uv = coords
        mat = bpy.data.materials.new(mesh.name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(root / "textures" / f"{tag['id']}.png"))
        texture.interpolation = "Closest"
        mat.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
        mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
        mesh.materials.append(mat)
        planes.append(plane)
    # Optical camera coordinates (right, down, forward) -> Blender (right, up, back).
    optical_from_blender = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    for frame in manifest["frames"]:
        obj.location = frame["camera_position"]
        obj.rotation_euler = (Matrix(frame["camera_rotation"]) @ optical_from_blender).to_euler()
        for plane, tag in zip(planes, manifest["tags"]):
            normal = Matrix(tag["rotation"]).col[0]
            eye_delta = obj.location - plane.location
            plane.hide_render = frame["case"].get("hide_tags", False) or eye_delta.dot(normal) <= 0
        scene.render.filepath = str(root / frame["image"])
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(root / "scene.blend"))
    print("Rendered dataset. Optional noise/blur: python simulation_tests/postprocess.py --dataset", root)


if __name__ == "__main__":
    main()
