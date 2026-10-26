"""GPU field camera using AdvantageScope's GLB, with depth-tested AprilTags."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

import numpy as np

from generate import corners, project


def find_field_asset(explicit=None):
    if explicit:
        folder = Path(explicit)
        if folder.is_file():
            folder = folder.parent
        candidates = [folder]
    else:
        roots = [Path(__file__).parent / "generated/assets",
                 Path(os.environ.get("APPDATA", Path.home()/".config")) / "AdvantageScope/autoAssets"]
        candidates = [p for root in roots for p in sorted(root.glob("Field3d_2026FRCField*"), reverse=True)]
    for folder in candidates:
        if (folder/"model.glb").is_file() and (folder/"config.json").is_file():
            return folder.resolve()
    raise FileNotFoundError("2026 field asset not found. Run python chassis_sim/download_field.py "
                            "or supply --field-asset PATH to an AdvantageScope field asset folder.")


def asset_transform(config):
    """AdvantageScope model coordinates -> field coordinates, blue-wall origin."""
    matrix = np.eye(4)
    for rotation in config.get("rotations", []):
        angle = np.radians(rotation["degrees"])
        c, s = np.cos(angle), np.sin(angle)
        axis = rotation["axis"]
        r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
             "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        step = np.eye(4)
        step[:3, :3] = r
        matrix = step @ matrix
    if config.get("coordinateSystem") != "wall-blue":
        raise ValueError("Field camera requires a wall-blue field asset")
    # AdvantageScope's centered CAD scene faces the opposite end from WPILib's
    # blue-wall origin. Its wall-blue display applies a 180-degree field rotation.
    matrix = np.diag([-1., -1., 1., 1.]) @ matrix
    matrix[:3, 3] = np.array(config.get("position", [0, 0, 0])) + [
        config["widthInches"]*.0254/2, config["heightInches"]*.0254/2, 0]
    return matrix


def projection(camera, width, height, near=.02, far=80.):
    # Optical pixel centers are integer coordinates; GL samples at half pixels.
    cx, cy = camera["cx"]+.5, camera["cy"]+.5
    return np.array([[2*camera["fx"]/width, 0, 1-2*cx/width, 0],
                     [0, 2*camera["fy"]/height, 2*cy/height-1, 0],
                     [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                     [0, 0, -1, 0]], dtype=np.float32)


class GPUScene:
    def __init__(self, folder, camera, width, height, tags, textures):
        import moderngl
        import trimesh
        self.gl = moderngl
        self.ctx = moderngl.create_standalone_context(require=330)
        self.width, self.height = width, height
        self.camera, self.tags = camera, tags
        self.projection = projection(camera, width, height)
        self.fbo = self.ctx.framebuffer([self.ctx.texture((width, height), 3)],
                                        self.ctx.depth_renderbuffer((width, height)))
        self.solid = self.ctx.program(vertex_shader='''#version 330
            uniform mat4 mvp;
            in vec3 position; in vec3 normal; in vec4 color;
            out vec3 n; out vec4 c;
            void main() { gl_Position=mvp*vec4(position,1); n=normal; c=color; }
        ''', fragment_shader='''#version 330
            in vec3 n; in vec4 c; out vec4 pixel;
            void main() {
                float light=.65+.35*abs(dot(normalize(n),normalize(vec3(.3,-.4,1))));
                pixel=vec4(c.rgb*light,c.a);
            }
        ''')
        self.tag_program = self.ctx.program(vertex_shader='''#version 330
            uniform mat4 mvp; in vec3 position; in vec2 uv; out vec2 texture_uv;
            void main() { gl_Position=mvp*vec4(position,1); texture_uv=uv; }
        ''', fragment_shader='''#version 330
            uniform sampler2D marker; in vec2 texture_uv; out vec4 pixel;
            void main() { float v=texture(marker,texture_uv).r; pixel=vec4(v,v,v,1); }
        ''')
        self.tag_program['marker'] = 0
        transform = asset_transform(json.loads((Path(folder)/'config.json').read_text(encoding='utf-8')))
        scene = trimesh.load_scene(Path(folder)/'model.glb', process=False)
        batches = [[], []]
        # Bake node transforms, then batch into just two field draw calls.
        for node in scene.graph.nodes_geometry:
            node_transform, geometry = scene.graph[node]
            mesh = scene.geometry[geometry]
            # Replace CAD vinyl patterns with exact native-family textures from the field JSON.
            # Keep backing plates and mounts, so nearby geometry still occludes the markers.
            if 'apriltag vinyl' in str(node).lower() or 'apriltag vinyl' in str(geometry).lower():
                continue
            world = transform @ node_transform
            positions = mesh.vertices @ world[:3, :3].T + world[:3, 3]
            normals = mesh.vertex_normals @ np.linalg.inv(world[:3, :3])
            material = getattr(mesh.visual, 'material', None)
            color = np.asarray(getattr(material, 'baseColorFactor', [180, 180, 180, 255]), dtype=float)/255
            if len(color) != 4:
                color = np.array([.7, .7, .7, 1.])
            vertices = np.column_stack((positions, normals, np.tile(color, (len(positions), 1))))
            batches[int(color[3] < .95)].append(vertices[mesh.faces.reshape(-1)].astype('f4'))
        self.field = [self._solid_vao(np.concatenate(batch)) if batch else None for batch in batches]
        self.tag_meshes = []
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        order = [0, 1, 2, 0, 2, 3]
        for tag in tags:
            normal = np.asarray(tag['rotation'])[:, 0]
            points = corners(tag, camera['tag_size_m']*1.25) + normal*.003
            data = np.column_stack((points, uv))[order].astype('f4')
            vao = self.ctx.vertex_array(self.tag_program, [(self.ctx.buffer(data.tobytes()), '3f 2f', 'position', 'uv')])
            image = textures[tag['id']]
            texture = self.ctx.texture((image.shape[1], image.shape[0]), 1, image.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
            self.tag_meshes.append((np.asarray(tag['position']), normal, vao, texture))
        self.renderer = self.ctx.info['GL_RENDERER']
        self.terrain_data = None
        self.terrain_vao = None

    def _solid_vao(self, vertices):
        buffer = self.ctx.buffer(np.asarray(vertices, dtype='f4').tobytes())
        return self.ctx.vertex_array(self.solid, [(buffer, '3f 3f 4f', 'position', 'normal', 'color')])

    def render(self, position, rotation, draw_field=True, blockers=(), with_visibility=False, terrain_vertices=None):
        self.fbo.use()
        self.fbo.clear(.55, .65, .75, 1, depth=1)
        self.ctx.enable(self.gl.DEPTH_TEST)
        self.ctx.disable(self.gl.CULL_FACE | self.gl.BLEND)
        self.ctx.depth_func = '<='
        self.fbo.depth_mask = True
        # Camera optical axes right/down/forward -> OpenGL right/up/back.
        view = np.eye(4)
        view[:3, :3] = np.diag([1, -1, -1]) @ np.asarray(rotation).T
        view[:3, 3] = -view[:3, :3] @ np.asarray(position)
        mvp = (self.projection @ view).astype('f4').T.tobytes()
        self.solid['mvp'].write(mvp)
        self.tag_program['mvp'].write(mvp)
        if draw_field and self.field[0]:
            self.field[0].render()
        data = None if terrain_vertices is None else np.asarray(terrain_vertices,dtype='f4').tobytes()
        if data != self.terrain_data:
            if self.terrain_vao is not None:
                self.terrain_vao.release()
            self.terrain_vao = None if data is None else self._solid_vao(terrain_vertices)
            self.terrain_data = data
        if self.terrain_vao is not None:
            self.terrain_vao.render()
        temporary = []
        if blockers:
            import trimesh
            for center, size in blockers:
                box = trimesh.creation.box(extents=size)
                data = np.column_stack((box.vertices + center, box.vertex_normals,
                                         np.tile([.2, .2, .2, 1.], (len(box.vertices), 1))))
                vao = self._solid_vao(data[box.faces.reshape(-1)])
                vao.render()
                temporary.append(vao)
        for center, normal, vao, texture in self.tag_meshes:
            if np.dot(np.asarray(position)-center, normal) > 0:
                texture.use(0)
                vao.render()
        if draw_field and self.field[1]:
            self.ctx.enable(self.gl.BLEND)
            self.ctx.blend_func = self.gl.SRC_ALPHA, self.gl.ONE_MINUS_SRC_ALPHA
            self.fbo.depth_mask = False
            self.field[1].render()
            self.fbo.depth_mask = True
        rgb = np.frombuffer(self.fbo.read(components=3, alignment=1), np.uint8).reshape(self.height, self.width, 3)[::-1]
        for vao in temporary:
            vao.release()
        image = rgb[:, :, ::-1].copy()  # BGR for OpenCV
        if not with_visibility:
            return image
        depth = np.frombuffer(self.fbo.read(attachment=-1, components=1, dtype='f4'),
                              np.float32).reshape(self.height, self.width)[::-1]
        return image, self.visibility(position, rotation, depth)

    def visibility(self, position, rotation, depth):
        """Conservative, depth-tested labels independent of Vortex detections."""
        import cv2
        labels = []
        position, rotation = np.asarray(position), np.asarray(rotation)
        for tag in self.tags:
            normal = np.asarray(tag['rotation'])[:, 0]
            if np.dot(position-tag['position'], normal) <= 0:
                continue
            points = corners(tag, self.camera['tag_size_m']) + normal*.003
            uv, z = project(points, position, rotation, self.camera)
            if np.any(z <= .02) or not np.all(np.isfinite(uv)):
                continue
            area = abs(cv2.contourArea(uv))
            if area < 4:
                continue
            # Probe the entire black-border square, including edges. A missing
            # pose is only a failure for a fully in-frame, sufficiently large tag.
            grid = np.linspace(0, 1, 7)
            samples = np.array([points[0]+u*(points[1]-points[0])+v*(points[3]-points[0])
                                for u in grid for v in grid])
            pixel, _ = project(samples, position, rotation, self.camera)
            x, y = np.rint(pixel).astype(int).T
            inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            rays = np.column_stack(((x-self.camera['cx'])/self.camera['fx'],
                                     (y-self.camera['cy'])/self.camera['fy'], np.ones(len(x))))
            camera_normal = rotation.T @ normal
            plane_point = rotation.T @ (points[0]-position)
            denominator = rays @ camera_normal
            distance = np.divide(np.dot(plane_point, camera_normal), denominator,
                                 out=np.zeros(len(x)), where=np.abs(denominator) > 1e-9)
            expected_depth = np.divide(80*.02/(80-.02), distance,
                                       out=np.full(len(x), np.inf), where=distance > 0)
            expected_depth = 80/(80-.02)-expected_depth
            visible = np.zeros(len(x), bool)
            visible[inside] = depth[y[inside], x[inside]] >= expected_depth[inside]-2e-6
            full = bool(np.all(uv[:, 0] > 8) and np.all(uv[:, 0] < self.width-8)
                        and np.all(uv[:, 1] > 8) and np.all(uv[:, 1] < self.height-8))
            fraction = float(visible.mean())
            labels.append(dict(id=tag['id'], visible_fraction=fraction, area_px=area,
                               fully_in_frame=full, clearly_visible=full and area >= 225 and bool(visible.all())))
        return labels


def _worker(connection, settings):
    scene = None
    try:
        scene = GPUScene(**settings)
        connection.send(('ready', scene.renderer))
        while True:
            request = connection.recv()
            if request is None:
                break
            image = scene.render(*request)
            connection.send(('frame', image))
    except EOFError:
        pass
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        if scene is not None:
            scene.ctx.release()
        connection.close()


class FieldRenderer:
    """Own the OpenGL context in one persistent process, independent of Tk."""
    def __init__(self, folder, camera, width, height, tags, textures):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        settings = dict(folder=str(folder), camera=camera, width=width, height=height, tags=tags, textures=textures)
        self.process = context.Process(target=_worker, args=(child, settings), daemon=True)
        self.process.start()
        child.close()
        try:
            kind, self.renderer = self._receive(90)
            if kind != 'ready':
                raise RuntimeError(self.renderer)
        except Exception:
            self.close()
            raise

    def _receive(self, timeout=30):
        if not self.connection.poll(timeout):
            raise TimeoutError('Field renderer timed out')
        return self.connection.recv()

    def render(self, position, rotation, draw_field=True, blockers=(), with_visibility=False, terrain_vertices=None):
        self.connection.send((position, rotation, draw_field, blockers, with_visibility, terrain_vertices))
        kind, value = self._receive()
        if kind != 'frame':
            raise RuntimeError(value)
        return value

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (OSError, EOFError):
                pass
            self.process.join(3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(3)
        self.connection.close()
