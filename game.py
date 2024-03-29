import pyray as rl
from glm import *
import json
from types import SimpleNamespace
from dataclasses import dataclass
from collections import defaultdict, deque
import util
import math
import time
import random
import itertools
import os
import re
from contextlib import contextmanager
import perlin_noise
import textwrap
#from pyinstrument import Profiler

PLAYER_SIZE = 16
TILE_SIZE = 16
ENRAGE_RANGE=32*2
SUCK_RANGE=100
assert SUCK_RANGE > ENRAGE_RANGE

class SpatialHash:
    cell_size = 64
    recs: list[rl.Rectangle]
    def __init__(self, recs):
        self.recs = recs
        self.spatial = defaultdict(list)

        for i, r in enumerate(self.recs):
            self.spatial[(r.x // self.cell_size, r.y // self.cell_size)].append(i)

    def near(self, point):
        h = (point[0] // self.cell_size, point[1] // self.cell_size)
        r = []
        for d in [(-1, -1), (0, -1), (1, -1),
                  (-1, 0), (0, 0), (1, 0),
                  (-1, 1), (0, 1), (1, 1)]:
            r.extend([self.recs[i] for i in self.spatial[(h[0] + d[0], h[1] + d[1])]])
        return r

sorted_texs = []
def add_texture_sorted(tex, src_rect, dest_rec):
    sorted_texs.append((tex, src_rect, dest_rec))

@dataclass
class Turnstile:
    rec: rl.Rectangle
    locked: bool = True
    unlock_time = 0

def tex_rect(tex, frames=1, frame_i=0, flipv=False):
    w = tex.width / frames
    return (w*frame_i, 0, w, tex.height * (-1 if flipv else 1))

def rect_to_screen_space(rect):
    p = rl.get_world_to_screen_2d((rect.x, rect.y), camera)
    return (p.x, p.y, rect.width, rect.height)


def pad_rect(rect, padding):
    return rl.Rectangle(rect.x + padding, rect.y + padding, rect.width - padding * 2, rect.height - padding * 2)


class Map:
    def __init__(self, map_json):
        def get_layer(name):
            r, = [x for x in map_json["layers"] if x["name"] == name]
            return r

        self.name = map_json["values"]["StationName"]
        self.entities = get_layer("Entities")["entities"]
        self.decals = get_layer("Decals")["decals"]
        self.hints = []
        self.messages_func = None
        for decal in self.decals:
            decal['texture'] = getattr(textures, decal["texture"].removesuffix(".png"))
            w, h = decal['texture'].width, decal['texture'].height
            decal['x'] = decal['x'] - decal['originX'] * w
            decal['y'] = decal['y'] - decal['originY'] * h
            if decal['values']['hint']:
                self.hints.append((rl.Rectangle(decal['x'], decal['y'], w, h), decal['values']['hint']))
        self.hero_spawn, = self.get_entity_pos("hero")
        self.wander_points = self.get_entity_pos("spawnpoint")
        self.exit = [rl.Rectangle(z["x"], z["y"], z["width"], z["height"]) for z in self.entities if z["name"] == "safezone"]
        ghostvac = [e for e in self.entities if e["name"] == "ghostvac"][0]
        self.ghostvac = vec2(ghostvac["x"], ghostvac["y"])
        self.ghostvac_name = None
        self.ghostvac_cable = [rl.Vector2(*(self.ghostvac + vec2(16, 16)))] + [rl.Vector2(n["x"], n["y"]) for n in ghostvac["nodes"]]
        self.turnstiles = [Turnstile(rl.Rectangle(z.x, z.y, TILE_SIZE, TILE_SIZE)) for z in self.get_entity_pos("turnstile")]
        self.walls = get_layer("WallTiles")["data2D"]
        self.wall_recs = SpatialHash([rl.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                                      for y, row in enumerate(self.walls)
                                      for x, t in enumerate(row) if t != -1] +
                                     [rl.Rectangle(self.ghostvac.x, self.ghostvac.y + TILE_SIZE - 2, TILE_SIZE * 2, TILE_SIZE + 2)])
        self.bg = get_layer("BGTiles")["data2D"]
        self.strict_wander = map_json["values"]["StrictWander"]

    def get_entity_pos(self, name):
        return [vec2(e["x"], e["y"]) for e in self.entities if e["name"] == name]

    def get_next_wander_point(self, ghost_pos):
        if self.strict_wander:
            return random.choice(self.wander_points)
        else:
            return random.choice([random.choice(self.wander_points)] +
                          [ghost_pos + vec2(random.uniform(-40, 40), random.uniform(-40, 40))] +
                          ([state.player] if length(state.player - ghost_pos) < 500 else []))


    def draw(self):
        tile_coord = (state.player // TILE_SIZE).to_tuple()
        around = 9
        ys = range(int(max(0, tile_coord[1]-around)), int(min(len(self.bg), tile_coord[1]+around)))
        xs = range(int(max(0, tile_coord[0]-around)), int(min(len(self.bg[0]), tile_coord[0]+around)))
        for y in ys:
            for x in xs:
                bgtile = self.bg[y][x]
                walltile = self.walls[y][x]
                if walltile != -1:
                    rl.draw_texture_pro(textures.tiles,
                                        (walltile * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE),
                                        (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                                        (0, 0),
                                        0,
                                        rl.WHITE)
                elif bgtile != -1:
                    rl.draw_texture_pro(textures.tiles,
                                        (bgtile * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE),
                                        (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                                        (0, 0),
                                        0,
                                        rl.WHITE)


        for decal in self.decals:
            rl.draw_texture(decal["texture"], int(decal["x"]), int(decal["y"]), rl.WHITE)

        for t in self.turnstiles:
            if t.locked:
                rl.draw_texture_pro(
                    textures.turnstile,
                    tex_rect(textures.turnstile, frames=2, frame_i=0),
                    t.rec,
                    (0, 0),
                    0,
                    rl.WHITE)
            else:
                rl.draw_texture_pro(
                    textures.turnstile,
                    tex_rect(textures.turnstile, frames=2, frame_i=1),
                    t.rec,
                    (0, 0),
                    0,
                    rl.WHITE)
                t.unlock_time -= rl.get_frame_time()
                if t.unlock_time < 0:
                    t.locked = True

        for i in range(len(self.ghostvac_cable)-1):
            rl.draw_line_bezier(self.ghostvac_cable[i], self.ghostvac_cable[i+1], 2, rl.DARKPURPLE)

        frame_width = textures.ghostvac.width / 2
        if self.ghostvac_name is not None:
            frame_i = int((rl.get_time() % 1) / 0.5)
            if state.ghost_state == 'sucking':
                frame_i = int((rl.get_time() % 0.2) / 0.1)
        elif state.ghost_state == 'sucked':
            frame_i = 1
        else:
            frame_i = 0

        add_texture_sorted(textures.ghostvac,
                           rl.Rectangle(frame_i * frame_width, 0, frame_width, textures.ghostvac.height),
                           rl.Rectangle(self.ghostvac.x,
                                        self.ghostvac.y,
                                        frame_width,
                                        textures.ghostvac.height))


    def resolve_collisions(self, entity_pos, entity_size=(TILE_SIZE, TILE_SIZE)):
        entity_rec = rl.Rectangle(entity_pos[0], entity_pos[1], entity_size[0], entity_size[1])

        if (p := util.resolve_map_collision(self.wall_recs.near(entity_pos) + [t.rec for t in self.turnstiles if t.locked], entity_rec)) is not None:
            entity_pos[0], entity_pos[1] = p[0], p[1]





SCALE = 4
WIDTH, HEIGHT = 300 * SCALE, 200 * SCALE
rl.init_window(WIDTH, HEIGHT, "Aberration Station")
rl.init_audio_device()

font = rl.load_font_ex("resources/mago3.ttf", 24, None, 0)
rl.gui_set_font(font)
rl.gui_set_style(rl.DEFAULT, rl.TEXT_SIZE, 24)

textures = SimpleNamespace()
for f in os.listdir('resources'):
    if f.endswith('.png'):
        setattr(textures, f.removesuffix('.png'), rl.load_texture('resources/' + f))

# setup repeat sounds
sounds = SimpleNamespace()
sound_files = {x for x in os.listdir('resources') if x.endswith('.wav')}
repeats = {(g.group(1), x) for x in sound_files if (g := re.search(r'(\w+)_(\d+)\.wav$', x))}
for k, g in itertools.groupby(sorted(repeats), key=lambda x: x[0]):
    setattr(sounds, k, [rl.load_sound('resources/' + x[1]) for x in g])
sound_files -= repeats
for f in sound_files:
    setattr(sounds, f.removesuffix('.wav'), rl.load_sound('resources/' + f))

rl.set_sound_volume(sounds.notify, 0.3)

bg_soundscape = rl.load_music_stream("resources/bg.ogg")
scared_loop = rl.load_music_stream("resources/scared_loop.ogg")
victory_music = rl.load_music_stream("resources/victory.ogg")
victory_music.looping = False
victory_music_final = rl.load_music_stream("resources/victory_final.ogg")
victory_music_final.looping = True

GHOST_TRAIL_TTL = 2

state = SimpleNamespace(phone=None,
                        run_time=0)

camera = rl.Camera2D(
    (WIDTH // SCALE // 2, HEIGHT // SCALE // 2),
    (0, 0),
    0,
    1)

@dataclass
class Pulse:
    pos: vec2
    size: float
    ttl: float = 5
    active: bool = True


class Spring:
    """Damped spring. Based on https://www.youtube.com/watch?v=KPoeNZZ6H4s"""

    def __init__(self, f, z, r, x0):
        self.k1 = z / (math.pi * f)
        self.k2 = 1 / ((2 * math.pi * f) ** 2)
        self.k3 = r * z / (2 * math.pi * f)
        self.xp = x0
        self.y = x0
        self.yd = type(x0)(0)

    def update(self, x, xd=None):
        dt = rl.get_frame_time()
        if dt == 0:
            return x

        if xd is None:
            xd = (x - self.xp) / dt
            self.xp = x
        # This breaks the first frame, so I'm just using k2 and hoping for
        # stability since I don't have any high frequency stuff ??
        #k2_stable = max(self.k2, 1.1 * ((dt**2) / 4 + dt * self.k1 / 2))
        self.y += dt * self.yd
        self.yd += (
            dt * (x + self.k3 * xd - self.y - self.k1 * self.yd) / self.k2
        )
        return self.y

class StatusMessage:
    def __init__(self, text):
        self.text = text
        self.height = 30

    def draw(self, pos):
        t = Text(self.text, pos, origin=(0, 0), color=rl.WHITE)
        r = pad_rect(t.rect(), -5)
        rl.draw_rectangle_rounded(r, 0.1, 5, rl.LIGHTGRAY)
        t.draw()

class TextMessage:
    def __init__(self, text):
        self.text = text
        self.gui_text = Text(self.text, (0, 0), origin=(0, 0), color=rl.DARKGRAY)

    @property
    def height(self):
        m = self.gui_text.measure().y
        return m + 20

    def draw(self, pos):
        self.gui_text.anchor_point = pos
        r = pad_rect(self.gui_text.rect(), -5)
        rl.draw_rectangle_rounded(r, 0.1, 5, rl.WHITE)
        rl.draw_triangle(
            (r.x, r.y + r.height + 4),
            (r.x + 14 + 10, r.y + r.height - 10),
            (r.x + 5 + 10, r.y + r.height - 10 - 10),
            rl.WHITE)
        self.gui_text.draw()

class Phone:
    def __init__(self, messages_func):
        self._is_showing = False
        self.last_read = 0
        self.last_read_ttl = 2
        self.hide()
        self.pos_spring = Spring(2, 0.5, 0, self.goal_pos)
        self._popup_state = 'hidden'
        self.scan_results = ""
        self.scan_coro = None
        self.unit_system = 'imperial'
        self.page = 0
        self.pages = 4
        self.last_pic = None
        self.pic_contents = set()
        self.messages = []
        self.discovered = set()
        self.messages_coro = messages_func(self) if messages_func else None
        self.puzzle_state = 'hidden'
        self.clue_textures = [textures.a4, textures.a3, textures.a2, textures.a1]

    @property
    def is_messaging_active(self):
        return self._is_showing and self.page == 0

    def add_message(self, msg):
        self.messages.append(msg)
        if self.is_messaging_active:
            self.last_read = len(self.messages)
            self.last_read_ttl = 2

    def show(self):
        self._is_showing = True
        self.goal_pos = vec2(WIDTH // 2, HEIGHT // 2)

    @property
    def is_camera_active(self):
        return self._is_showing and self.page == 2

    @property
    def screen_rect(self):
        return pad_rect(self.rect, 10)

    @property
    def screen_space_rect(self):
        return rl.Rectangle(self.screen_rect.x / SCALE,
                            self.screen_rect.y / SCALE,
                            self.screen_rect.width / SCALE,
                            self.screen_rect.height / SCALE)

    def hide(self):
        self._is_showing = False
        self.goal_pos = vec2(WIDTH + 300, HEIGHT + 500)

    def in_viewfinder(self, rect):
        p = rl.get_world_to_screen_2d(rl.Vector2(rect.x, rect.y), camera)
        r = rl.Rectangle(p.x, p.y, rect.width, rect.height)
        return self.is_camera_active and rl.check_collision_recs(self.screen_space_rect, r)

    def update(self, dt, shake):
        self.pos_spring.update(self.goal_pos + shake)
        if self.scan_coro is not None:
            try:
                next(self.scan_coro)
            except StopIteration:
                self.scan_coro = None
        if self._popup_state == 'idle': # cleanup popup
            self.hide()
            self._popup_state = 'hidden'
        if self._popup_state == 'updated':
            self._popup_state = 'idle'

        # update messages
        if self.messages_coro:
            try:
                next(self.messages_coro)
            except StopIteration:
                self.messages_coro = None

    def pixels_to_dist(self, pixels):
        meters = (pixels / 32)
        if self.unit_system == 'metric':
            return f'{meters:0.1f}m'
        if self.unit_system == 'imperial':
            return f'{meters * 3.2:0.1f}ft'

    def scan(self):
        def f():
            for i in range(3):
                for x in ['', '.', '..', '...']:
                    self.scan_results = f"scanning{x}"
                    yield from wait_time(0.1)
            diff = state.ghost_pos - state.player
            plen = length(diff)
            ghostdir = ""
            if abs(diff.y) > abs(diff.x):
                if diff.y < 0:
                    ghostdir = "NORTH"
                else:
                    ghostdir = "SOUTH"
            else:
                if diff.x < 0:
                    ghostdir = "WEST"
                else:
                    ghostdir = "EAST"

            self.scan_results = f"Aberration detected to the:\n{ghostdir}"
            #plen = length(state.player - state.ghost_pos)
            #self.scan_results = f"Aberration detected: {self.pixels_to_dist(plen)}"
            if plen < ENRAGE_RANGE:
                self.scan_results += "\nIN RANGE"
        self.scan_coro = f()

    @property
    def pos(self):
        return self.pos_spring.y

    @property
    def rect(self):
        width, height = 350, 600
        return rl.Rectangle(self.pos.x - width / 2, self.pos.y - height / 2, width, height)

    @contextmanager
    def popup(self, title, rect, padding=20):
        if self._popup_state == 'hidden':
            # init
            self.show()
        self._popup_state = 'updated'
        rl.draw_rectangle_rec(rect, rl.GRAY)
        draw_text(title, (rect.x + rect.width // 2, rect.y + 20))
        rect.y += 30 + padding
        rect.height -= 2 * padding
        rect.x += padding
        rect.width -= 2 * padding
        yield rect


#pulses = []
#ghost_pulses = []

def mclamp(v, len):
    if (l := length(v)) > len:
        return v / l
    return v

flashlight_shader = rl.load_shader_from_memory(
    None,
    """\
#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;
uniform vec4 colDiffuse;

// Output fragment color
out vec4 finalColor;

// NOTE: Add here your custom variables

// NOTE: Render size values should be passed from code
const float renderWidth = {WIDTH};
const float renderHeight = {HEIGHT};

float radius = 400.0;
float angle = 0.8;

uniform vec2 pos = vec2(200.0, 200.0);

void main()
{
    vec2 texSize = vec2(renderWidth, renderHeight);

    float dist = length(fragTexCoord*texSize - pos);

    if (dist < radius)
    {
        vec4 color = texture2D(texture0, fragTexCoord)*colDiffuse*fragColor;
        finalColor = vec4(color.rgb, 1 - dist/radius);
    } else {
        finalColor = vec4(0, 0, 0, 0);
    }
}
    """.replace("{WIDTH}", str(WIDTH)).replace("{HEIGHT}", str(HEIGHT))
    )

chromatic_aberration = rl.load_shader_from_memory(
    None,
    """\
#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;

uniform vec2 rOffset;
uniform vec2 gOffset;
uniform vec2 bOffset;

// Output fragment color
out vec4 finalColor;

float vignette(vec2 uv){
    uv *= 1.0 - uv.xy;
    float vignette = uv.x * uv.y * 15.0;
    return pow(vignette, 0.3 + rOffset.x * 100);
}

void main()
{
    vec4 rValue = texture2D(texture0, fragTexCoord - rOffset);
    vec4 gValue = texture2D(texture0, fragTexCoord - gOffset);
    vec4 bValue = texture2D(texture0, fragTexCoord - bOffset);

    finalColor = vec4(rValue.r, gValue.g, bValue.b, 1.0) * vignette(fragTexCoord);
}
    """.replace("{WIDTH}", str(WIDTH)).replace("{HEIGHT}", str(HEIGHT))
    )

def tiles_around(tile_coord, around = 10):
    ys = range(int(max(0, tile_coord[1]-around)), int(min(len(tiles_2d), tile_coord[1]+around)))
    xs = range(int(max(0, tile_coord[0]-around)), int(min(len(tiles_2d[0]), tile_coord[0]+around)))
    for y in ys:
        for x in xs:
            yield x, y, tiles_2d[y][x]

def player_rec():
    return rl.Rectangle(state.player.x, state.player.y, PLAYER_SIZE, PLAYER_SIZE)

def player_origin():
    return state.player + (PLAYER_SIZE // 2, PLAYER_SIZE // 2)

def player_irec():
    return rl.Rectangle(int(state.player.x), int(state.player.y), PLAYER_SIZE, PLAYER_SIZE)

@dataclass
class Text:
    text: str
    anchor_point: vec2
    origin: tuple[float, float] = (0.5, 0.5)
    size: int = 24
    color: rl.Color = rl.WHITE
    spacing: float = 1.3

    def measure(self):
        return rl.measure_text_ex(font, self.text, self.size, self.spacing)

    def pos(self):
        v = self.measure()
        try:
            return self.anchor_point[0] - v.x * self.origin[0], self.anchor_point[1] - v.y * self.origin[1]
        except:
            breakpoint()

    def rect(self):
        v = self.measure()
        return rl.Rectangle(*self.pos(), v.x, v.y)

    def draw(self):
        rl.draw_text_ex(font, self.text, self.pos(), self.size, self.spacing, self.color)

def draw_text(text, pos, origin=(0.5, 0.5), size=24, color=rl.WHITE):
    t = Text(text, pos, origin, size, color)
    t.draw()
    return t

def draw_todo_item(text, pos, is_done, origin=(0.5, 0.5)):
    t = draw_text(text, pos, origin, color=rl.BLACK)
    if is_done:
        rl.draw_line_ex((pos[0], pos[1]), (vec2(t.measure().x + 20, 0) + pos).to_tuple(), 3, rl.BLACK)

def dead_loop():
    rl.stop_sound(sounds.howl)
    text = ""
    img = textures.dead
    def coro():
        nonlocal text, img
        global current_func
        rl.play_sound(sounds.thump)
        yield from wait_time(2)
        rl.play_sound(sounds.death_bell)
        for _ in wait_time(4):
            draw_text("You passed out!", (WIDTH // 2, HEIGHT // 2 - 80), size=48)
            scale = 6
            rl.draw_texture_pro(textures.dead,
                                rl.Rectangle(0, 0, textures.dead.width, textures.dead.height),
                                rl.Rectangle(WIDTH // 2 - (textures.dead.width * scale) // 2,
                                             HEIGHT // 2 - (textures.dead.height * scale) // 2 + 40,
                                             textures.dead.width * scale,
                                             textures.dead.height * scale), rl.Vector2(), 0, rl.WHITE)
            yield
        for _ in wait_time(4):
            draw_text(f"This is {maps[0].name}", (WIDTH // 2, HEIGHT // 2), size=48)
            yield
        current_func = game_loop

    c = coro()
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        try:
            next(c)
        except StopIteration:
            return
        rl.end_drawing()


def victory_loop():
    maps.pop(0)
    state.phone = None # reset phone for next level
    rl.stop_sound(sounds.howl)
    rl.play_music_stream(victory_music)
    text = ""
    def coro():
        nonlocal text
        global current_func
        text = "You caught the aberration!"
        yield from wait_time(4)
        if maps:
            text = "Next station:"
            yield from wait_time(4, allow_skip=True)
            text = f"This is {maps[0].name}"
            yield from wait_time(4, allow_skip=True)
            current_func = game_loop
        else:
            minutes = state.run_time // 60
            seconds = state.run_time % 60
            text = f"You win! Your time: {minutes:02.0f}:{seconds:06.03f}"
            victory_dance_frames = 8
            width = int(textures.victory_dance.width / victory_dance_frames)
            height = textures.victory_dance.height

            rl.play_music_stream(victory_music_final)
            while True:
                rl.update_music_stream(victory_music_final)
                frame = int((10 * rl.get_time()) % victory_dance_frames)
                rl.draw_texture_pro(textures.victory_dance,
                                    rl.Rectangle(width * frame, 0, width, height),
                                    rl.Rectangle((rl.get_time() * 100) % (WIDTH + 260) - 130, HEIGHT - 200, width * 6, height * 6), rl.Vector2(width // 2, height // 2), 0, rl.WHITE)
                yield

    c = coro()
    while not rl.window_should_close():
        rl.update_music_stream(victory_music)
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        try:
            next(c)
        except StopIteration:
            return
        draw_text(text, (WIDTH // 2, HEIGHT // 2), size=48)
        rl.end_drawing()


def wait_for_key_press():
    while True:
        if rl.is_key_released(rl.KEY_SPACE):
            yield
            break
        yield

def wait_time(delay, allow_skip=False):
    t = rl.get_time()
    while t + delay > rl.get_time():
        if allow_skip and rl.is_key_released(rl.KEY_SPACE):
            yield
            return
        yield


def render_with_shader(from_renderbuf, to_renderbuf, shader):
    rl.begin_texture_mode(to_renderbuf)
    rl.begin_shader_mode(shader)
    rl.clear_background(rl.BLACK)
    rl.draw_texture_pro(from_renderbuf.texture, rl.Rectangle(0, 0, WIDTH // SCALE, -HEIGHT // SCALE), rl.Rectangle(0, 0, WIDTH, HEIGHT), rl.Vector2(), 0, rl.WHITE)
    rl.end_shader_mode()
    rl.end_texture_mode()

def intro_loop():
    global current_func
    text = ""
    def coro():
        nonlocal text
        text = "1. Find the ghost"
        yield from wait_for_key_press()
        text = "2. Anger it by taking its picture"
        yield from wait_for_key_press()
        text = "3. Run back to your GHOSTVAC to capture it"
        yield from wait_for_key_press()
        text = "<SPACE> shows/hides your phone"
        yield from wait_for_key_press()
        text = f"This is {maps[0].name}"
        yield from wait_time(4, allow_skip=True)

    c = iter(coro())
    while not rl.window_should_close():
        try:
            next(c)
        except StopIteration:
            current_func = game_loop
            return
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        draw_text(text, (WIDTH // 2, HEIGHT // 2), size=48)
        draw_text("(press space to continue)", (WIDTH // 2, HEIGHT // 2 + 48), size=24)
        rl.end_drawing()

def send_message(phone, text, skip_wait=False):
    if not skip_wait:
        yield from wait_time(random.uniform(0, 2))
        phone.messages.append(StatusMessage("."))
        for _ in wait_time(len(text) / 12):
            phone.messages[-1] = StatusMessage(".")
            yield from wait_time(0.3)
            phone.messages[-1] = StatusMessage("..")
            yield from wait_time(0.3)
            phone.messages[-1] = StatusMessage("...")
            yield from wait_time(0.3)
        phone.messages.pop()
    if not phone.is_messaging_active:
        rl.play_sound(sounds.notify)
    phone.add_message(TextMessage(text))
    if skip_wait:
        phone.last_read = len(phone.messages)

def messages1(phone):
    notify("<space> shows/hides your phone")
    yield from send_message(phone, textwrap.dedent("""\
                                            so, this station
                                            is haunted by
                                            ANGRY BOB
                                            the ghost of a used
                                            car salesman who just
                                            hated public transit"""), skip_wait=True)
    yield from send_message(phone, "i got u setup with\nthe state of the art\nGHOSTVAC")
    yield from send_message(phone, "just piss off that\nghost, run back to the\nvac and it'll do the rest\n:)")
    yield from send_message(phone, "oh, to piss it off\njust take a pic of it")
    yield from send_message(phone, "most ghosts hate\nthat lulz")
    yield from send_message(phone, "gl bro!")

def hint_dialog_coro(phone, hint_dialog):
    while hint_dialog:
        to_del = []
        for hint in hint_dialog:
            if hint in phone.discovered:
                for line in hint_dialog[hint]:
                    yield from send_message(phone, line)
                to_del.append(hint)
            else:
                yield
        for hint in to_del:
            del hint_dialog[hint]

def messages2(phone):
    yield from send_message(phone, textwrap.dedent("""\
                                                   uh.. i don't know
                                                   the name of the ghost
                                                   so the vac won't work"""), skip_wait=True)
    yield from send_message(phone, "(it needs a name\nto lock on)")
    yield from send_message(phone, "use your camera to\ncollect clues!")

    hint_dialog = {
        'FRED': ["YO nice, it's FRED", "i've activated the vac"],
        'DEAD': ["uh yehh he's dead.", "we already knew that."],
        }
    yield from hint_dialog_coro(phone, hint_dialog)

def messages3(phone):
    yield from send_message(phone, textwrap.dedent("""\
                                                   this station is
                                                   haunted by a
                                                   ferroequinologist"""), skip_wait=True)

    yield from send_message(phone, textwrap.dedent("""\
                                                   you know, a
                                                   person who's
                                                   obsessed with
                                                   trains."""))
    yield from send_message(phone, textwrap.dedent("""\
                                                   i totally didn't
                                                   like... google
                                                   that ahead of time
                                                   to look smart."""))

    yield from send_message(phone, textwrap.dedent("""\
                                                   anyway, you're
                                                   gonna need to
                                                   figure out his
                                                   name too."""))

    hint_dialog = {
        'HI': ['"HI"?', "uh. maybe it's friendly?"],
        'AI': ["AI...?"],
        'SICKLE T': ["uhh that sickle isn't\nominous at all"],
        'JR': ["just a junior ghost?"],
        }
    yield from hint_dialog_coro(phone, hint_dialog)
    notify("all hints found")
    notify("use your phone to reconstruct the message")
    phone.puzzle_state = 'solving'
    yield from send_message(phone, "i guess they make\nup a larger message?")
    yield from send_message(phone, "i've updated the\naberration pal app\non your phone\nso you can try\nput the pieces\ntogether.")


maps = []
for p in ("resources/station1.json", "resources/station2.json", "resources/station3.json"):
    with open(p) as f:
        maps.append(Map(json.load(f)))
maps[0].ghostvac_name = "BOB"
maps[0].messages_func = messages1
maps[1].messages_func = messages2
maps[2].messages_func = messages3

class GuiRow:
    def __init__(self, container_rect):
        self.container = rl.Rectangle(container_rect.x, container_rect.y, container_rect.width, container_rect.height)
        self.y = container_rect.y

    def _make_space(self, height):
        if height < 0:
            height *= -1
            self.container.height -= height
            y = self.container.y + self.container.height
        else:
            y = self.y
            self.y += height
        return y, height

    def row_rect(self, height, width=None):
        y, height = self._make_space(height)
        width = self.container.width if width is None else width
        return rl.Rectangle(self.container.x + (self.container.width - width) / 2, y, width, height)

    def rect_remainder(self):
        return rl.Rectangle(self.container.x, self.y, self.container.width, self.container.height - (self.container.y - self.y))

    def row_vec2(self, height, origin=(0, 0)):
        y, height = self._make_space(height)
        return vec2(self.container.x + self.container.width * origin[0], y + origin[1] * height)

# HACK
panel_scroll = rl.Vector2()
panel_view = rl.Rectangle()
@contextmanager
def scroll_panel(rect, title, content_height):
    rl.gui_scroll_panel(rect, title, rl.Rectangle(0, 0, rect.width - 15, content_height), panel_scroll, panel_view)
    rl.begin_scissor_mode(int(panel_view.x), int(panel_view.y), int(panel_view.width), int(panel_view.height))
    g = GuiRow(rl.Rectangle(rect.x + panel_scroll.x, rect.y + panel_scroll.y, rect.width, content_height))
    g.row_rect(40)
    yield g
    rl.end_scissor_mode()

notifications = []
notifications_shown = set()
def notify(text, once=False):
    global notifications
    if text in notifications_shown:
        return
    notifications_shown.add(text)
    notifications = notifications[:4]
    notifications.append((5, text))




def game_loop():
    map = maps[0]
    last_dir = vec2(1, 0)
    light_offset = vec2()
    canvas = rl.load_render_texture(WIDTH // SCALE, HEIGHT // SCALE)
    canvas2 = rl.load_render_texture(WIDTH, HEIGHT)
    rl.set_texture_wrap(canvas2.texture, rl.TEXTURE_WRAP_CLAMP)
    wait_time = 0
    last_beep = 0
    global current_func

    state.player = vec2(*map.hero_spawn)
    state.ghost_pos = vec2()
    state.ghost_health = 1
    for i in range(100):
        state.ghost_pos = vec2(random.choice(map.wander_points))
        if length(state.ghost_pos - player_origin()) > 500:
            break
    if i == 99:
        print("WARNING: RANDOMNESS FAILED?!!")
    state.ghost_target = vec2(0, 0)
    state.target_ttl = 0
    state.ghost_state = 'wandering'
    #notifications.clear()

    pnoise = perlin_noise.PerlinNoise(octaves=5)

    state.phone = state.phone or Phone(map.messages_func or None)
    fear = 0
    MAX_HEALTH = 5
    fear_health = MAX_HEALTH

    ghost_nframes = 5
    def ghost_rect():
        frame_width = textures.ghost.width / ghost_nframes
        return rl.Rectangle(state.ghost_pos.x - frame_width / 2,
                     state.ghost_pos.y - textures.ghost.height / 2,
                     frame_width * state.ghost_health,
                     textures.ghost.height * state.ghost_health)

    step_time = 0
    STEP_LENGTH = 0.2

    rl.play_music_stream(bg_soundscape)
    rl.play_music_stream(scared_loop)

    while not rl.window_should_close():
        state.run_time += rl.get_frame_time()
        #prof = Profiler()
        #prof.start()
        rl.update_music_stream(bg_soundscape)
        rl.update_music_stream(scared_loop)
        rl.set_music_volume(scared_loop, clamp((1 - (fear_health / MAX_HEALTH)) * 2, 0, 1))
        phone = state.phone
        input = vec2()
        if rl.is_key_down(rl.KEY_DOWN): input.y += 1
        if rl.is_key_down(rl.KEY_UP): input.y -= 1
        if rl.is_key_down(rl.KEY_LEFT): input.x -= 1
        if rl.is_key_down(rl.KEY_RIGHT): input.x += 1

        if rl.is_key_down(rl.KEY_W): input.y -= 1
        if rl.is_key_down(rl.KEY_A): input.x -= 1
        if rl.is_key_down(rl.KEY_S): input.y += 1
        if rl.is_key_down(rl.KEY_D): input.x += 1

        if (l := length(player_origin() - state.ghost_pos)) < ENRAGE_RANGE * 2 and state.ghost_state == 'enraged':
            fear = clamp(fear + rl.get_frame_time() * 2, 0, 1)
        else:
            fear = clamp(fear - rl.get_frame_time(), 0, 1)
        if fear > 0:
            fear_health -= fear * rl.get_frame_time()
        else:
            fear_health = min(fear_health + 0.3 * rl.get_frame_time(), MAX_HEALTH)
        if fear_health <= 0:
            current_func = dead_loop

            if map.ghostvac_name is None:
                notify("hint: you need to discover the ghost name before taking it's pic", once=True)
                phone.add_message(TextMessage("noo, dude you\ngotta find the\nghost's name before\ntriggering it."))
                phone.add_message(TextMessage("use your camera\nto snap clues!"))
            else:
                notify("hint: run back to the GHOSTVAC before the ghost catches you", once=True)

            return

        #if rl.is_key_released(rl.KEY_X):
        #    rl.play_sound(sounds.raspberry)
        #    if l < ENRAGE_RANGE:
        #        state.ghost_state = 'enraged'

        #if rl.is_key_released(rl.KEY_SPACE):
        #    pulses.append(Pulse(player_origin(), 0))

        if (vec_length := length(input)) > 1:
            input /= vec_length

        prev_pos = vec2(state.player)
        speed_mod = 0.5 if phone._is_showing else 1
        state.player += input * speed_mod * 200 * rl.get_frame_time()

        map.resolve_collisions(state.player, (PLAYER_SIZE, PLAYER_SIZE))
        draw_rec = player_irec()
        if length(input) > 0:
            step_time += rl.get_frame_time() * speed_mod
            if step_time > STEP_LENGTH:
                step_time = 0
                rl.play_sound(random.choice(sounds.step))
            if step_time > STEP_LENGTH / 2:
                draw_rec.y -= 2

        if input != vec2():
            last_dir = normalize(input)

        if any([rl.check_collision_recs(z, player_rec()) for z in map.exit]):
            if state.ghost_state == 'sucked':
                current_func = victory_loop
                return
            else:
                state.player = prev_pos
                notify("I can't leave until I've dealt with the ghost")

        shake = vec2(pnoise([camera.target.x + rl.get_time() / 5]), pnoise([camera.target.y + rl.get_time() / 5])) * 10
        shake_amt = (1 - (fear_health / MAX_HEALTH))
        camera.target = (vec2(ivec2(player_origin())) + shake * shake_amt).to_tuple()

        #for p in pulses:
        #    if p.active:
        #        p.ttl -= rl.get_frame_time()
        #        p.size += rl.get_frame_time() * 80
        #pulses[:] = [p for p in pulses if p.ttl > 0]
        #ghost_pulses[:] = [p for p in ghost_pulses if rl.get_time() < p[2]]

        # move ghost
        ghost_to_vac_dist = length(state.ghost_pos - map.ghostvac)
        if state.ghost_state == 'wandering':
            state.target_ttl -= rl.get_frame_time()
            if state.target_ttl <= 0:
                state.target_ttl = random.uniform(6, 20)

                state.ghost_target = map.get_next_wander_point(state.ghost_pos)

            if map.name == 'Fulton St' and not rl.is_sound_playing(sounds.toottoot) and (rl.get_time() % 10 < 1):
                rl.play_sound(sounds.toottoot)
                rl.set_sound_volume(sounds.toottoot, clamp((800 - length(state.player - state.ghost_pos)) / 800, 0.1, 0.7))

        elif state.ghost_state == 'enraged':
            state.ghost_target = state.player
            if map.ghostvac_name is not None and ghost_to_vac_dist < SUCK_RANGE:
                state.ghost_state = 'sucking'
                rl.play_sound(sounds.vacuum)
        elif state.ghost_state == 'sucking':
            state.ghost_target = map.ghostvac + 32
            if ghost_to_vac_dist < 64:
                state.ghost_health -= rl.get_frame_time() * 0.5
            if state.ghost_health < 0:
                state.ghost_state = 'sucked'
                rl.play_sound(sounds.ghost_sucked)

        state.ghost_pos += mclamp(state.ghost_target - state.ghost_pos, 10 * rl.get_frame_time())
        #state.ghost_pos = clamp(state.ghost_pos, (0, 0), (WIDTH, HEIGHT))

        dist = length(state.ghost_pos - state.player)
        rate = 0 if dist > 200 else (200 - dist) / 200
        if state.ghost_state != 'sucked' and rate > 0 and last_beep + (1 - min(0.8, rate)) * 0.2 < rl.get_time():
            last_beep = rl.get_time()
            rl.set_sound_pitch(sounds.beep, 1 + max(0, rate - 0.5))
            rl.play_sound(sounds.beep)
        phone.update(rl.get_frame_time(), shake * 20 * shake_amt)

        # draw
        rl.begin_texture_mode(canvas)
        rl.begin_mode_2d(camera)
        rl.clear_background(rl.BLACK)

        map.draw()

        add_texture_sorted(textures.hero, (0, 0, (-1 if last_dir.x < 0 else 1) * TILE_SIZE, TILE_SIZE), draw_rec)

        sorted_texs.sort(key=lambda x: x[2].y)
        for tex, src, dest in sorted_texs:
            rl.draw_texture_pro(tex, src, dest, rl.Vector2(), 0, rl.WHITE)
        sorted_texs.clear()
        #rl.draw_rectangle_rec(player_irec(), rl.BLUE)
        #for pos, size, death_time in ghost_pulses:
        #    col = rl.fade(rl.GRAY, (death_time - rl.get_time()) / GHOST_TRAIL_TTL)
        #    rl.draw_circle_lines_v(pos, size, col)
        #for pulse in pulses:
        #    if abs(length(state.ghost_pos - pulse.pos) - pulse.size) < 5:
        #        ghost_pulses.append((pulse.pos.to_tuple(), pulse.size, rl.get_time() + GHOST_TRAIL_TTL))
        #    else:
        #        rl.draw_circle_lines_v(pulse.pos.to_tuple(), pulse.size, rl.WHITE)

        if state.ghost_state != 'wandering':
            frame_width = textures.ghost.width / ghost_nframes
            if state.ghost_state == 'enraged':
                #rl.draw_circle_v(state.ghost_pos.to_tuple(), 20, rl.RED)
                frame_i = int((rl.get_time() * 15) % 4)
                if not rl.is_sound_playing(sounds.howl):
                    rl.play_sound(sounds.howl)
                    rl.set_sound_volume(sounds.howl, clamp((500 - length(state.player - state.ghost_pos)) / 500, 0.1, 1))
            if state.ghost_state == 'sucking':
                frame_i = 4
            add_texture_sorted(textures.ghost,
                               rl.Rectangle(frame_width * frame_i,
                                            0,
                                            frame_width * (-1 if state.ghost_target.x < state.ghost_pos.x else 1),
                                            textures.ghost.height),
                               ghost_rect())

        rl.end_mode_2d()
        rl.end_texture_mode()

        loc = rl.get_shader_location(flashlight_shader, "pos")
        # lerp light
        light_offset += last_dir - light_offset
        if (l := length(light_offset)):
            light_offset /= l

        p = vec2(WIDTH // 2, HEIGHT // 2) + light_offset * 150
        rl.set_shader_value(flashlight_shader, loc, rl.Vector2(p.x, HEIGHT - p.y), rl.SHADER_UNIFORM_VEC2)
        render_with_shader(canvas, canvas2, flashlight_shader)

        # render ghost for cam
        if state.ghost_state == 'wandering':
            frame_width = textures.ghosthappy.width / 3
            frame_i = int((rl.get_time() * 15) % 4)
            rl.begin_texture_mode(canvas)
            rl.draw_texture_pro(textures.ghosthappy,
                                (frame_width * frame_i,
                                 0,
                                 frame_width * (-1 if state.ghost_target.x < state.ghost_pos.x else 1),
                                 textures.ghosthappy.height),
                                rect_to_screen_space(ghost_rect()),
                                (0, 0),
                                0,
                                rl.fade(rl.WHITE, 0.5))
            rl.end_texture_mode()


        rl.begin_texture_mode(canvas2)
        # draw phone ui
        rl.draw_rectangle_rounded(phone.rect, 0.1, 5, rl.WHITE)
        rl.begin_scissor_mode(int(phone.screen_rect.x), int(phone.screen_rect.y), int(phone.screen_rect.width), int(phone.screen_rect.height))
        gui = GuiRow(phone.screen_rect)
        r = gui.row_rect(20)
        rl.draw_rectangle_rec(r, rl.BLACK)
        rl.draw_rectangle_rec(rl.Rectangle(r.x + r.width - 24, r.y + 4, 18, r.height - 8), rl.WHITE)
        rl.draw_rectangle_rec(rl.Rectangle(r.x + r.width - 8, r.y + 6, 4, r.height - 12), rl.WHITE)
        bottom_row = gui.row_rect(-50)

        if phone.page == 0:
            rl.draw_rectangle_rec(gui.rect_remainder(), rl.GRAY)
            gui.row_rect(10)
            gui.row_rect(-60)
            if phone._is_showing and phone.last_read_ttl > 0:
                phone.last_read_ttl -= rl.get_frame_time()
                if phone.last_read_ttl <= 0:
                    phone.last_read = len(phone.messages)
            with scroll_panel(gui.rect_remainder(), "Messages: donnie", sum(m.height for m in phone.messages) + 40) as panel_gui:
                for i, message in enumerate(phone.messages):
                    if i == phone.last_read and not isinstance(message, StatusMessage):
                        draw_text("----- unread  ", panel_gui.row_vec2(28, origin=(1, 0.5)), origin=(1, 0.5), color=rl.ORANGE)
                    message.draw(panel_gui.row_vec2(message.height) + vec2(10, 0))
        elif phone.page == 1:
            rl.draw_rectangle_rec(gui.rect_remainder(), rl.PURPLE)
            draw_text("Aberration Pal", gui.row_vec2(60, origin=(0.5, 0.5)), color=rl.WHITE)
            if rl.gui_button(gui.row_rect(50, width=120), "Scan"):
                phone.scan()
            gui.row_rect(10)
            draw_text(phone.scan_results, gui.row_vec2(30) + vec2(5, 0), origin=(0, 0), color=rl.BLACK)
            draw_text("TODO", gui.row_vec2(30) + vec2(5, 0), origin=(0, 0), color=rl.BLACK)
            draw_todo_item("- discover ghost name", gui.row_vec2(30) + vec2(5, 0), origin=(0, 0.5), is_done=map.ghostvac_name is not None)
            draw_todo_item("- trigger ghost", gui.row_vec2(30) + vec2(5, 0), origin=(0, 0.5), is_done=state.ghost_state != 'wandering')
            draw_todo_item("- capture in vac", gui.row_vec2(30) + vec2(5, 0), origin=(0, 0.5), is_done=state.ghost_state == 'sucked')
            draw_todo_item("- leave the station", gui.row_vec2(30) + vec2(5, 0), origin=(0, 0.5), is_done=False)
            row = gui.row_rect(textures.a1.height)
            if phone.puzzle_state == 'solving' or phone.puzzle_state == 'solved':
                for i, tex in enumerate(phone.clue_textures):
                    w = tex.width * SCALE
                    h = tex.height * SCALE
                    cw = (w + (3 if phone.puzzle_state == 'solving' else 0))
                    x = row.x + i * cw + (row.width - (cw * len(phone.clue_textures))) / 2
                    rl.draw_texture_pro(tex,
                                        tex_rect(tex),
                                        (x, row.y, w, h),
                                        (0, 0),
                                        0,
                                        rl.WHITE)

                    if phone.puzzle_state == 'solving':
                        if i < len(phone.clue_textures)-1 and rl.gui_button((x + w - 20, row.y + h + 20, 40, 20), "<>"):
                            other = (i + 1) % len(phone.clue_textures)
                            phone.clue_textures[i], phone.clue_textures[other] = phone.clue_textures[other], phone.clue_textures[i]
                if phone.puzzle_state == 'solving' and phone.clue_textures == [textures.a1, textures.a2, textures.a3, textures.a4]:
                    phone.puzzle_state = 'solved'
                    map.ghostvac_name = "ARTHUR"

        elif phone.page == 2:
            rl.draw_texture_pro(canvas.texture, rl.Rectangle(0, 0, WIDTH // SCALE, -HEIGHT // SCALE), rl.Rectangle(0, 0, WIDTH, HEIGHT), rl.Vector2(), 0, rl.Color(220, 220, 255, 255))

            draw_text('CAMERA', gui.row_vec2(30, origin=(0.5, 0.5)), origin=(0.5, 0.5), color=rl.GREEN)
            button_pos = gui.row_vec2(-100, origin=(0.5, 0.5))
            if rl.gui_button(rl.Rectangle(button_pos.x - 20, button_pos.y - 20, 40, 40), ""):
                rl.play_sound(sounds.camera)
                img = rl.load_image_from_texture(canvas.texture)
                rl.image_crop(img, phone.screen_space_rect)
                if phone.last_pic is not None:
                    rl.unload_texture(phone.last_pic)
                phone.last_pic = rl.load_texture_from_image(img)
                rl.unload_image(img)

                if phone.in_viewfinder(ghost_rect()):
                    state.ghost_state = 'enraged'

                for rec, hint in map.hints:
                    if phone.in_viewfinder(rec):
                        notify(f"Discovered: {hint}")
                        if hint == 'FRED':
                            map.ghostvac_name = "FRED"
                            rl.play_sound(sounds.bling)
                        phone.discovered.add(hint)


            last_pic_preview = rl.Rectangle(phone.screen_rect.x + 10, phone.screen_rect.y + 400, 60, 110)
            if phone.last_pic:
                rl.draw_texture_pro(phone.last_pic,
                                    tex_rect(phone.last_pic, flipv=True),
                                    last_pic_preview,
                                    rl.Vector2(0, 0),
                                    0,
                                    rl.WHITE)
            else:
                rl.draw_rectangle_rec(last_pic_preview, rl.GRAY)
            icon_width, icon_height = textures.camera_icon.width, textures.camera_icon.height
            rl.draw_texture_pro(textures.camera_icon,
                                rl.Rectangle(0, 0, icon_width, icon_height),
                                rl.Rectangle(button_pos.x - icon_width, button_pos.y - icon_height, icon_width * 2, icon_height * 2),
                                rl.Vector2(0, 0), 0, rl.WHITE)
        elif phone.page == 3:
            gui.row_rect(20)
            rl.draw_texture_pro(textures.ghostvac_off,
                                tex_rect(textures.ghostvac_off),
                                gui.row_rect(120, 120),
                                (0, 0),
                                0,
                                rl.WHITE)

            draw_text("GHOSTVAC 3000", gui.row_vec2(40, origin=(0.5, 0.5)), color=rl.BLACK, size=32)
            if map.ghostvac_name:
                draw_text(f"Target name: {map.ghostvac_name}", gui.row_vec2(30, origin=(0.5, 0.5)), color=rl.GREEN)
                draw_text("Status: ARMED", gui.row_vec2(30, origin=(0.5, 0.5)), color=rl.GREEN)
                draw_text("Ready to vacuum some", gui.row_vec2(15, origin=(0.5, 0.5)), color=rl.GREEN)
                draw_text("VERMIN", gui.row_vec2(15, origin=(0.5, 0.5)), color=rl.GREEN)
                draw_text(">:)", gui.row_vec2(30, origin=(0.5, 0.5)), color=rl.GREEN)
            else:
                draw_text("Target name: ???", gui.row_vec2(30, origin=(0.5, 0.5)), color=rl.GRAY)
                draw_text("Status: INACTIVE", gui.row_vec2(30, origin=(0.5, 0.5)), color=rl.RED)
                draw_text("Required: TARGET NAME", gui.row_vec2(15, origin=(0.5, 0.5)), color=rl.GRAY)

        for t in map.turnstiles:
            if length(player_origin() - (t.rec.x + TILE_SIZE / 2, t.rec.y + TILE_SIZE / 2)) < 20 and t.locked:
                with phone.popup("Unlock turnstile?", gui.row_rect(140)) as p:
                    if rl.gui_button(rl.Rectangle(p.x, p.y, 250, 50), "Swipe transit card"):
                        t.locked = False
                        t.unlock_time = 8
                        phone.hide()
                        rl.play_sound(sounds.turnstile)

        if rl.gui_button(rl.Rectangle(bottom_row.x, bottom_row.y, 50, bottom_row.height), "<-"):
            phone.page = (phone.page - 1) % phone.pages
        if rl.gui_button(rl.Rectangle(bottom_row.x + bottom_row.width - 50, bottom_row.y, 50, bottom_row.height), "->"):
            phone.page = (phone.page + 1) % phone.pages
        for i in range(phone.pages):
            center = (bottom_row.x + bottom_row.width / 2 - 30 + (60 / phone.pages) * i,
                      bottom_row.y + bottom_row.height / 2)
            if i == phone.page:
                rl.draw_circle_v(center, 5, rl.BLACK)
            else:
                rl.draw_circle_lines_v(center, 5, rl.BLACK)
        rl.end_scissor_mode()

        if rl.is_key_released(rl.KEY_SPACE):
            if phone._is_showing:
                phone.hide()
            else:
                phone.show()

        notifications[:] = [(ttl - rl.get_frame_time(), text) for ttl, text in notifications if ttl > 0]
        for i, n in enumerate(notifications):
            draw_text(n[1], (WIDTH // 2, 40 + 20 * i), color=rl.fade(rl.WHITE, min(n[0], 1)))

        rl.end_texture_mode()

        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        loc = rl.get_shader_location(chromatic_aberration, "rOffset")
        rl.set_shader_value(chromatic_aberration, loc, rl.Vector2(0.01 * fear), rl.SHADER_UNIFORM_VEC2)
        loc = rl.get_shader_location(chromatic_aberration, "bOffset")
        rl.set_shader_value(chromatic_aberration, loc, rl.Vector2(0, 0.005 * fear), rl.SHADER_UNIFORM_VEC2)
        rl.begin_shader_mode(chromatic_aberration)
        rl.draw_texture_pro(canvas2.texture, rl.Rectangle(0, 0, WIDTH, -HEIGHT), rl.Rectangle(0, 0, WIDTH, HEIGHT), rl.Vector2(), 0, rl.WHITE)
        rl.end_shader_mode()
        #prof.stop()
        #if rl.get_frame_time() > 0.017:
        #    prof.print(show_all=True, timeline=True)

        #rl.draw_fps(10, 10)

        rl.end_drawing()


current_func = intro_loop

rl.set_target_fps(60)
try:
    while not rl.window_should_close():
        current_func()
finally:
    rl.close_audio_device()
