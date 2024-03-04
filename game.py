import pyray as rl
from glm import *
import json
from types import SimpleNamespace
from dataclasses import dataclass
from collections import defaultdict, deque
import util
import time
import random
import itertools
import os
import re
from contextlib import contextmanager

PLAYER_SIZE = 16
TILE_SIZE = 16

class SpatialHash:
    cell_size = 128
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

@dataclass
class Turnstile:
    rec: rl.Rectangle
    locked: bool = True


@contextmanager
def popup(title, width=300, height=200, padding=10):
    rect = rl.Rectangle(WIDTH // 2 - width // 2, HEIGHT // 2 - height // 2, width, height)
    rl.draw_rectangle_rec(rect, rl.GRAY)
    draw_text(title, (WIDTH // 2, HEIGHT // 2 - height // 2 + 10))
    rect.y += 30 + padding
    rect.height -= 2 * padding
    rect.x += padding
    rect.width -= 2 * padding
    yield rect

class Map:
    def __init__(self, map_json):
        def get_layer(name):
            r, = [x for x in map_json["layers"] if x["name"] == name]
            return r

        self.name = map_json["values"]["StationName"]
        self.entities = get_layer("Entities")["entities"]
        self.decals = get_layer("Decals")["decals"]
        self.hero_spawn, = self.get_entity_pos("hero")
        self.wander_points = self.get_entity_pos("spawnpoint")
        self.exit = [rl.Rectangle(z["x"], z["y"], z["width"], z["height"]) for z in self.entities if z["name"] == "safezone"]
        self.turnstiles = [Turnstile(rl.Rectangle(z.x, z.y, TILE_SIZE, TILE_SIZE)) for z in self.get_entity_pos("turnstile")]
        self.walls = get_layer("WallTiles")["data2D"]
        self.wall_recs = SpatialHash([rl.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE) for y, row in enumerate(self.walls) for x, t in enumerate(row) if t != -1])
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
        # bg layer
        for x, y, tile in tiles_around(self.bg, ivec2(state.player // TILE_SIZE)):
            if tile == -1:
                continue
            rl.draw_texture_pro(textures.tiles,
                                rl.Rectangle(tile * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE),
                                rl.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                                rl.Vector2(),
                                0,
                                rl.WHITE)

        for x, y, tile in tiles_around(self.walls, ivec2(state.player // TILE_SIZE)):
            if tile == -1:
                continue
            rl.draw_texture_pro(textures.tiles,
                                rl.Rectangle(tile * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE),
                                rl.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                                rl.Vector2(),
                                0,
                                rl.WHITE)

        for decal in self.decals:
            tex = getattr(textures, decal["texture"].removesuffix(".png"))
            rl.draw_texture(tex, decal["x"] - tex.width * decal["originX"], decal["y"] - tex.height - decal["originY"], rl.WHITE)

        for t in self.turnstiles:
            if t.locked:
                rl.draw_rectangle_rec(t.rec, rl.LIGHTGRAY)
            else:
                rl.draw_rectangle_lines_ex(t.rec, 1, rl.LIGHTGRAY)

    def resolve_collisions(self, entity_pos, entity_size=(TILE_SIZE, TILE_SIZE)):
        entity_rec = rl.Rectangle(entity_pos[0], entity_pos[1], entity_size[0], entity_size[1])
        if (p := util.resolve_map_collision(self.wall_recs.near(entity_pos) + [t.rec for t in self.turnstiles if t.locked], entity_rec)) is not None:
            entity_pos[0], entity_pos[1] = p[0], p[1]





SCALE = 4
WIDTH, HEIGHT = 300 * SCALE, 200 * SCALE
rl.init_window(WIDTH, HEIGHT, "Aberration Station")
rl.init_audio_device()

font = rl.load_font_ex("mago3.ttf", 24, None, 0)
rl.gui_set_font(font)
rl.gui_set_style(rl.DEFAULT, rl.TEXT_SIZE, 24)

textures = SimpleNamespace()
for f in os.listdir('.'):
    if f.endswith('.png'):
        setattr(textures, f.removesuffix('.png'), rl.load_texture(f))

# setup repeat sounds
sounds = SimpleNamespace()
sound_files = {x for x in os.listdir('.') if x.endswith('.wav')}
repeats = {(g.group(1), x) for x in sound_files if (g := re.search(r'(\w+)_(\d+)\.wav$', x))}
for k, g in itertools.groupby(sorted(repeats), key=lambda x: x[0]):
    setattr(sounds, k, [rl.load_sound(x[1]) for x in g])
sound_files -= repeats
for f in sound_files:
    setattr(sounds, f.removesuffix('.wav'), rl.load_sound(f))

bg_soundscape = rl.load_music_stream("bg.ogg")
victory_music = rl.load_music_stream("victory.ogg")
victory_music.looping = False

GHOST_TRAIL_TTL = 2

state = SimpleNamespace()

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

pulses = []
ghost_pulses = []

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

def tiles_around(tiles_2d, tile_coord, around = 10):
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

def draw_text(text, pos, origin=(0.5, 0.5), size=24, color=rl.WHITE):
    spacing = 1.3
    v = rl.measure_text_ex(font, text, size, spacing)
    pos = (pos[0] - v.x * origin[0], pos[1] - v.y * origin[1])
    rl.draw_text_ex(font, text, (pos[0], pos[1]), size, spacing, color)


def victory_loop():
    rl.stop_sound(sounds.howl)
    rl.play_music_stream(victory_music)
    text = ""
    def coro():
        nonlocal text
        global current_func
        text = "You escaped!"
        yield from wait_time(4)
        if maps:
            text = "On to the next station:"
            yield from wait_time(4, allow_skip=True)
            text = f"This is {maps[0].name}"
            yield from wait_time(4, allow_skip=True)
            current_func = game_loop
        else:
            text = "You win! <INSERT VICTORY DANCE>"
            while True:
                yield

    c = coro()
    while not rl.window_should_close():
        try:
            next(c)
        except StopIteration:
            return
        rl.update_music_stream(victory_music)
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        draw_text(text, (WIDTH // 2, HEIGHT // 2), size=48)
        rl.end_drawing()


def wait_time(delay, allow_skip=False):
    t = rl.get_time()
    while t + delay > rl.get_time():
        if allow_skip and rl.is_key_released(rl.KEY_SPACE):
            yield
            return
        yield

def intro_loop():
    global current_func
    text = ""
    def coro():
        nonlocal text
        text = "Find the ghost, then return to the entrance."
        yield from wait_time(4, allow_skip=True)
        text = "<SPACE> sends a pulse to detect abberations"
        yield from wait_time(4, allow_skip=True)
        text = "<X> pisses off the ghost when you're in range"
        yield from wait_time(4, allow_skip=True)
        text = "Don't get caught."
        yield from wait_time(4, allow_skip=True)
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
        rl.end_drawing()


maps = []
for p in ("station1.json", "station2.json"):
    with open(p) as f:
        maps.append(Map(json.load(f)))

def game_loop():
    map = maps.pop(0)
    last_dir = vec2(1, 0)
    canvas = rl.load_render_texture(WIDTH // SCALE, HEIGHT // SCALE)
    wait_time = 0
    last_beep = 0
    global current_func

    state.player = vec2(*map.hero_spawn)
    state.ghost_pos = vec2()
    for i in range(100):
        state.ghost_pos = vec2(random.choice(map.wander_points))
        if length(state.ghost_pos - player_origin()) > 500:
            break
    if i == 99:
        print("WARNING: RANDOMNESS FAILED?!!")
    state.ghost_target = vec2(0, 0)
    state.target_ttl = 0
    state.ghost_state = 'wandering'
    notifications = []

    def notify(text, ttl=5):
        nonlocal notifications
        notifications = notifications[:4]
        notifications.append((5, text))

    step_time = 0
    STEP_LENGTH = 0.2

    rl.play_music_stream(bg_soundscape)
    while not rl.window_should_close():
        rl.update_music_stream(bg_soundscape)
        update_time = time.time()
        input = vec2()
        if rl.is_key_down(rl.KEY_DOWN): input.y += 1
        if rl.is_key_down(rl.KEY_UP): input.y -= 1
        if rl.is_key_down(rl.KEY_LEFT): input.x -= 1
        if rl.is_key_down(rl.KEY_RIGHT): input.x += 1

        if rl.is_key_released(rl.KEY_X):
            rl.play_sound(sounds.raspberry)
            if length(player_origin() - state.ghost_pos) < 30:
                state.ghost_state = 'enraged'

        input.x += rl.get_gamepad_axis_movement(0, rl.GAMEPAD_AXIS_LEFT_X)
        input.y += rl.get_gamepad_axis_movement(0, rl.GAMEPAD_AXIS_LEFT_Y)

        if rl.is_key_released(rl.KEY_SPACE):
            pulses.append(Pulse(player_origin(), 0))

        if (vec_length := length(input)) > 1:
            input /= vec_length

        prev_pos = vec2(state.player)
        state.player += input * 200 * rl.get_frame_time()

        if input != vec2():
            last_dir = normalize(input)

        map.resolve_collisions(state.player, (PLAYER_SIZE, PLAYER_SIZE))

        if any([rl.check_collision_recs(z, player_rec()) for z in map.exit]):
            if state.ghost_state == 'enraged':
                current_func = victory_loop
                return
            else:
                state.player = prev_pos
                notify("I can't leave until I've dealt with the ghost")

        camera.target = ivec2(player_origin()).to_tuple()

        for p in pulses:
            if p.active:
                p.ttl -= rl.get_frame_time()
                p.size += rl.get_frame_time() * 80
        pulses[:] = [p for p in pulses if p.ttl > 0]
        ghost_pulses[:] = [p for p in ghost_pulses if rl.get_time() < p[2]]

        # move ghost
        if state.ghost_state == 'wandering':
            state.target_ttl -= rl.get_frame_time()
            if state.target_ttl <= 0:
                state.target_ttl = random.uniform(6, 20)

                state.ghost_target = map.get_next_wander_point(state.ghost_pos)
        else:
            state.ghost_target = state.player

        state.ghost_pos += mclamp(state.ghost_target - state.ghost_pos, 10 * rl.get_frame_time())
        #state.ghost_pos = clamp(state.ghost_pos, (0, 0), (WIDTH, HEIGHT))

        dist = length(state.ghost_pos - state.player)
        rate = 0 if dist > 200 else (200 - dist) / 200
        if rate > 0 and last_beep + (1 - min(0.8, rate)) * 0.2 < rl.get_time():
            last_beep = rl.get_time()
            rl.set_sound_pitch(sounds.beep, 1 + max(0, rate - 0.5))
            rl.play_sound(sounds.beep)

        update_time = (time.time() - update_time) * 1000

        # draw
        draw_time = time.time()
        rl.begin_texture_mode(canvas)
        rl.begin_mode_2d(camera)
        rl.clear_background(rl.DARKGRAY)

        map.draw()

        for rec in map.wall_recs.near(state.player):
            rl.draw_rectangle_lines_ex(rec, 1, rl.RED)

        draw_rec = player_irec()
        if length(input) > 0:
            step_time += rl.get_frame_time()
            if step_time > STEP_LENGTH:
                step_time = 0
                rl.play_sound(random.choice(sounds.step))
            if step_time > STEP_LENGTH / 2:
                draw_rec.y -= 2

        rl.draw_texture_pro(textures.hero, (0, 0, (-1 if last_dir.x < 0 else 1) * TILE_SIZE, TILE_SIZE), draw_rec, rl.Vector2(), 0, rl.WHITE)
        #rl.draw_rectangle_rec(player_irec(), rl.BLUE)
        for pos, size, death_time in ghost_pulses:
            col = rl.fade(rl.GRAY, (death_time - rl.get_time()) / GHOST_TRAIL_TTL)
            rl.draw_circle_lines_v(pos, size, col)
        for pulse in pulses:
            if abs(length(state.ghost_pos - pulse.pos) - pulse.size) < 5:
                ghost_pulses.append((pulse.pos.to_tuple(), pulse.size, rl.get_time() + GHOST_TRAIL_TTL))
            else:
                rl.draw_circle_lines_v(pulse.pos.to_tuple(), pulse.size, rl.WHITE)

        if state.ghost_state == 'enraged':
            rl.draw_circle_v(state.ghost_pos.to_tuple(), 20, rl.RED)
            if not rl.is_sound_playing(sounds.howl):
                rl.play_sound(sounds.howl)
                rl.set_sound_volume(sounds.howl, clamp((500 - length(state.player - state.ghost_pos)) / 500, 0.1, 1))

        rl.end_mode_2d()
        rl.end_texture_mode()
        draw_time = (time.time() - draw_time) * 1000

        loc = rl.get_shader_location(flashlight_shader, "pos")
        p = vec2(WIDTH // 2, HEIGHT // 2) + last_dir * 150
        rl.set_shader_value(flashlight_shader, loc, rl.Vector2(p.x, HEIGHT - p.y), rl.SHADER_UNIFORM_VEC2)
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        rl.begin_shader_mode(flashlight_shader)
        rl.draw_texture_pro(canvas.texture, rl.Rectangle(0, 0, WIDTH // SCALE, -HEIGHT // SCALE), rl.Rectangle(0, 0, WIDTH, HEIGHT), rl.Vector2(), 0, rl.WHITE)
        rl.end_shader_mode()

        for t in map.turnstiles:
            if length(player_origin() - (t.rec.x + TILE_SIZE / 2, t.rec.y + TILE_SIZE / 2)) < 20 and t.locked:
                with popup("Pay fare?") as p:
                    if rl.gui_button(rl.Rectangle(p.x, p.y, 100, 30), "Unlock"):
                        t.locked = False

        notifications[:] = [(ttl - rl.get_frame_time(), text) for ttl, text in notifications if ttl > 0]
        for i, n in enumerate(notifications):
            draw_text(n[1], (WIDTH // 2, 40 + 20 * i), color=rl.fade(rl.WHITE, min(n[0], 1)))

        rl.draw_fps(10, 10)
        rl.draw_text(f"update: {update_time:0.8f}", 10, 30, 20, rl.WHITE)
        rl.draw_text(f"draw: {draw_time:0.8f}", 10, 50, 20, rl.WHITE)
        rl.draw_text(f"wait_time: {wait_time:0.8f}", 10, 70, 20, rl.WHITE)
        wait_time = time.time()
        rl.end_drawing()
        wait_time = (time.time() - wait_time) * 1000

current_func = intro_loop

rl.set_target_fps(60)
try:
    while not rl.window_should_close():
        current_func()
finally:
    rl.close_audio_device()
