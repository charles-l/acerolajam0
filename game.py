import pyray as rl
from glm import *
import json
from types import SimpleNamespace
from dataclasses import dataclass
from collections import defaultdict
import util
import time
import random

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

with open("station2.json") as f:
    map_json = json.load(f)
entities = [x for x in map_json["layers"] if x["name"] == "Entities"][0]["entities"]
wander_points = [vec2(p["x"], p["y"]) for p in entities]
walls = [x for x in map_json["layers"] if x["name"] == "WallTiles"][0]["data2D"]
wall_recs = SpatialHash([rl.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE) for y, row in enumerate(walls) for x, t in enumerate(row) if t != -1])
bg = [x for x in map_json["layers"] if x["name"] == "BGTiles"][0]["data2D"]

SCALE = 4
WIDTH, HEIGHT = 300 * SCALE, 200 * SCALE
rl.init_window(WIDTH, HEIGHT, "My awesome game")
rl.init_audio_device()

beep = rl.load_sound("beep.wav")
howl = rl.load_sound("noise.wav")
tiles_tex = rl.load_texture("tiles.png")
GHOST_TRAIL_TTL = 2

state = SimpleNamespace(
    player = vec2(),
    ghost_pos = vec2(random.choice(wander_points)),
    ghost_target = vec2(0, 0),
    target_ttl = 0,
    ghost_state = 'wandering',
    )

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

last_beep = 0

def tiles_around(tiles_2d, tile_coord, around = 10):
    ys = range(int(max(0, tile_coord[1]-around)), int(min(len(tiles_2d), tile_coord[1]+around)))
    xs = range(int(max(0, tile_coord[0]-around)), int(min(len(tiles_2d[0]), tile_coord[0]+around)))
    for y in ys:
        for x in xs:
            yield x, y, tiles_2d[y][x]

last_dir = vec2(1, 0)
canvas = rl.load_render_texture(WIDTH // SCALE, HEIGHT // SCALE)
wait_time = 0
rl.set_target_fps(60)
try:
    while not rl.window_should_close():
        update_time = time.time()
        input = vec2()
        if rl.is_key_down(rl.KEY_DOWN): input.y += 1
        if rl.is_key_down(rl.KEY_UP): input.y -= 1
        if rl.is_key_down(rl.KEY_LEFT): input.x -= 1
        if rl.is_key_down(rl.KEY_RIGHT): input.x += 1

        if rl.is_key_released(rl.KEY_X):
            if length(state.player - state.ghost_pos) < 30:
                state.ghost_state = 'enraged'

        input.x += rl.get_gamepad_axis_movement(0, rl.GAMEPAD_AXIS_LEFT_X)
        input.y += rl.get_gamepad_axis_movement(0, rl.GAMEPAD_AXIS_LEFT_Y)

        if rl.is_key_released(rl.KEY_SPACE):
            pulses.append(Pulse(vec2(state.player), 0))

        if (vec_length := length(input)) > 1:
            input /= vec_length

        state.player += input * 200 * rl.get_frame_time()

        if input != vec2():
            last_dir = normalize(input)

        if (p := util.resolve_map_collision(wall_recs.near(state.player), rl.Rectangle(state.player.x, state.player.y, PLAYER_SIZE, PLAYER_SIZE))) is not None:
            state.player = p

        camera.target = ivec2(state.player).to_tuple()

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

                state.ghost_target = random.choice([random.choice(wander_points)] +
                                                   [state.ghost_pos + vec2(random.uniform(-40, 40), random.uniform(-40, 40))] +
                                                   ([state.player] if length(state.player - state.ghost_pos) < 500 else []))
        else:
            state.ghost_target = state.player

        state.ghost_pos += mclamp(state.ghost_target - state.ghost_pos, 10 * rl.get_frame_time())
        #state.ghost_pos = clamp(state.ghost_pos, (0, 0), (WIDTH, HEIGHT))

        dist = length(state.ghost_pos - state.player)
        rate = 0 if dist > 200 else (200 - dist) / 200
        if rate > 0 and last_beep + (1 - min(0.8, rate)) * 0.2 < rl.get_time():
            last_beep = rl.get_time()
            rl.set_sound_pitch(beep, 1 + max(0, rate - 0.5))
            rl.play_sound(beep)

        update_time = (time.time() - update_time) * 1000

        # draw
        draw_time = time.time()
        rl.begin_texture_mode(canvas)
        rl.begin_mode_2d(camera)
        rl.clear_background(rl.DARKGRAY)

        for x, y, tile in tiles_around(bg, ivec2(state.player // TILE_SIZE)):
            if tile == -1:
                continue
            rl.draw_texture_pro(tiles_tex,
                                rl.Rectangle(tile * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE),
                                rl.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                                rl.Vector2(),
                                0,
                                rl.WHITE)

        for x, y, tile in tiles_around(walls, ivec2(state.player // TILE_SIZE)):
            if tile == -1:
                continue
            rl.draw_texture_pro(tiles_tex,
                                rl.Rectangle(tile * TILE_SIZE, 0, TILE_SIZE, TILE_SIZE),
                                rl.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                                rl.Vector2(),
                                0,
                                rl.WHITE)

        #for wall in walls:
        #    rl.draw_rectangle_rec(wall, rl.WHITE)

        for rec in wall_recs.near(state.player):
            rl.draw_rectangle_lines_ex(rec, 1, rl.RED)

        rl.draw_rectangle(int(state.player.x), int(state.player.y), PLAYER_SIZE, PLAYER_SIZE, rl.BLUE)
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
            print(rl.is_sound_playing(howl))
            if not rl.is_sound_playing(howl):
                print('play')
                rl.play_sound(howl)
                rl.set_sound_volume(howl, clamp((500 - length(state.player - state.ghost_pos)) / 500, 0.1, 1))

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
        rl.draw_fps(10, 10)
        rl.draw_text(f"update: {update_time:0.8f}", 10, 30, 20, rl.WHITE)
        rl.draw_text(f"draw: {draw_time:0.8f}", 10, 50, 20, rl.WHITE)
        rl.draw_text(f"wait_time: {wait_time:0.8f}", 10, 70, 20, rl.WHITE)
        wait_time = time.time()
        rl.end_drawing()
        wait_time = (time.time() - wait_time) * 1000
finally:
    rl.close_audio_device()
