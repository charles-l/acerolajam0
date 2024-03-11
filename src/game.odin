package main
import "core:runtime"
import rl "raylib"
import "core:math/linalg"
import c "core:c"
import "core:mem"
import "core:strings"
import "core:encoding/json"

IS_WASM :: ODIN_ARCH == .wasm32 || ODIN_ARCH == .wasm64p32

@export
_fltused: c.int = 0

mainArena: mem.Arena
mainData: [mem.Megabyte * 20]byte
temp_allocator: mem.Scratch_Allocator

camera: rl.Camera2D
pos := rl.Vector3{0, 10, 0}
color := rl.RED
sp: Spring

ctx: runtime.Context

font: rl.Font

TILE_SIZE :: 16
SCALE: i32 = 4
WIDTH: i32 = 300 * SCALE
HEIGHT: i32 = 200 * SCALE

last_dir := rl.Vector2{1, 0}
light_offset := rl.Vector2{}
hero_id: EntityID
ghost_vac_id: EntityID

CELL_SIZE :: 128
SpatialHash :: struct {
    rects: []rl.Rectangle,
    spatial: map[[2]i16][dynamic]int,
}

SpatialIterator :: struct {
    sp: ^SpatialHash,
    start_bucket: [2]i16,
    di: int,
    bi: int,
}

spatial_query_near :: proc(s: ^SpatialHash, v: rl.Vector2) -> SpatialIterator {
    return SpatialIterator{
        sp = s,
        start_bucket = [2]i16{i16(v.x / CELL_SIZE), i16(v.y / CELL_SIZE)},
    }
}

iterate_spatial :: proc (it: ^SpatialIterator) -> (rl.Rectangle, bool) {
    dirs := [?][2]i16{
        { -1, -1 }, { 0, -1 }, { 1, -1 },
        { -1, 0 }, { 0, 0 }, { 1, 0 },
        { -1, 1 }, { 0, 1 }, { 1, 1 }
    }
    for ; it.di < len(dirs); it.di += 1 {
        bucket_key := it.start_bucket + dirs[it.di]
        if bucket_key in it.sp.spatial {
            bucket := it.sp.spatial[bucket_key]
            for it.bi < len(bucket) {
                r := it.sp.rects[bucket[it.bi]]
                it.bi += 1
                return r, true
            }
            it.bi = 0
        }
    }
    return rl.Rectangle{}, false
}

spatial_hash_build :: proc(rects: []rl.Rectangle) -> SpatialHash {
    s := SpatialHash{rects=rects}
    for r, i in rects {
        key := [2]i16{i16(r.x / CELL_SIZE), i16(r.y / CELL_SIZE)}
        if !(key in s.spatial) {
            s.spatial[key] = make([dynamic]int)
        }
        append(&s.spatial[key], i)
    }
    return s
}




Map :: struct {
    name: string,
    bg_tiles: [][]u8,
    wall_tiles: [][]u8,
    static_colliders: SpatialHash,
    cable: [4]rl.Vector2
}

stations := [3]Map{}

textures := struct {
    tiles: rl.Texture,
    hero: rl.Texture,
    ghostvac: rl.Texture,
}{}

textures2: map[string]rl.Texture

import "core:fmt"
load_map :: proc(path: cstring) -> Map {
    bytes_read: u32 = ---
    data := rl.LoadFileData(path, &bytes_read)
    json_data, err := json.parse(data[:bytes_read])
    if err != .None {
        rl.TraceLog(rl.TraceLogLevel.ERROR, "Failed to load map file")
        return Map{}
    }

    defer rl.UnloadFileData(data)
    base := json_data.(json.Object)
    m := Map{}
    m.name = base["values"].(json.Object)["StationName"].(json.String)

    rect_from_json :: proc(obj: json.Object) -> rl.Rectangle {
        return rl.Rectangle{
            f32(obj["x"].(json.Float)),
            f32(obj["y"].(json.Float)),
            16,
            16
        }
    }

    spatial_hash_recs: [dynamic]rl.Rectangle
    for layer in base["layers"].(json.Array) {
        layer_o := layer.(json.Object)
        if layer_o["name"].(json.String) == "Entities" {
            for j_entity in layer_o["entities"].(json.Array) {
                o_entity := j_entity.(json.Object)
                name := o_entity["name"].(json.String)
                switch name {
                case "hero":
                    hero_id = create_entity()
                    aabb := add_component(rl.Rectangle, hero_id)
                    aabb^ = rect_from_json(o_entity);
                    sprite := add_component(Sprite, hero_id)
                    sprite.tex = &textures.hero
                    sprite.source_rect = rl.Rectangle{0, 0, 16, 16}
                    //tweens := add_component(Tweens, cube)
                case "ghostvac":
                    ghost_vac_id = create_entity()
                    aabb := add_component(rl.Rectangle, ghost_vac_id)
                    aabb^ = rect_from_json(o_entity);
                    sprite := add_component(Sprite, ghost_vac_id)
                    sprite.tex = &textures.ghostvac
                    sprite.source_rect = rl.Rectangle{0, 0, 16, 16}
                    //tweens := add_component(Tweens, cube)
                case "turnstile":
                    t_id := create_entity()
                    aabb := add_component(rl.Rectangle, t_id)
                    aabb^ = rect_from_json(o_entity);
                    //tweens := add_component(Tweens, cube)
                case:
                    cstr, err := strings.clone_to_cstring(name, allocator=context.temp_allocator)
                    assert(err == nil)
                    rl.TraceLog(rl.TraceLogLevel.WARNING, "Unhandled case %s", cstr)
                }
            }
        }

        if layer_o["name"].(json.String) == "Decals" {
            for j_decal in layer_o["decals"].(json.Array) {
                o_decal := j_decal.(json.Object)
                texture_name := o_decal["texture"].(json.String)

                if !(texture_name in textures2) {
                    r := strings.concatenate([]string{"resources/", texture_name}, allocator=context.temp_allocator)
                    cstr, err := strings.clone_to_cstring(r, allocator=context.temp_allocator)
                    textures2[texture_name] = rl.LoadTexture(cstr)
                }
                texture := &textures2[texture_name]

                decal_id := create_entity()
                sprite := add_component(Sprite, decal_id)
                aabb := add_component(rl.Rectangle, decal_id)

                aabb^ = rect_from_json(o_decal);
                aabb.width = f32(texture.width)
                aabb.height = f32(texture.height)
                sprite.tex = texture
                sprite.source_rect = rl.Rectangle{0, 0, f32(texture.width), f32(texture.height)}

                //tweens := add_component(Tweens, cube)
            }
        }

        read_data2d :: proc(data2d: json.Array, spatial_hash_recs: ^[dynamic]rl.Rectangle) -> [][]u8 {
            layer_grid := make([][]u8, len(data2d))
            for row, y in data2d {
                row_out := make([]u8, len(row.(json.Array)))
                for j, x in row.(json.Array) {
                    i := j.(json.Float)
                    row_out[x] = u8(i)
                    if spatial_hash_recs != nil && i != -1 {
                        append(spatial_hash_recs, rl.Rectangle{
                            f32(x) * TILE_SIZE,
                            f32(y) * TILE_SIZE,
                            TILE_SIZE,
                            TILE_SIZE,
                        })
                    }
                }
                layer_grid[y] = row_out
            }
            return layer_grid
        }

        if layer_o["name"].(json.String) == "BGTiles" {
            data2d := layer_o["data2D"].(json.Array)
            m.bg_tiles = read_data2d(data2d, nil)
        }

        if layer_o["name"].(json.String) == "WallTiles" {
            data2d := layer_o["data2D"].(json.Array)
            m.wall_tiles = read_data2d(data2d, &spatial_hash_recs)
        }
    }

    assert(len(m.bg_tiles) == len(m.wall_tiles))
    assert(len(m.bg_tiles[0]) == len(m.wall_tiles[0]))

    m.static_colliders = spatial_hash_build(spatial_hash_recs[:])

    return m
}

@export
init :: proc "c" () {
    using rl
    context = runtime.default_context()
    // needed to setup some runtime type information in odin
    #force_no_inline runtime._startup_runtime()

    when IS_WASM {
        mem.arena_init(&mainArena, mainData[:])
        context.allocator = mem.arena_allocator(&mainArena)

        mem.scratch_allocator_init(&temp_allocator, mem.Megabyte * 2)
        context.temp_allocator = mem.scratch_allocator(&temp_allocator)

        TraceLog(rl.TraceLogLevel.INFO, "Setup hardcoded arena allocators")
    }
    ctx = context

    init_ecs()

    camera.offset = Vector2{f32(WIDTH / SCALE / 2), f32(HEIGHT / SCALE / 2)};
    camera.target = Vector2{};
    camera.rotation = 0;
    camera.zoom = 1
    InitWindow(WIDTH, HEIGHT, "Aberration Station")
    textures.tiles = rl.LoadTexture("resources/tiles.png")
    textures.hero = rl.LoadTexture("resources/hero.png")
    textures.ghostvac = rl.LoadTexture("resources/ghostvac.png")

    InitAudioDevice()
    SetTargetFPS(60)

    t := tween(&pos, rl.Vector3{0, 0, 0}, 3)
    t.ease_proc = ease_out_elastic
    tween(&color, rl.Color{255, 255, 0, 255}, 3)

    sp = make_spring(2, 0.5, 0, 0)

    textures2 = make(map[string]rl.Texture)

    stations[0] = load_map("resources/station1.json")

    font = rl.LoadFont("resources/mago3.ttf")
    rl.gui_font(font)
    DEFAULT: i32 = 0
    TEXT_SIZE: i32 = 16
    rl.gui_set_style(DEFAULT, TEXT_SIZE, 24)
}

@export
cleanup :: proc "c" () {
    context = ctx
    free_ecs()
    rl.CloseAudioDevice()
    rl.CloseWindow()
    #force_no_inline runtime._cleanup_runtime()
}

draw_text :: proc (s: string, pos: rl.Vector2) {
    cstr, err := strings.clone_to_cstring(s, allocator=context.temp_allocator)
    if err != nil {
        panic("can't alloc string")
    }

    rl.DrawTextEx(font, cstr, pos, 26, 1, rl.WHITE)
}

entity_pos :: proc(e: EntityID) -> rl.Vector2 {
    r := get_component(rl.Rectangle, e)
    return rl.Vector2{r.x, r.y}
}

resolve_static_collisions :: proc(it: SpatialIterator, actor: ^rl.Rectangle) {
    for 0..<3 {
    }
}

i := 0
@export
update :: proc "c" () {
    using rl
    context = ctx
    defer free_all(context.temp_allocator)
    BeginDrawing()
    defer EndDrawing()

    { // tween system
        view := make_scene_view(Tweens)
        for e in iterate_scene_view(&view) {
            update_tweeners(&get_component(Tweens, e)^.tweeners, rl.GetFrameTime());
        }
    }

    input := rl.Vector2{}
    { // keyboard input
        if rl.IsKeyDown(rl.KeyboardKey.DOWN) do input.y += 1
        if rl.IsKeyDown(rl.KeyboardKey.UP) do input.y -= 1
        if rl.IsKeyDown(rl.KeyboardKey.LEFT) do input.x -= 1
        if rl.IsKeyDown(rl.KeyboardKey.RIGHT) do input.x += 1

        l := linalg.length(input)
        if l > 1 {
            input = input / l
        }

        input *= 3
    }

    r := get_component(rl.Rectangle, hero_id)
    r.x += input.x
    r.y += input.y

    ClearBackground(BLACK)
    BeginMode2D(camera)
    {
        { // tiles
            bg_tiles := stations[0].bg_tiles
            wall_tiles := stations[0].wall_tiles
            rows := len(bg_tiles)
            cols := len(bg_tiles[0])
            maybe_draw_tile :: proc(grid: [][]u8, x: int, y: int) {
                if grid[y][x] != 0xFF {
                    rl.DrawTexturePro(textures.tiles,
                        rl.Rectangle {
                            f32(grid[y][x]) * f32(TILE_SIZE),
                            0,
                            f32(TILE_SIZE),
                            f32(TILE_SIZE)
                        },
                        rl.Rectangle{
                            f32(x) * f32(TILE_SIZE),
                            f32(y) * f32(TILE_SIZE),
                            f32(TILE_SIZE),
                            f32(TILE_SIZE)
                        },
                        rl.Vector2{},
                        0,
                        rl.WHITE)
                }
            }
            for iy in 0..<rows {
                for ix in 0..<cols {
                    maybe_draw_tile(bg_tiles, ix, iy)
                    maybe_draw_tile(wall_tiles, ix, iy)
                }
            }
        }
        { // rects/sprites
            view := make_scene_view(rl.Rectangle)
            for e in iterate_scene_view(&view) {
                dest := get_component(rl.Rectangle, e)^
                if s, ok := get_component_safe(Sprite, e).?; ok {
                    DrawTexturePro(s.tex^, s.source_rect, dest, rl.Vector2{}, 0, rl.WHITE)
                } else {
                    DrawRectangleRec(dest, rl.RED)
                }
            }

            it := spatial_query_near(&stations[0].static_colliders, entity_pos(hero_id))
            for rect in iterate_spatial(&it) {
                //rl.DrawRectangleLinesEx(rect, 1, rl.RED)
            }
        }
    }
    EndMode2D()

    draw_text("HI!", rl.Vector2{10, 10})
}


