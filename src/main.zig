const std = @import("std");
const rl = @import("common.zig").rl;

//pub usingnamespace @cImport({
//    @cDefine("RAYGUI_IMPLEMENTATION", "1");
//    @cInclude("raygui.h");
//});

extern fn init() callconv(.C) void;
extern fn update() callconv(.C) void;
extern fn cleanup() callconv(.C) void;

pub fn main() !void {
    init();

    while (!rl.WindowShouldClose()) {
        update();
    }

    cleanup();
}
