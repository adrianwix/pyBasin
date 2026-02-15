const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build a shared library (.so) instead of an executable.
    // This is what Python loads via ctypes.CDLL().
    const lib = b.addLibrary(.{
        .name = "zigsolve",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/solver_lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Link libc so that std.Thread.spawn uses pthread_create instead of
    // raw clone(). Without this, Zig-spawned threads lack glibc TLS
    // initialization, causing crashes when calling into Cython/libm code
    // (e.g. sin()) that accesses thread-local state.
    lib.root_module.link_libc = true;

    b.installArtifact(lib);

    const build_step = b.step("lib", "Build the shared library");
    build_step.dependOn(b.getInstallStep());
}
