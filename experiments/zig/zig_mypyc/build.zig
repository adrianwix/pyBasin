const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "zigsolve",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/solver_lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(lib);

    const build_step = b.step("lib", "Build the shared library");
    build_step.dependOn(b.getInstallStep());
}
