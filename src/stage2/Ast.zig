//! Abstract Syntax Tree for Zig source code.
//! For Zig syntax, the root node is at nodes[0] and contains the list of
//! sub-nodes.
//! For Zon syntax, the root node is at nodes[0] and contains lhs as the node
//! index of the main expression.

/// Reference to externally-owned data.
source: [:0]const u8,

tokens: std.zig.Ast.TokenList.Slice,
/// The root AST node is assumed to be index 0. Since there can be no
/// references to the root node, this means 0 is available to indicate null.
nodes: std.zig.Ast.NodeList.Slice,
extra_data: []std.zig.Ast.Node.Index,

errors: []const std.zig.Ast.Error,

pub fn deinit(tree: *Ast, gpa: Allocator) void {
    tree.tokens.deinit(gpa);
    tree.nodes.deinit(gpa);
    gpa.free(tree.extra_data);
    gpa.free(tree.errors);
    tree.* = undefined;
}

pub const Mode = enum { zig, zon };

/// Result should be freed with tree.deinit() when there are
/// no more references to any of the tokens or nodes.
pub fn parse(
    handle: *Handle,
    gpa: Allocator,
    source: [:0]const u8,
    mode: Mode,
    version: i32,
) Allocator.Error!Ast {
    var tokens = std.zig.Ast.TokenList{};
    defer tokens.deinit(gpa);

    // Empirically, the zig std lib has an 8:1 ratio of source bytes to token count.
    const estimated_token_count = source.len / 8;
    try tokens.ensureTotalCapacity(gpa, estimated_token_count);

    var check_interval: i32 = 350;
    std.log.debug("parsing: version: {}", .{version});
    var tokenizer = std.zig.Tokenizer.init(source);
    while (true) {
        const token = tokenizer.next();
        try tokens.append(gpa, .{
            .tag = token.tag,
            .start = @as(u32, @intCast(token.loc.start)),
        });
        if (token.tag == .eof) break;
        if (version != 0) {
            if (check_interval == 0) {
                check_interval = 350;
                handle.*.impl.lock.lock();
                defer handle.*.impl.lock.unlock();
                // std.log.debug("doing a ver check: {} vs {}", .{handle.*.version, version});
                if (handle.*.version != version) {
                    std.log.debug("cancelling a parse for version: {}", .{version});
                    return error.OutOfMemory; // XXX error.Outdated
                }
            }
            check_interval -= 1;
        }
    }

    var parser: Parse = .{
        .source = source,
        .gpa = gpa,
        .token_tags = tokens.items(.tag),
        .token_starts = tokens.items(.start),
        .errors = .{},
        .nodes = .{},
        .extra_data = .{},
        .scratch = .{},
        .tok_i = 0,
    };
    defer parser.errors.deinit(gpa);
    defer parser.nodes.deinit(gpa);
    defer parser.extra_data.deinit(gpa);
    defer parser.scratch.deinit(gpa);

    // Empirically, Zig source code has a 2:1 ratio of tokens to AST nodes.
    // Make sure at least 1 so we can use appendAssumeCapacity on the root node below.
    const estimated_node_count = (tokens.len + 2) / 2;
    try parser.nodes.ensureTotalCapacity(gpa, estimated_node_count);

    switch (mode) {
        .zig => try parser.parseRoot(),
        .zon => try parser.parseZon(),
    }

    // TODO experiment with compacting the MultiArrayList slices here
    return Ast{
        .source = source,
        .tokens = tokens.toOwnedSlice(),
        .nodes = parser.nodes.toOwnedSlice(),
        .extra_data = try parser.extra_data.toOwnedSlice(gpa),
        .errors = try parser.errors.toOwnedSlice(gpa),
    };
}

const std = @import("std");
const testing = std.testing;
const Ast = @This();
const Allocator = std.mem.Allocator;
const Parse = @import("Parse.zig");
const Handle = @import("../DocumentStore.zig").Handle;

test {
    testing.refAllDecls(@This());
}
