from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import z3

from faultless.ir import BasicBlock, Function, If, LoopOp, AddressMapping, AddressSet, Join, Write, SymbolicExpression

Z3ExpressionT = SymbolicExpression | z3.BoolRef | int | bool
def visualize(object: Function | AddressMapping | Z3ExpressionT, show_successors_in_blocks: bool = False):
    if isinstance(object, Function):
        print(visualize_function(object, show_successors_in_blocks))
    elif isinstance(object, AddressMapping):
        print(visualize_memory(object, show_successors_in_blocks))
    elif isinstance(object, Z3ExpressionT):
        print(visualize_z3expr(object))
    else:
        raise TypeError(f"No support for visualizing objects of type {type(object)}")

@dataclass(frozen=True)
class BlockBox:
    lines: List[str]
    width: int
    height: int


@dataclass(frozen=True)
class EdgeComponent:
    x: int
    top: int
    bottom: int

class VisualizationError(Exception):
    """A problem occured when generating the visualization."""


def visualize_function(fn: Function, show_successors_in_blocks: bool = False) -> str:
    blocks = list(fn.basic_blocks)
    if not blocks:
        return ""

    def build_box(block: BasicBlock) -> BlockBox:
        text_lines = [f"BB{block.id}"] + [str(ins) for ins in block]
        if show_successors_in_blocks and block.successors:
            succ_desc = "-> (" + ", ".join(f"BB{succ.id}" for succ in block.successors) + ")"
            text_lines.append(succ_desc)
        if len(text_lines) % 2 == 0:
            text_lines.append("") # adds an extra line so the block has a midpoint and we don't get UNSAT for trying to center things on the midpoint.
        inner_width = max((len(line) for line in text_lines), default=0)
        if inner_width % 2 == 0:
            inner_width += 1
        width = inner_width + 4
        height = len(text_lines) + 2
        return BlockBox(lines=text_lines, width=width, height=height)

    boxes: Dict[BasicBlock, BlockBox] = {block: build_box(block) for block in blocks}

    def find_back_edges(nodes: Iterable[BasicBlock]) -> List[Tuple[BasicBlock, BasicBlock]]:
        visited: set[BasicBlock] = set()
        stack: set[BasicBlock] = set()
        back_edges: List[Tuple[BasicBlock, BasicBlock]] = []

        def dfs(node: BasicBlock):
            visited.add(node)
            stack.add(node)
            for succ in node.successors:
                if succ not in visited:
                    dfs(succ)
                elif succ in stack:
                    back_edges.append((node, succ))
            stack.remove(node)

        for node in nodes:
            if node not in visited:
                dfs(node)
        return back_edges

    back_edges = find_back_edges(blocks)
    back_edge_set = {(tail, head) for tail, head in back_edges}

    def natural_loop(head: BasicBlock, tail: BasicBlock) -> set[BasicBlock]:
        loop_nodes = {head, tail}
        work = [tail]
        while work:
            node = work.pop()
            for pred in node.predecessors:
                if pred not in loop_nodes:
                    loop_nodes.add(pred)
                    work.append(pred)
        return loop_nodes

    loops_by_head: Dict[BasicBlock, set[BasicBlock]] = {}
    tails_by_head: Dict[BasicBlock, List[BasicBlock]] = {}
    for tail, head in back_edges:
        loop_nodes = natural_loop(head, tail)
        loops_by_head.setdefault(head, set()).update(loop_nodes)
        tails_by_head.setdefault(head, []).append(tail)

    loop_heads = set(loops_by_head.keys())
    loop_false_succ: Dict[BasicBlock, BasicBlock] = {}
    loop_body_succ: Dict[BasicBlock, BasicBlock] = {}
    for head in loop_heads:
        if len(head.successors) == 2:
            succ0, succ1 = head.successors
            loop_nodes = loops_by_head[head]
            if succ0 in loop_nodes and succ1 not in loop_nodes:
                loop_body_succ[head] = succ0
                loop_false_succ[head] = succ1
            elif succ1 in loop_nodes and succ0 not in loop_nodes:
                loop_body_succ[head] = succ1
                loop_false_succ[head] = succ0

    # Predecessor sets without loop back edges (transitive closure).
    direct_predecessors: Dict[BasicBlock, set[BasicBlock]] = {
        block: set() for block in blocks
    }
    for pred in blocks:
        for succ in pred.successors:
            if (pred, succ) not in back_edge_set:
                direct_predecessors[succ].add(pred)

    non_backedge_predecessors: Dict[BasicBlock, set[BasicBlock]] = {
        block: set() for block in blocks
    }
    for block in blocks:
        seen: set[BasicBlock] = set()
        queue: List[BasicBlock] = list(direct_predecessors[block])
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            non_backedge_predecessors[block].add(node)
            for pred in direct_predecessors[node]:
                if pred not in seen:
                    queue.append(pred)

    def block_ends_with_loopop(block: BasicBlock) -> bool:
        if not block.instructions:
            return False
        return isinstance(block.instructions[-1].op, LoopOp)

    def primary_exit_block(head: BasicBlock) -> BasicBlock | None:
        if block_ends_with_loopop(head):
            return head
        seen: set[BasicBlock] = set()
        queue: List[BasicBlock] = [head]
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if node is not head and block_ends_with_loopop(node):
                return node
            for succ in node.successors:
                if succ not in seen:
                    queue.append(succ)
        return None

    def exempt_loop_body_blocks(head: BasicBlock) -> set[BasicBlock]:
        primary_exit = primary_exit_block(head)
        if primary_exit is None:
            return set()
        exempt: set[BasicBlock] = {primary_exit}
        seen: set[BasicBlock] = set()
        queue: List[BasicBlock] = [head]
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if node is primary_exit:
                continue
            exempt.add(node)
            for succ in node.successors:
                if succ not in seen:
                    queue.append(succ)
        return exempt

    solver = z3.Optimize()

    x_vars: Dict[BasicBlock, z3.IntNumRef] = {
        block: z3.Int(f"bb_{block.id}_x") for block in blocks
    }
    y_vars: Dict[BasicBlock, z3.IntNumRef] = {
        block: z3.Int(f"bb_{block.id}_y") for block in blocks
    }
    cx_vars: Dict[BasicBlock, z3.IntNumRef] = {
        block: z3.Int(f"bb_{block.id}_cx") for block in blocks
    }
    cy_vars: Dict[BasicBlock, z3.IntNumRef] = {
        block: z3.Int(f"bb_{block.id}_cy") for block in blocks
    }

    for block in blocks:
        solver.add(x_vars[block] >= 0)
        solver.add(y_vars[block] >= 0)
        width = boxes[block].width
        height = boxes[block].height
        solver.add(2 * cx_vars[block] == 2 * x_vars[block] + width - 1)
        solver.add(2 * cy_vars[block] == 2 * y_vars[block] + height - 1)

    def z3_min(values: List[z3.ArithRef]) -> z3.ArithRef:
        result = values[0]
        for value in values[1:]:
            result = z3.If(value < result, value, result)
        return result

    def z3_max(values: List[z3.ArithRef]) -> z3.ArithRef:
        result = values[0]
        for value in values[1:]:
            result = z3.If(value > result, value, result)
        return result

    def left(block: BasicBlock) -> z3.ArithRef:
        return x_vars[block]

    def right(block: BasicBlock) -> z3.ArithRef:
        return x_vars[block] + boxes[block].width - 1

    def top(block: BasicBlock) -> z3.ArithRef:
        return y_vars[block]

    def bottom(block: BasicBlock) -> z3.ArithRef:
        return y_vars[block] + boxes[block].height - 1

    # Boxes are non-overlapping.
    for i, block_a in enumerate(blocks):
        for block_b in blocks[i + 1:]:
            solver.add(
                z3.Or(
                    right(block_a) < left(block_b),
                    right(block_b) < left(block_a),
                    bottom(block_a) < top(block_b),
                    bottom(block_b) < top(block_a),
                )
            )

    # The padding between the boxes.
    h_pad = 3
    v_pad = 5
    for i, block_a in enumerate(blocks):
        for block_b in blocks[i + 1:]:
            solver.add(
                z3.Or(
                    right(block_a) + h_pad <= left(block_b),
                    right(block_b) + h_pad <= left(block_a),
                    bottom(block_a) + v_pad <= top(block_b),
                    bottom(block_b) + v_pad <= top(block_a),
                )
            )

    def is_indent_if(block: BasicBlock) -> bool:
        return len(block.successors) == 2 and block.successors[0] in non_backedge_predecessors[block.successors[1]]

    # The box at the false branch is to the left of the box of the true branch (except for the false branch of loops).
    for block in blocks:
        if len(block.successors) == 2 and block not in loop_heads:
            if primary_exit_block(block) == block:
                continue
            assert not block_ends_with_loopop(block)
            true_succ, false_succ = block.successors
            if (block, true_succ) in back_edge_set or (block, false_succ) in back_edge_set:
                continue
            if not is_indent_if(block): # This interferes with the indent-if constraint pattern 
                solver.add(right(false_succ) < left(true_succ))

    # The midpoint on the top/bottom edge of a box should be the midpoint between the leftmost edge of the leftmost
    # successor and the rightmost edge of the rightmost successor (excluding back edges).
    for block in blocks:
        if primary_exit_block(block) == block:
            continue
        successors = [
            succ
            for succ in block.successors
            if (block, succ) not in back_edge_set and len(succ.predecessors) == 1
        ]
        if is_indent_if(block):
            # This is the case of
            # if (...) {
            #     // ...
            # }
            # // ...
            # We want the if body to be indented and the post-loop block to be vertically aligned with the if block.
            true_succ, false_succ = block.successors
            solver.add(left(block) == left(false_succ))
            false_succ_width = boxes[false_succ].width
            diff = non_backedge_predecessors[false_succ] - non_backedge_predecessors[block]
            diff.remove(block) # a block can't be indented relative to itself, and we don't want the if-block to be indented anyway.
            for indented_block in diff:
                # Indent the width of the post-loop block so there is always room to route an edge to it.
                solver.add(left(block) + false_succ_width <= left(indented_block))
        elif successors:
            leftmost = z3_min([left(succ) for succ in successors])
            rightmost = z3_max([right(succ) for succ in successors])
            solver.add(left(block) + right(block) == leftmost + rightmost)
                      

    # Successors should be below their predecessors, except for loop back edges.
    VERTICAL_SEP = 5
    for block in blocks:
        for succ in block.successors:
            if (block, succ) in back_edge_set:
                continue
            solver.add(bottom(block) + VERTICAL_SEP <= top(succ))

    # Prefer successors of If blocks to share the same y coordinate when possible.
    for block in blocks:
        if len(block.successors) != 2 or not block.instructions:
            continue
        if isinstance(block.instructions[-1].op, If):
            left_succ, right_succ = block.successors
            solver.add_soft(y_vars[left_succ] == y_vars[right_succ])

    # Successors that leave a loop should be placed after the loop.
    for head, loop_nodes in loops_by_head.items():
        max_loop_bottom = z3_max([bottom(node) for node in loop_nodes])
        for node in loop_nodes:
            for succ in node.successors:
                if succ in loop_nodes:
                    continue
                solver.add(top(succ) > max_loop_bottom)

    # Loop body blocks should be to the right of the loop head.
    LOOP_BODY_RIGHT_PAD = 5
    for head, loop_nodes in loops_by_head.items():
        exempt_nodes = exempt_loop_body_blocks(head)
        primary_exit = primary_exit_block(head)
        if primary_exit is not None:
            solver.add(left(primary_exit) == left(head))
            if len(primary_exit.successors) >= 2:
                exit_succ = primary_exit.successors[1]
                assert exit_succ not in loop_nodes, (
                    "Primary exit block false successor must be outside the loop."
                )
                solver.add(left(exit_succ) == left(primary_exit))
        for node in loop_nodes:
            if node == head:
                continue
            if node in exempt_nodes:
                continue
            solver.add(left(node) >= left(head) + LOOP_BODY_RIGHT_PAD)

    back_edge_x: Dict[BasicBlock, z3.IntNumRef] = {
        head: z3.Int(f"back_edge_{head.id}_x") for head in loop_heads
    }
    back_edge_top: Dict[BasicBlock, z3.IntNumRef] = {
        head: z3.Int(f"back_edge_{head.id}_top") for head in loop_heads
    }
    back_edge_bottom: Dict[BasicBlock, z3.IntNumRef] = {
        head: z3.Int(f"back_edge_{head.id}_bottom") for head in loop_heads
    }

    # Back edge vertical components.
    for head, tails in tails_by_head.items():
        solver.add(back_edge_x[head] >= 0)
        solver.add(back_edge_top[head] == cy_vars[head])
        max_tail_center = z3_max([cy_vars[tail] for tail in tails])
        solver.add(back_edge_bottom[head] == max_tail_center)
        solver.add(back_edge_top[head] != back_edge_bottom[head])
        loop_nodes = loops_by_head[head]
        rightmost = z3_max([right(node) for node in loop_nodes])
        solver.add(back_edge_x[head] >= rightmost + 3)

    # Nested loops: inner back edges are left of outer back edges with spacing.
    NESTED_BACKEDGE_PAD = 3
    loop_heads_list = list(loop_heads)
    for inner in loop_heads_list:
        for outer in loop_heads_list:
            if inner == outer:
                continue
            if inner in loops_by_head.get(outer, set()):
                solver.add(back_edge_x[inner] + NESTED_BACKEDGE_PAD <= back_edge_x[outer])

    max_right = z3.Int("max_right")
    max_bottom = z3.Int("max_bottom")
    rights: List[z3.ArithRef] = [right(block) for block in blocks]
    bottoms: List[z3.ArithRef] = [bottom(block) for block in blocks]
    for head in loop_heads:
        rights.append(back_edge_x[head])
        bottoms.append(back_edge_bottom[head])
    solver.add(max_right >= z3_max(rights))
    solver.add(max_bottom >= z3_max(bottoms))
    solver.minimize(max_right)
    solver.minimize(max_bottom)

    if solver.check() != z3.sat:
        raise VisualizationError("Failed to solve layout constraints.")

    model = solver.model()

    def eval_int(expr: z3.ArithRef) -> int:
        value = model.eval(expr, model_completion=True)
        return int(value.as_long())

    block_pos: Dict[BasicBlock, Tuple[int, int]] = {
        block: (eval_int(x_vars[block]), eval_int(y_vars[block])) for block in blocks
    }

    back_edge_components: Dict[BasicBlock, EdgeComponent] = {}
    for head in loop_heads:
        back_edge_components[head] = EdgeComponent(
            x=eval_int(back_edge_x[head]),
            top=eval_int(back_edge_top[head]),
            bottom=eval_int(back_edge_bottom[head]),
        )

    width = max(
        ([block_pos[block][0] + boxes[block].width for block in blocks]
         + [component.x + 1 for component in back_edge_components.values()]),
        default=1,
    )
    height = max(
        ([block_pos[block][1] + boxes[block].height for block in blocks]
         + [component.bottom + 1 for component in back_edge_components.values()]),
        default=1,
    )
    canvas: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]

    box_interior: set[Tuple[int, int]] = set()

    def place_char(x: int, y: int, ch: str):
        if x < 0 or y < 0 or y >= height or x >= width:
            return
        existing = canvas[y][x]
        if existing == " " or existing == ch:
            canvas[y][x] = ch
            return
        if existing in ("-", "|") and ch in ("-", "|") and existing != ch:
            canvas[y][x] = "+"
            return
        if existing == "+":
            return
        canvas[y][x] = ch

    def place_edge(x: int, y: int, ch: str):
        if (x, y) in box_interior:
            return
        place_char(x, y, ch)

    def draw_vline(x: int, y1: int, y2: int, ch: str):
        if y1 > y2:
            y1, y2 = y2, y1
        for y in range(y1, y2 + 1):
            place_edge(x, y, ch)

    for block in blocks:
        x, y = block_pos[block]
        box = boxes[block]
        inner_width = box.width - 4
        border = "+" + "-" * (box.width - 2) + "+"
        for yy in range(y + 1, y + box.height - 1):
            for xx in range(x + 1, x + box.width - 1):
                box_interior.add((xx, yy))
        for idx, ch in enumerate(border):
            place_char(x + idx, y, ch)
            place_char(x + idx, y + box.height - 1, ch)
        for row_idx, line in enumerate(box.lines):
            content = "| " + line.ljust(inner_width) + " |"
            for col_idx, ch in enumerate(content):
                place_char(x + col_idx, y + 1 + row_idx, ch)

    def dump_canvas() -> str:
        return "\n".join("".join(row).rstrip() for row in canvas)

    def replace_with_plus(x: int, y: int, allowed: set[str]):
        existing = canvas[y][x]
        if existing not in allowed:
            raise VisualizationError(
                f"Edge path blocked at ({x}, {y}) by '{existing}'.\n{dump_canvas()}"
            )
        canvas[y][x] = "+"

    def draw_vertical_edge(src: BasicBlock, dst: BasicBlock):
        src_x, src_y = block_pos[src]
        dst_x, dst_y = block_pos[dst]
        src_box = boxes[src]
        dst_box = boxes[dst]
        edge_x = src_x + 2
        start_y = src_y + src_box.height - 1
        if not (dst_x <= edge_x <= dst_x + dst_box.width - 1):
            raise VisualizationError("Vertical edge does not align with successor box.")
        replace_with_plus(edge_x, start_y, {"+", "-"})
        for y in range(start_y + 1, dst_y):
            if canvas[y][edge_x] != " ":
                raise VisualizationError(
                    f"Edge path blocked at ({edge_x}, {y}) by '{canvas[y][edge_x]}'.\n{dump_canvas()}"
                )
            canvas[y][edge_x] = "|"
        replace_with_plus(edge_x, dst_y, {"+", "-"})

    drawn_edges: set[tuple[BasicBlock, BasicBlock]] = set()

    for head in loop_heads:
        primary_exit = primary_exit_block(head)
        if primary_exit is None:
            continue
        if len(primary_exit.successors) >= 2:
            post_loop = primary_exit.successors[1]
            edge_key = (primary_exit, post_loop)
            if edge_key not in drawn_edges:
                draw_vertical_edge(primary_exit, post_loop)
                drawn_edges.add(edge_key)

    for block in blocks:
        if len(block.successors) != 2:
            continue
        true_succ, false_succ = block.successors
        if true_succ not in non_backedge_predecessors[false_succ]:
            continue
        edge_key = (block, false_succ)
        if edge_key not in drawn_edges:
            draw_vertical_edge(block, false_succ)
            drawn_edges.add(edge_key)

    def draw_hline_checked(y: int, x1: int, x2: int):
        if x1 > x2:
            return
        for x in range(x1, x2 + 1):
            if canvas[y][x] != " ":
                raise VisualizationError(
                    f"Edge path blocked at ({x}, {y}) by '{canvas[y][x]}'.\n{dump_canvas()}"
                )
            canvas[y][x] = "-"

    def draw_vline_checked(x: int, y1: int, y2: int):
        if y1 > y2:
            return
        for y in range(y1, y2 + 1):
            if canvas[y][x] != " ":
                raise VisualizationError(
                    f"Edge path blocked at ({x}, {y}) by '{canvas[y][x]}'.\n{dump_canvas()}"
                )
            canvas[y][x] = "|"

    def draw_vline_over(y1: int, y2: int, x: int):
        if y1 > y2:
            return
        for y in range(y1, y2 + 1):
            existing = canvas[y][x]
            if existing == " ":
                canvas[y][x] = "|"
            elif existing in ("|", "+", "-"):
                continue
            else:
                raise VisualizationError(
                    f"Edge path blocked at ({x}, {y}) by '{existing}'.\n{dump_canvas()}"
                )

    def row_is_open(y: int, x1: int, x2: int) -> bool:
        if x1 > x2:
            x1, x2 = x2, x1
        if y < 0 or y >= height:
            return False
        for x in range(x1, x2 + 1):
            if canvas[y][x] not in (" ", "|"):
                return False
        return True

    def direct_vertical_candidate(src: BasicBlock, dst: BasicBlock) -> int | None:
        src_x, src_y = block_pos[src]
        dst_x, dst_y = block_pos[dst]
        src_box = boxes[src]
        dst_box = boxes[dst]
        src_min = src_x + 2
        src_max = src_x + src_box.width - 3
        dst_min = dst_x + 2
        dst_max = dst_x + dst_box.width - 3
        overlap_min = max(src_min, dst_min)
        overlap_max = min(src_max, dst_max)
        if overlap_min > overlap_max:
            return None
        if len(src.successors) == 2:
            true_succ, false_succ = src.successors
            src_mid = src_x + src_box.width // 2
            if dst == false_succ:
                overlap_max = min(overlap_max, src_mid - 1)
            elif dst == true_succ:
                overlap_min = max(overlap_min, src_mid + 1)
            if overlap_min > overlap_max:
                return None
        return (overlap_min + overlap_max) // 2

    def draw_direct_vertical(src: BasicBlock, dst: BasicBlock, x: int):
        src_x, src_y = block_pos[src]
        dst_x, dst_y = block_pos[dst]
        src_box = boxes[src]
        dst_box = boxes[dst]
        start_y = src_y + src_box.height - 1
        end_y = dst_y
        replace_with_plus(x, start_y, {"+", "-"})
        for y in range(start_y + 1, end_y):
            existing = canvas[y][x]
            if existing == " ":
                canvas[y][x] = "|"
            elif existing in ("|", "+", "-"):
                continue
            else:
                raise VisualizationError(f"Edge path blocked at ({x}, {y}) by '{existing}'.")
        replace_with_plus(x, end_y, {"+", "-"})

    for head, component in back_edge_components.items():
        draw_vline(component.x, component.top, component.bottom, "|")

        for tail in tails_by_head.get(head, []):
            tail_x, tail_y = block_pos[tail]
            tail_box = boxes[tail]
            tail_right = tail_x + tail_box.width - 1
            tail_mid_y = tail_y + tail_box.height // 2
            back_edge_is_true = tail.successors and tail.successors[0] == head
            direct_ok = component.top <= tail_mid_y <= component.bottom
            if direct_ok:
                for x in range(tail_right + 1, component.x):
                    if canvas[tail_mid_y][x] != " ":
                        direct_ok = False
                        break
            if direct_ok:
                replace_with_plus(tail_right, tail_mid_y, {"+", "|"})
                draw_hline_checked(tail_mid_y, tail_right + 1, component.x - 1)
                replace_with_plus(component.x, tail_mid_y, {"|", "+"})
                continue

            if back_edge_is_true:
                start_x = tail_x + tail_box.width - 3
            else:
                start_x = tail_x + 2
            start_y = tail_y + tail_box.height - 1
            replace_with_plus(start_x, start_y, {"+", "-"})
            route_y = None
            for y in range(start_y + 2, height):
                if canvas[y][start_x] != " ":
                    continue
                blocked = False
                for x in range(start_x + 1, component.x):
                    if canvas[y][x] != " ":
                        blocked = True
                        break
                if not blocked:
                    route_y = y
                    break
            if route_y is None:
                raise VisualizationError("No empty row available for back edge routing.")
            draw_vline_checked(start_x, start_y + 1, route_y)
            replace_with_plus(start_x, route_y, {"|", "+"})
            draw_hline_checked(route_y, start_x + 1, component.x - 1)
            if route_y > component.bottom:
                draw_vline_checked(component.x, component.bottom + 1, route_y)
                component = EdgeComponent(component.x, component.top, route_y)
                back_edge_components[head] = component
            replace_with_plus(component.x, route_y, {"|", "+"})

        head_x, head_y = block_pos[head]
        head_box = boxes[head]
        head_right = head_x + head_box.width - 1
        replace_with_plus(component.x, component.top, {"|", "+"})
        if component.x <= head_right:
            raise VisualizationError("Back edge component must be to the right of its loop head.")
        for x in range(head_right + 1, component.x):
            if canvas[component.top][x] != " ":
                raise VisualizationError(f"Back edge path blocked at ({x}, {component.top}).")
            canvas[component.top][x] = "-"
        replace_with_plus(head_right, component.top, {"+", "-", "|"})

    direct_edge_for_succ: Dict[BasicBlock, int] = {}
    preferred_row_for_succ: Dict[BasicBlock, int] = {}

    for block in blocks:
        for succ in block.successors:
            if (block, succ) in back_edge_set:
                continue
            if (block, succ) in drawn_edges:
                continue
            direct_x = direct_vertical_candidate(block, succ)
            if direct_x is None:
                continue
            draw_direct_vertical(block, succ, direct_x)
            drawn_edges.add((block, succ))
            direct_edge_for_succ[succ] = direct_x

    for block in blocks:
        for succ in block.successors:
            if (block, succ) in back_edge_set:
                continue
            if (block, succ) in drawn_edges:
                continue
            start_x = None
            if len(block.successors) == 2:
                true_succ, false_succ = block.successors
                if succ == false_succ:
                    start_x = block_pos[block][0] + 2
                elif succ == true_succ:
                    start_x = block_pos[block][0] + boxes[block].width - 3
            if start_x is None:
                start_x = block_pos[block][0] + boxes[block].width // 2

            start_y = block_pos[block][1] + boxes[block].height - 1
            replace_with_plus(start_x, start_y, {"+", "-"})
            first_y = start_y + 1
            if first_y >= height:
                raise VisualizationError("Edge path leaves canvas.")
            if canvas[first_y][start_x] not in (" ", "|"):
                if canvas[first_y][start_x] != "-":
                    raise VisualizationError(
                        f"Edge path blocked at ({start_x}, {first_y}) by '{canvas[first_y][start_x]}'.\n{dump_canvas()}"
                    )
            if canvas[first_y][start_x] == " ":
                canvas[first_y][start_x] = "|"

            succ_x, succ_y = block_pos[succ]
            succ_center = succ_x + boxes[succ].width // 2
            target_x = direct_edge_for_succ.get(succ, succ_center)
            row = preferred_row_for_succ.get(succ)
            if row is not None:
                if row <= first_y or row >= succ_y:
                    row = None
                else:
                    for x in range(min(start_x, target_x), max(start_x, target_x) + 1):
                        if canvas[row][x] not in (" ", "|", "-", "+"):
                            row = None
                            break
            if row is None:
                for y in range(first_y + 1, succ_y):
                    if row_is_open(y, start_x, target_x):
                        row = y
                        break
            if row is None:
                raise VisualizationError("No open row available for edge routing.")

            draw_vline_over(first_y + 1, row - 1, start_x)
            replace_with_plus(start_x, row, {" ", "|", "+"})
            merged = False
            row_shared = preferred_row_for_succ.get(succ) == row
            if start_x < target_x:
                for x in range(start_x + 1, target_x):
                    existing = canvas[row][x]
                    if existing not in (" ", "|", "-", "+"):
                        raise VisualizationError(
                            f"Edge path blocked at ({x}, {row}) by '{existing}'.\n{dump_canvas()}"
                        )
                    if row_shared and existing in ("|", "-", "+"):
                        canvas[row][x] = "+"
                        merged = True
                        break
                    if existing == " ":
                        canvas[row][x] = "-"
                if not merged and canvas[row][target_x] in ("|", "+") and succ in direct_edge_for_succ:
                    replace_with_plus(target_x, row, {"|", "+"})
                    merged = True
            elif start_x > target_x:
                for x in range(start_x - 1, target_x, -1):
                    existing = canvas[row][x]
                    if existing not in (" ", "|", "-", "+"):
                        raise VisualizationError(
                            f"Edge path blocked at ({x}, {row}) by '{existing}'.\n{dump_canvas()}"
                        )
                    if row_shared and existing in ("|", "-", "+"):
                        canvas[row][x] = "+"
                        merged = True
                        break
                    if existing == " ":
                        canvas[row][x] = "-"
                if not merged and canvas[row][target_x] in ("|", "+") and succ in direct_edge_for_succ:
                    replace_with_plus(target_x, row, {"|", "+"})
                    merged = True
            if merged:
                drawn_edges.add((block, succ))
                preferred_row_for_succ.setdefault(succ, row)
                continue

            replace_with_plus(target_x, row, {" ", "|", "+"})
            draw_vline_over(row + 1, succ_y - 1, target_x)
            replace_with_plus(target_x, succ_y, {"+", "-"})
            drawn_edges.add((block, succ))
            preferred_row_for_succ.setdefault(succ, row)

    return "\n".join("".join(row).rstrip() for row in canvas)


def visualize_memory(address_mapping: AddressMapping, show_successors_in_nodes: bool = False) -> str:
    lines: List[str] = []

    def build_nodes(root: AddressSet) -> Tuple[List[AddressSet], Dict[AddressSet, List[AddressSet]]]:
        order: List[AddressSet] = []
        edges: Dict[AddressSet, List[AddressSet]] = {}
        seen: set[AddressSet] = set()

        def walk(node: AddressSet):
            if node in seen:
                return
            seen.add(node)
            order.append(node)
            children: List[AddressSet] = []
            if isinstance(node, Join):
                children = [node.true, node.false]
            elif isinstance(node, Write) and node.history is not None:
                children = [node.history]
            edges[node] = children
            for child in children:
                walk(child)

        walk(root)
        return order, edges

    def build_box(node: AddressSet, node_id: int, children: List[AddressSet]) -> BlockBox:
        def add_multiline(text_lines: List[str], label: str, value: object) -> None:
            raw_lines = str(value).splitlines() or [""]
            prefix = f"{label}: "
            text_lines.append(prefix + raw_lines[0])
            indent = " " * len(prefix)
            for line in raw_lines[1:]:
                text_lines.append(indent + line)

        if isinstance(node, Write):
            text_lines = [f"Write #{node_id}"]
            add_multiline(text_lines, "offset", node.offset)
            add_multiline(text_lines, "value", node.value)
        elif isinstance(node, Join):
            text_lines = [f"Join #{node_id}"]
            add_multiline(text_lines, "cond", node.condition)
        else:
            text_lines = [f"{node.__class__.__name__} #{node_id}"]
        if show_successors_in_nodes and children:
            succ_desc = "-> (" + ", ".join(f"#{node_ids[child]}" for child in children) + ")"
            text_lines.append(succ_desc)
        if len(text_lines) % 2 == 0:
            text_lines.append("")
        inner_width = max((len(line) for line in text_lines), default=0)
        if inner_width % 2 == 0:
            inner_width += 1
        width = inner_width + 4
        height = len(text_lines) + 2
        return BlockBox(lines=text_lines, width=width, height=height)

    def z3_min(values: List[z3.ArithRef]) -> z3.ArithRef:
        result = values[0]
        for value in values[1:]:
            result = z3.If(value < result, value, result)
        return result

    def z3_max(values: List[z3.ArithRef]) -> z3.ArithRef:
        result = values[0]
        for value in values[1:]:
            result = z3.If(value > result, value, result)
        return result

    for base_address, root in address_mapping.mapping.items():
        lines.append(f"Base {base_address}:")
        if root is None:
            lines.append("`--[empty]")
            lines.append("")
            continue

        order, edges = build_nodes(root)
        node_ids = {node: idx + 1 for idx, node in enumerate(order)}
        boxes = {node: build_box(node, node_ids[node], edges.get(node, [])) for node in order}

        solver = z3.Optimize()
        x_vars = {node: z3.Int(f"mem_{node_ids[node]}_x") for node in order}
        y_vars = {node: z3.Int(f"mem_{node_ids[node]}_y") for node in order}

        for node in order:
            solver.add(x_vars[node] >= 0)
            solver.add(y_vars[node] >= 0)

        def left(node: AddressSet) -> z3.ArithRef:
            return x_vars[node]

        def right(node: AddressSet) -> z3.ArithRef:
            return x_vars[node] + boxes[node].width - 1

        def top(node: AddressSet) -> z3.ArithRef:
            return y_vars[node]

        def bottom(node: AddressSet) -> z3.ArithRef:
            return y_vars[node] + boxes[node].height - 1

        # Boxes are non-overlapping.
        for i, node_a in enumerate(order):
            for node_b in order[i + 1:]:
                solver.add(
                    z3.Or(
                        right(node_a) < left(node_b),
                        right(node_b) < left(node_a),
                        bottom(node_a) < top(node_b),
                        bottom(node_b) < top(node_a),
                    )
                )

        # Successors should be below their predecessors.
        VERTICAL_SEP = 5
        for parent, children in edges.items():
            for child in children:
                solver.add(bottom(parent) + VERTICAL_SEP <= top(child))

        # Non-successor blocks keep a vertical sep of 5.
        for i, node_a in enumerate(order):
            for node_b in order[i + 1:]:
                if node_b in edges.get(node_a, []) or node_a in edges.get(node_b, []):
                    continue
                solver.add(
                    z3.Or(
                        bottom(node_a) + VERTICAL_SEP <= top(node_b),
                        bottom(node_b) + VERTICAL_SEP <= top(node_a),
                        top(node_a) == top(node_b),
                    )
                )

        # Join false branch is to the left of the true branch.
        for node in order:
            if isinstance(node, Join):
                solver.add(right(node.false) < left(node.true))

        # Join children centered around their parent.
        for node in order:
            if isinstance(node, Join):
                solver.add(left(node) + right(node) == left(node.false) + right(node.true))

        # Write nodes centered under their predecessors.
        predecessors: Dict[AddressSet, List[AddressSet]] = {node: [] for node in order}
        for parent, children in edges.items():
            for child in children:
                predecessors.setdefault(child, []).append(parent)
        for node in order:
            if isinstance(node, Write) and predecessors.get(node):
                preds = predecessors[node]
                if all(isinstance(pred, Write) for pred in preds):
                    leftmost = z3_min([left(pred) for pred in preds])
                    rightmost = z3_max([right(pred) for pred in preds])
                    solver.add(left(node) + right(node) == leftmost + rightmost)

        max_right = z3.Int("mem_max_right")
        max_bottom = z3.Int("mem_max_bottom")
        solver.add(max_right >= z3_max([right(n) for n in order]))
        solver.add(max_bottom >= z3_max([bottom(n) for n in order]))
        solver.minimize(max_right)
        solver.minimize(max_bottom)

        if solver.check() != z3.sat:
            raise VisualizationError("Failed to solve memory layout constraints.")

        model = solver.model()

        def eval_int(expr: z3.ArithRef) -> int:
            value = model.eval(expr, model_completion=True)
            return int(value.as_long())

        pos = {node: (eval_int(x_vars[node]), eval_int(y_vars[node])) for node in order}
        width = max((pos[node][0] + boxes[node].width for node in order), default=1)
        height = max((pos[node][1] + boxes[node].height for node in order), default=1)
        canvas: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]
        occupancy: List[List[object | None]] = [[None for _ in range(width)] for _ in range(height)]

        @dataclass(frozen=True)
        class Edge:
            parent: AddressSet
            child: AddressSet

        def place_char(x: int, y: int, ch: str):
            if x < 0 or y < 0 or y >= height or x >= width:
                return
            existing = canvas[y][x]
            if existing == " " or existing == ch:
                canvas[y][x] = ch
                return
            if existing == "+":
                return
            canvas[y][x] = ch

        def mark_box(x: int, y: int, box: BlockBox):
            if 0 <= x < width and 0 <= y < height:
                occupancy[y][x] = box

        def mark_edge(x: int, y: int, edge: "Edge"):
            if 0 <= x < width and 0 <= y < height:
                occupancy[y][x] = edge

        def place_edge(x: int, y: int, ch: str, edge: "Edge"):
            if isinstance(occupancy[y][x], BlockBox):
                return
            existing = canvas[y][x]
            if existing == " " or existing == ch:
                canvas[y][x] = ch
                mark_edge(x, y, edge)
                return
            if existing == "+":
                return
            if ch == "-" and existing == "|":
                canvas[y][x] = "-"
                mark_edge(x, y, edge)
                return
            if ch == "|" and existing == "-":
                return
            canvas[y][x] = ch
            mark_edge(x, y, edge)

        def place_corner(x: int, y: int, edge: "Edge"):
            existing = occupancy[y][x]
            if isinstance(existing, BlockBox):
                place_char(x, y, "+")
                return
            if isinstance(existing, Edge) and existing.child is not edge.child:
                return
            place_edge(x, y, "+", edge)

        for node in order:
            x, y = pos[node]
            box = boxes[node]
            inner_width = box.width - 4
            border = "+" + "-" * (box.width - 2) + "+"
            for yy in range(y, y + box.height):
                for xx in range(x, x + box.width):
                    mark_box(xx, yy, box)
            for idx, ch in enumerate(border):
                place_char(x + idx, y, ch)
                place_char(x + idx, y + box.height - 1, ch)
            for row_idx, line in enumerate(box.lines):
                content = "| " + line.ljust(inner_width) + " |"
                for col_idx, ch in enumerate(content):
                    place_char(x + col_idx, y + 1 + row_idx, ch)

        def column_clear(x: int, y1: int, y2: int) -> bool:
            if y1 > y2:
                return True
            for y in range(y1, y2 + 1):
                if isinstance(occupancy[y][x], BlockBox):
                    return False
            return True

        def find_clear_column(x_min: int, x_max: int, y1: int, y2: int) -> int | None:
            if x_min > x_max:
                return None
            for x in range(x_min, x_max + 1):
                if column_clear(x, y1, y2):
                    return x
            return None

        def find_clear_column_near(
            x_min: int, x_max: int, y1: int, y2: int, target: int
        ) -> int | None:
            if x_min > x_max:
                return None
            best_x = None
            best_dist = None
            for x in range(x_min, x_max + 1):
                if not column_clear(x, y1, y2):
                    continue
                dist = abs(x - target)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_x = x
            return best_x

        def parent_edge_range(parent: AddressSet, child: AddressSet) -> tuple[int, int]:
            px, _ = pos[parent]
            pbox = boxes[parent]
            min_x = px + 2
            max_x = px + pbox.width - 3
            if isinstance(parent, Join):
                mid = px + pbox.width // 2
                if child is parent.false:
                    max_x = min(max_x, mid - 1)
                elif child is parent.true:
                    min_x = max(min_x, mid + 1)
            return min_x, max_x

        def try_draw_direct_edge(parent: AddressSet, child: AddressSet) -> bool:
            px, py = pos[parent]
            cx, cy = pos[child]
            pbox = boxes[parent]
            cbox = boxes[child]
            parent_min, parent_max = parent_edge_range(parent, child)
            child_min = cx + 2
            child_max = cx + cbox.width - 3
            overlap_min = max(parent_min, child_min)
            overlap_max = min(parent_max, child_max)
            if overlap_min > overlap_max:
                return False
            x = (overlap_min + overlap_max) // 2
            start_y = py + pbox.height - 1
            end_y = cy
            if end_y <= start_y:
                return False
            edge = Edge(parent=parent, child=child)
            clear_x = find_clear_column_near(
                overlap_min, overlap_max, start_y + 1, end_y - 1, x
            )
            if clear_x is None:
                print("Warning: direct edge passes through a box.")
            else:
                x = clear_x
            place_corner(x, start_y, edge)
            for y in range(start_y + 1, end_y):
                if canvas[y][x] not in (" ", "|"):
                    return False
                place_edge(x, y, "|", edge)
            place_corner(x, end_y, edge)
            return True

        drawn_edges: set[Tuple[AddressSet, AddressSet]] = set()
        for parent, children in edges.items():
            for child in children:
                if try_draw_direct_edge(parent, child):
                    drawn_edges.add((parent, child))

        def row_is_open(y: int, x1: int, x2: int) -> bool:
            if x1 > x2:
                x1, x2 = x2, x1
            if y < 0 or y >= height:
                return False
            for x in range(x1, x2 + 1):
                if isinstance(occupancy[y][x], BlockBox):
                    return False
                if canvas[y][x] not in (" ", "|"):
                    return False
            return True

        for parent, children in edges.items():
            for child in children:
                if (parent, child) in drawn_edges:
                    continue
                px, py = pos[parent]
                cx, cy = pos[child]
                pbox = boxes[parent]
                cbox = boxes[child]
                parent_min, parent_max = parent_edge_range(parent, child)
                if parent_min <= parent_max:
                    start_x = (parent_min + parent_max) // 2
                else:
                    start_x = px + pbox.width // 2
                start_y = py + pbox.height - 1
                end_y = cy
                if end_y <= start_y:
                    continue
                edge = Edge(parent=parent, child=child)
                place_corner(start_x, start_y, edge)
                first_y = start_y + 1
                if canvas[first_y][start_x] == " ":
                    place_edge(start_x, first_y, "|", edge)
                row = None
                child_min = cx + 2
                child_max = cx + cbox.width - 3
                end_x = cx + cbox.width // 2
                for y in range(first_y + 1, end_y):
                    if row_is_open(y, start_x, end_x) and column_clear(end_x, y + 1, end_y - 1):
                        row = y
                        break
                    alt_x = find_clear_column(child_min, child_max, y + 1, end_y - 1)
                    if alt_x is not None and row_is_open(y, start_x, alt_x):
                        end_x = alt_x
                        row = y
                        break
                if row is None:
                    raise VisualizationError("No open row available for memory edge routing.")
                for y in range(first_y + 1, row):
                    if canvas[y][start_x] == " ":
                        place_edge(start_x, y, "|", edge)
                place_corner(start_x, row, edge)
                if start_x < end_x:
                    merged = False
                    for x in range(start_x + 1, end_x):
                        if canvas[row][x] == "|":
                            existing = occupancy[row][x]
                            if isinstance(existing, Edge) and existing.child is child:
                                place_edge(x, row, "+", edge)
                                merged = True
                                break
                            place_edge(x, row, "-", edge)
                        elif canvas[row][x] == " ":
                            place_edge(x, row, "-", edge)
                elif start_x > end_x:
                    merged = False
                    for x in range(start_x - 1, end_x, -1):
                        if canvas[row][x] == "|":
                            existing = occupancy[row][x]
                            if isinstance(existing, Edge) and existing.child is child:
                                place_edge(x, row, "+", edge)
                                merged = True
                                break
                            place_edge(x, row, "-", edge)
                        elif canvas[row][x] == " ":
                            place_edge(x, row, "-", edge)
                if merged:
                    drawn_edges.add((parent, child))
                    continue
                place_corner(end_x, row, edge)
                for y in range(row + 1, end_y):
                    if canvas[y][end_x] == " ":
                        place_edge(end_x, y, "|", edge)
                place_corner(end_x, end_y, edge)
                drawn_edges.add((parent, child))

        lines.extend("".join(row).rstrip() for row in canvas)
        lines.append("")
        # print("\n".join(lines).rstrip())

    return "\n".join(lines).rstrip()


def visualize_z3expr(expr: SymbolicExpression | z3.BoolRef | int | bool) -> str:
    class _TreeNode:
        def __init__(self, label_lines: List[str], children: List["_TreeNode"], is_if: bool = False):
            self.label_lines = label_lines
            self.children = children
            self.is_if = is_if
            self.box = _build_box(label_lines)
            self.subtree_width = 0
            self.subtree_height = 0
            self.x = 0
            self.y = 0

    def _build_box(lines: List[str]) -> BlockBox:
        inner_width = max((len(line) for line in lines), default=0)
        if inner_width % 2 == 0:
            inner_width += 1
        width = inner_width + 4
        height = len(lines) + 2
        return BlockBox(lines=lines, width=width, height=height)

    def _is_ite(e: z3.ExprRef) -> bool:
        return z3.is_app_of(e, z3.Z3_OP_ITE)

    def _collect_ites(e: z3.ExprRef) -> List[z3.ExprRef]:
        ites: List[z3.ExprRef] = []

        def walk(node: z3.ExprRef):
            if _is_ite(node):
                ites.append(node)
                return
            for i in range(node.num_args()):
                child = node.arg(i)
                if isinstance(child, z3.ExprRef):
                    walk(child)

        walk(e)
        return ites

    def _replace_first(text: str, old: str, new: str) -> str:
        idx = text.find(old)
        if idx == -1:
            return text
        return text[:idx] + new + text[idx + len(old):]

    def _label_lines(text: str) -> List[str]:
        lines = text.splitlines()
        return lines if lines else [text]

    def _build_tree(e) -> _TreeNode:
        if isinstance(e, z3.ExprRef) and _is_ite(e):
            cond = e.arg(0)
            then_branch = e.arg(1)
            else_branch = e.arg(2)
            label_lines = _label_lines(str(cond))
            label_lines = ["If"] + label_lines
            return _TreeNode(label_lines, [_build_tree(else_branch), _build_tree(then_branch)], is_if=True)
        if isinstance(e, z3.ExprRef):
            ites = _collect_ites(e)
            label = str(e)
            children = []
            for idx, ite in enumerate(ites, start=1):
                placeholder = f"<child{idx}>"
                label = _replace_first(label, str(ite), placeholder)
                children.append(_build_tree(ite))
            return _TreeNode(_label_lines(label), children)
        return _TreeNode(_label_lines(str(e)), [])

    def _layout(node: _TreeNode, hpad: int = 2, vpad: int = 3) -> None:
        if not node.children:
            node.subtree_width = node.box.width
            node.subtree_height = node.box.height
            return
        for child in node.children:
            _layout(child, hpad, vpad)
        total_children_width = sum(child.subtree_width for child in node.children) + hpad * (len(node.children) - 1)
        node.subtree_width = max(node.box.width, total_children_width)
        node.subtree_height = node.box.height + vpad + max(child.subtree_height for child in node.children)

        start_x = (node.subtree_width - total_children_width) // 2
        current_x = start_x
        child_y = node.box.height + vpad
        for child in node.children:
            child.x = current_x
            child.y = child_y
            current_x += child.subtree_width + hpad

    def _draw_box(canvas: List[List[str]], x: int, y: int, box: BlockBox, is_if: bool) -> None:
        border = "+" + "-" * (box.width - 2) + "+"
        for idx, ch in enumerate(border):
            canvas[y][x + idx] = ch
            canvas[y + box.height - 1][x + idx] = ch
        for yy in range(y + 1, y + box.height - 1):
            canvas[yy][x] = "|"
            canvas[yy][x + box.width - 1] = "|"
        for row_idx, line in enumerate(box.lines):
            padded = line.center(box.width - 4)
            row = f"| {padded} |"
            for idx, ch in enumerate(row):
                canvas[y + 1 + row_idx][x + idx] = ch

    def _draw_edge(canvas: List[List[str]], px: int, py: int, cx: int, cy: int, parent_box_x: int, parent_box_width: int, child_box_x: int, child_box_width: int) -> None:
        def place(x: int, y: int, ch: str) -> None:
            if 0 <= y < len(canvas) and 0 <= x < len(canvas[y]):
                if canvas[y][x] == "+" and ch != "+":
                    return
                canvas[y][x] = ch

        start_y = py + 1
        end_y = cy - 1
        if start_y > end_y:
            return
        parent_left = parent_box_x + 2
        parent_right = parent_box_x + parent_box_width - 3
        child_left = child_box_x + 2
        child_right = child_box_x + child_box_width - 3
        overlap_left = max(parent_left, child_left)
        overlap_right = min(parent_right, child_right)
        use_vertical = overlap_left <= overlap_right and (overlap_right - overlap_left) >= 1
        if use_vertical:
            x = (overlap_left + overlap_right) // 2
            place(x, py, "+")
            for yy in range(start_y, end_y + 1):
                place(x, yy, "|")
            place(x, cy, "+")
            return
        place(px, start_y, "|")
        mid_y = start_y + 1
        if mid_y > end_y:
            return
        place(px, mid_y, "+")
        if px < cx:
            for xx in range(px + 1, cx):
                place(xx, mid_y, "-")
        elif cx < px:
            for xx in range(cx + 1, px):
                place(xx, mid_y, "-")
        place(cx, mid_y, "+")
        for yy in range(mid_y + 1, end_y + 1):
            place(cx, yy, "|")
        place(cx, cy, "+")

    def _render(node: _TreeNode, canvas: List[List[str]], ox: int, oy: int) -> None:
        box_x = ox + (node.subtree_width - node.box.width) // 2
        box_y = oy
        _draw_box(canvas, box_x, box_y, node.box, node.is_if)
        parent_center_x = box_x + node.box.width // 2
        parent_bottom_y = box_y + node.box.height - 1
        if node.children:
            has_vertical = False
            parent_left = box_x + 2
            parent_right = box_x + node.box.width - 3
            for child in node.children:
                child_box_x = ox + child.x + (child.subtree_width - child.box.width) // 2
                child_left = child_box_x + 2
                child_right = child_box_x + child.box.width - 3
                overlap_left = max(parent_left, child_left)
                overlap_right = min(parent_right, child_right)
                if overlap_left <= overlap_right and (overlap_right - overlap_left) >= 1:
                    has_vertical = True
                    break
            if has_vertical:
                canvas[parent_bottom_y][parent_center_x] = "+"
        for child in node.children:
            child_box_x = ox + child.x + (child.subtree_width - child.box.width) // 2
            child_box_y = oy + child.y
            child_center_x = child_box_x + child.box.width // 2
            child_top_y = child_box_y
            _draw_edge(canvas, parent_center_x, parent_bottom_y, child_center_x, child_top_y, box_x, node.box.width, child_box_x, child.box.width)
            _render(child, canvas, ox + child.x, oy + child.y)
            if 0 <= child_top_y < len(canvas):
                canvas[child_top_y][child_center_x] = "+"

    root = _build_tree(expr)
    _layout(root)
    width = root.subtree_width
    height = root.subtree_height
    canvas = [[" " for _ in range(width)] for _ in range(height)]
    _render(root, canvas, 0, 0)
    return "\n".join("".join(row).rstrip() for row in canvas).rstrip()
