from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Tuple


@dataclass
class State:
    node: str
    g: int
    f: int
    parent: Optional["State"]


def read_input(file_path: str) -> Tuple[str, str, Dict[str, List[Tuple[str, int]]], Dict[str, int]]:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    h: Dict[str, int] = {}
    graph: Dict[str, List[Tuple[str, int]]] = {}
    start: Optional[str] = None
    goal: Optional[str] = None
    mode: Optional[str] = None

    for line in lines:
        if line.startswith("START"):
            start = line.split()[1]
        elif line.startswith("GOAL"):
            goal = line.split()[1]
        elif line == "NODES":
            mode = "nodes"
        elif line == "EDGES":
            mode = "edges"
        elif mode == "nodes":
            node, val = line.split()
            h[node] = int(val)
        elif mode == "edges":
            u, v, w = line.split()
            graph.setdefault(u, []).append((v, int(w)))

    if start is None or goal is None:
        raise ValueError("Input thiếu START hoặc GOAL.")
    if start not in h or goal not in h:
        raise ValueError("START/GOAL chưa có trong danh sách NODES.")

    return start, goal, graph, h


def in_path(state: State, target: str) -> bool:
    current: Optional[State] = state
    while current is not None:
        if current.node == target:
            return True
        current = current.parent
    return False


def reconstruct_path(state: Optional[State]) -> List[str]:
    path: List[str] = []
    current = state
    while current is not None:
        path.append(current.node)
        current = current.parent
    path.reverse()
    return path


def format_state_list(states: List[State]) -> str:
    return ", ".join(f"{s.node}({s.f})" for s in states)


def can_stop_with_optimal(frontier: List[State], best_cost: float) -> bool:
    if best_cost == inf:
        return False
    return all(state.f >= best_cost for state in frontier)


def write_table(output_file: str, rows: List[Dict[str, str]], best_path: List[str], best_cost: float) -> None:
    columns = ["Bước", "TT", "TTK", "k(u,v)", "h(v)", "g(v)", "f(v)", "DS L1", "Danh sách L"]

    widths: Dict[str, int] = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(row.get(col, "")))

    def border_line() -> str:
        return "+" + "+".join("-" * (widths[col] + 2) for col in columns) + "+"

    def fmt_row(data: Dict[str, str]) -> str:
        return "| " + " | ".join(data.get(col, "").ljust(widths[col]) for col in columns) + " |"

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(border_line() + "\n")
        out.write(fmt_row({name: name for name in columns}) + "\n")
        out.write(border_line() + "\n")

        for row in rows:
            out.write(fmt_row(row) + "\n")
        out.write(border_line() + "\n")

        out.write("\n===== KẾT QUẢ =====\n")
        if best_path:
            out.write("Đường đi từ Trạng thái đầu => Trạng thái kết thúc: " + " -> ".join(best_path) + "\n")
            out.write(f"Chi phí: {int(best_cost)}\n")
        else:
            out.write("Không tìm thấy đường đi\n")


def branch_and_bound(start: str, goal: str, graph: Dict[str, List[Tuple[str, int]]], h: Dict[str, int], output_file: str) -> None:
    # Buoc 1: Khoi tao
    start_state = State(node=start, g=0, f=h[start], parent=None)
    L: List[State] = [start_state]
    best_cost = inf
    best_goal_state: Optional[State] = None
    rows: List[Dict[str, str]] = []
    step = 1

    # Buoc 2: Lap tim kiem
    while L:
        u_state = L.pop(0)

        # Buoc 3: Kiem tra trang thai dich
        if u_state.node == goal:
            if u_state.g <= best_cost:
                best_cost = u_state.g
                best_goal_state = u_state
                rows.append(
                    {
                        "Bước": str(step),
                        "TT": u_state.node,
                        "TTK": f"TTKT, tìm được đường đi chi phí {u_state.g}",
                        "Danh sách L": format_state_list(L),
                    }
                )

            if can_stop_with_optimal(L, best_cost):
                rows.append(
                    {
                        "Bước": str(step),
                        "TTK": "Dừng: đã tìm được đường đi tối ưu",
                        "Danh sách L": format_state_list(L),
                    }
                )
                break

            step += 1
            continue

        # Buoc 4: Sinh L1 va sap theo f tang dan
        child_records: List[Tuple[str, int, int, int, State]] = []
        for v, edge_cost in graph.get(u_state.node, []):
            if v not in h:
                raise ValueError(f"Thieu heuristic cho dinh {v}.")
            if in_path(u_state, v):
                continue

            g_v = u_state.g + edge_cost
            f_v = g_v + h[v]
            child_state = State(node=v, g=g_v, f=f_v, parent=u_state)
            child_records.append((v, edge_cost, h[v], g_v, child_state))

        sorted_child_records = sorted(child_records, key=lambda item: item[4].f)
        L1 = [item[4] for item in sorted_child_records]

        # Buoc 5: Chuyen L1 vao dau L
        L = L1 + L

        l1_str = format_state_list(L1)
        l_str = format_state_list(L)

        if not child_records:
            rows.append(
                {
                    "Bước": str(step),
                    "TT": u_state.node,
                    "TTK": "Không có trạng thái con hợp lệ",
                    "DS L1": "-",
                    "Danh sách L": l_str,
                }
            )
            step += 1
            continue

        for index, (v, edge_cost, h_v, g_v, child_state) in enumerate(child_records):
            rows.append(
                {
                    "Bước": str(step) if index == 0 else "",
                    "TT": u_state.node if index == 0 else "",
                    "TTK": v,
                    "k(u,v)": str(edge_cost),
                    "h(v)": str(h_v),
                    "g(v)": str(g_v),
                    "f(v)": str(child_state.f),
                    "DS L1": l1_str if index == 0 else "",
                    "Danh sách L": l_str if index == 0 else "",
                }
            )
        step += 1

    # Buoc 6: Truy vet va in ket qua
    best_path = reconstruct_path(best_goal_state)
    write_table(output_file, rows, best_path, best_cost)


def main() -> None:
    input_file = "input.txt"
    output_file = "output.txt"

    start, goal, graph, h = read_input(input_file)
    branch_and_bound(start, goal, graph, h, output_file)

    print(f"Da ghi ket qua ra file: {output_file}")


if __name__ == "__main__":
    main()