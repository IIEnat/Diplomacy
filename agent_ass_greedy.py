import random, networkx as nx
from agent_baselines import Agent

# ===== Opening book: fixed movement orders through end of 1902 =====
OPENING_BOOK = {
    ("ENGLAND","S1901M"): ["F LON - ENG","F EDI - NTH","A LVP - YOR"],
    ("ENGLAND","F1901M"): ["F NTH - NWY","F ENG - BEL","A YOR - EDI"],
    ("ENGLAND","S1902M"): ["F NWY H","F BEL - ENG","A EDI - NTH"],
    ("ENGLAND","F1902M"): ["F ENG - BEL","A NTH H","F NWY H"],

    ("FRANCE","S1901M"): ["A PAR - BUR","A MAR - SPA","F BRE - MAO"],
    ("FRANCE","F1901M"): ["F MAO - POR","A BUR - BEL","A SPA - GAS"],
    ("FRANCE","S1902M"): ["A GAS - SPA","A BEL H","F POR - MAO"],
    ("FRANCE","F1902M"): ["A BEL - HOL","F MAO - SPA","A SPA - MAR"],

    ("GERMANY","S1901M"): ["A MUN - RUH","A BER - KIE","F KIE - DEN"],
    ("GERMANY","F1901M"): ["A RUH - BEL","A KIE - HOL","F DEN - SWE"],
    ("GERMANY","S1902M"): ["A BEL H","A HOL - KIE","F SWE H"],
    ("GERMANY","F1902M"): ["A KIE - DEN","A BEL - HOL","F SWE H"],

    ("ITALY","S1901M"): ["A VEN - APU","A ROM - VEN","F NAP - ION"],
    ("ITALY","F1901M"): ["F ION C A APU - TUN","A APU - TUN VIA CONVOY","A VEN - PIE"],
    ("ITALY","S1902M"): ["F ION - EMS","A TUN H","A PIE - MAR"],
    ("ITALY","F1902M"): ["F EMS - AEG","A MAR H","A TUN H"],

    ("AUSTRIA","S1901M"): ["A VIE - GAL","A BUD - SER","F TRI - ALB"],
    ("AUSTRIA","F1901M"): ["A SER - GRE","F ALB S A SER - GRE","A GAL - RUM"],
    ("AUSTRIA","S1902M"): ["A GRE H","F ALB - ION","A RUM - BUL"],
    ("AUSTRIA","F1902M"): ["A BUL H","F ION - AEG","A GRE - SER"],

    ("RUSSIA","S1901M"): ["A MOS - UKR","A WAR - GAL","F SEV - BLA","F STP/SC - BOT"],
    ("RUSSIA","F1901M"): ["F BOT - SWE","A UKR S F BLA - RUM","F BLA - RUM","A GAL - WAR"],
    ("RUSSIA","S1902M"): ["F SWE H","A WAR - UKR","A RUM - SEV","F RUM H"],
    ("RUSSIA","F1902M"): ["F SWE - BAL","A SEV - UKR","A UKR - GAL","F RUM - SEV"],

    ("TURKEY","S1901M"): ["F ANK - BLA","A CON - BUL","A SMY - ARM"],
    ("TURKEY","F1901M"): ["A BUL - GRE","F BLA S A BUL - GRE","A ARM - SEV"],
    ("TURKEY","S1902M"): ["F BLA - RUM","A GRE - BUL","A SEV H"],
    ("TURKEY","F1902M"): ["F RUM - BLA","A BUL - CON","A SEV - UKR"],
}


# ---------- tiny utils ----------
_origin = lambda o: (o.split()+["",""])[1]
_norm = lambda s: s.split('/')[0] if s else s
_sanitize_move = lambda order: order.split(' VIA')[0] if order else order

# Book helpers
def _maybe_match(intended, poss):
    if intended in poss: return intended
    if " VIA CONVOY" in intended and intended.replace(" VIA CONVOY","") in poss:
        return intended.replace(" VIA CONVOY","")
    vc = intended+" VIA CONVOY" if (" - " in intended and intended.startswith("A ")) else None
    if vc and vc in poss: return vc
    n = " ".join(intended.split())
    for p in poss:
        if " ".join(p.split())==n: return p
    return None

def _hold_from(poss):
    for o in poss:
        if o.endswith(" H"): return o
    return poss[0] if poss else None
_norm = lambda s: s.split('/')[0] if s else s
_sanitize_move = lambda order: order.split(' VIA')[0] if order else order

def _is_adj(game, utype, a, b):
    a, b = _norm(a), _norm(b)
    return game.map.abuts(utype, a, '-', b)

# ---------- agent ----------
class StudentAgent(Agent):
    """
    Depth‑1 "greedy+" with deliberate supports, legality checks, and season awareness. Movement phase only.
    Adds *coordination mode* for early game: all units converge on a single focus
    target to create supported, coherent advances.
    """
    def __init__(self, agent_name='Greedy+ Agent'):
        super().__init__(agent_name)
        self.map_graph_army = None
        self.map_graph_navy = None
        self.COORD_YEARS = 1902  # coordinate through 1901–1902

    # ----- init / graphs -----
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self._build_graphs()

    def _build_graphs(self):
        if not self.game:
            raise Exception('Game Not Initialised.')
        A, N = nx.Graph(), nx.Graph()
        raw = list(self.game.map.loc_type.keys())
        for k in raw:
            t = self.game.map.loc_type[k]; u = k.upper()
            if t in ('LAND','COAST'): A.add_node(u)
            if t in ('WATER','COAST'): N.add_node(u)
        locs = [k.upper() for k in raw]
        for i in locs:
            for j in locs:
                if self.game.map.abuts('A', i, '-', j): A.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j): N.add_edge(i, j)
        self.map_graph_army, self.map_graph_navy = A, N

    # ----- helpers -----
    def _season(self):
        tag = getattr(self.game, 'get_current_phase', lambda: '')()
        return 'F' if (tag and tag[0] == 'F') else 'S'

    def _year(self):
        tag = getattr(self.game, 'get_current_phase', lambda: '')()
        try:
            return int(tag[1:5])
        except Exception:
            return None

    def _paths(self, utype, src):
        srcU = src.upper()
        G = self.map_graph_navy if utype=='F' else self.map_graph_army
        try:
            return nx.shortest_path(G, source=srcU)
        except Exception:
            return {}

    def _pick_target(self, utype, src, centers_mine, centers_other):
        paths = self._paths(utype, src)
        if not paths:
            return None
        WN, WE, WD = 3.0, 2.0, 0.8
        bestT, bestScore = None, -1e9
        for t in centers_other:
            tU = t.upper()
            if tU not in paths:
                continue
            dist = len(paths[tU]) - 1
            base = (WN if t not in centers_mine else WE)
            score = base - WD*dist
            if score > bestScore:
                bestScore, bestT = score, tU
        return bestT

    def _first_hop(self, utype, src, target):
        if not target:
            return None
        paths = self._paths(utype, src)
        if target not in paths or len(paths[target]) < 2:
            return None
        return paths[target][1]

    def _legal_supporters(self, mover_loc, dest, all_poss, my_units):
        sup = []
        for loc in my_units:
            if loc == mover_loc:
                continue
            poss = all_poss.get(loc, [])
            utype = None
            for o in poss:
                if o and o[0] in 'AF':
                    utype = o[0]
                    break
            if not utype:
                continue
            if _is_adj(self.game, utype, loc.upper(), dest.upper()):
                sup.append((loc, utype))
        return sup

    # ----- main API -----
    def get_actions(self):
        all_poss = self.game.get_all_possible_orders()
        orderable = self.game.get_orderable_locations(self.power_name)
        if self.game.phase_type != 'M':
            return []

        my_centers = set(self.game.get_centers(self.power_name))
        not_mine = [sc for sc in self.game.map.scs if sc not in my_centers]
        season = self._season()
        year = self._year()
        coord_mode = (year is not None and year <= self.COORD_YEARS)

        # Collect unit types
        units = []  # (loc, utype)
        for loc in orderable:
            poss = all_poss.get(loc, [])
            utype = None
            for o in poss:
                if o and o[0] in 'AF':
                    utype = o[0]
                    break
            if utype:
                units.append((loc, utype))

        # --- Opening book through 1902 (movement phases) ---
        tag = getattr(self.game, 'get_current_phase', lambda: '')()
        if year is not None and year <= 1902 and tag.endswith('M'):
            key = (self.power_name, tag)
            if key in OPENING_BOOK:
                used=set(); outs=[]
                for intended in OPENING_BOOK[key]:
                    origin = _origin(intended)
                    # find this unit's actual loc (handle coasts)
                    loc = origin if origin in orderable else next((c for c in orderable if c==origin or _norm(c)==_norm(origin)), None)
                    if loc and loc not in used:
                        poss = all_poss.get(loc, [])
                        m = _maybe_match(intended, poss)
                        if m:
                            outs.append(m); used.add(loc)
                # any remaining units: hold
                for loc in orderable:
                    if loc not in used:
                        poss = all_poss.get(loc, [])
                        h = _hold_from(poss)
                        if h: outs.append(h)
                return list(dict.fromkeys(outs))

        # --- Coordination focus target (single SC) ---
        focus = None
        if coord_mode:
            bestScore = 1e18
            for sc in not_mine:
                scU = sc.upper()
                total = 0
                reachable = False
                for loc, utype in units:
                    paths = self._paths(utype, loc)
                    if scU in paths:
                        total += (len(paths[scU]) - 1)
                        reachable = True
                    else:
                        total += 50  # large penalty
                if reachable and total < bestScore:
                    bestScore, focus = total, scU
        # fallback: if no focus or not coord mode, we allow per-unit targets but still try to buddy up

        # --- Build base orders (mover -> hop) ---
        orders = {}
        moves = []  # (loc, utype, dest)
        for loc, utype in units:
            poss = all_poss.get(loc, [])
            # In Fall, prefer to HOLD on owned SC to secure
            if season == 'F' and _norm(loc) in my_centers:
                hold = next((o for o in poss if o.endswith(' H')), (poss[0] if poss else None))
                if hold:
                    orders[loc] = hold
                continue
            # choose target
            if focus:
                tgt = focus
            else:
                tgt = self._pick_target(utype, loc, my_centers, not_mine)
            hop = self._first_hop(utype, loc, tgt)
            if hop:
                cand = None
                for o in poss:
                    p = o.split()
                    if len(p) >= 4 and p[0] == utype and p[1] == loc and p[2] == '-' and _norm(p[3]) == _norm(hop):
                        cand = o
                        break
                if cand:
                    orders[loc] = cand
                    moves.append((loc, utype, _norm(hop)))
                else:
                    hold = next((o for o in poss if o.endswith(' H')), (poss[0] if poss else None))
                    if hold:
                        orders[loc] = hold
            else:
                hold = next((o for o in poss if o.endswith(' H')), (poss[0] if poss else None))
                if hold:
                    orders[loc] = hold

        # --- Assign supports to the spearhead ---
        if moves:
            spear = moves[0]
            if focus:
                bestd = 1e9
                for loc, utype, hop in moves:
                    paths = self._paths(utype, hop)
                    d = len(paths[focus]) - 1 if (paths and focus in paths) else 999
                    if d < bestd:
                        bestd, spear = d, (loc, utype, hop)
            spear_loc, spear_type, spear_dest = spear
            supporters = self._legal_supporters(spear_loc, spear_dest, all_poss, [u for u, _ in units])
            supporters = [s for s in supporters if s[0] != spear_loc]
            supporters.sort(key=lambda t: (_norm(t[0]) in my_centers, t[0]))
            for s_loc, s_type in supporters[:2]:
                if s_loc in orders:
                    spear_move = _sanitize_move(orders[spear_loc])
                    orders[s_loc] = f"{s_type} {s_loc} S {spear_move}"

        return [orders[loc] for loc, _ in units if loc in orders and orders[loc]]

