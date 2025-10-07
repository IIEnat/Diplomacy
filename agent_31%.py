import random
import networkx as nx
from agent_baselines import Agent

# ===== Opening book: fixed movement orders for 1901 only (movement phases) =====
OPENING_BOOK = {
    ("ENGLAND","S1901M"): ["F LON - ENG","F EDI - NTH","A LVP - YOR"],
    ("ENGLAND","F1901M"): ["F NTH - NWY","F ENG - BEL","A YOR - EDI"],

    ("FRANCE","S1901M"): ["A PAR - BUR","A MAR - SPA","F BRE - MAO"],
    ("FRANCE","F1901M"): ["F MAO - POR","A BUR - BEL","A SPA - GAS"],

    ("GERMANY","S1901M"): ["A MUN - RUH","A BER - KIE","F KIE - DEN"],
    ("GERMANY","F1901M"): ["A RUH - BEL","A KIE - HOL","F DEN - SWE"],

    ("ITALY","S1901M"): ["A VEN - APU","A ROM - VEN","F NAP - ION"],
    ("ITALY","F1901M"): ["F ION C A APU - TUN","A APU - TUN VIA CONVOY","A VEN - PIE"],

    ("AUSTRIA","S1901M"): ["A VIE - GAL","A BUD - SER","F TRI - ALB"],
    ("AUSTRIA","F1901M"): ["A SER - GRE","F ALB S A SER - GRE","A GAL - RUM"],

    ("RUSSIA","S1901M"): ["A MOS - UKR","A WAR - GAL","F SEV - BLA","F STP/SC - BOT"],
    ("RUSSIA","F1901M"): ["F BOT - SWE","A UKR S F BLA - RUM","F BLA - RUM","A GAL - WAR"],

    ("TURKEY","S1901M"): ["F ANK - BLA","A CON - BUL","A SMY - ARM"],
    ("TURKEY","F1901M"): ["A BUL - GRE","F BLA S A BUL - GRE","A ARM - SEV"],
}



"""
DumbBot-style agent (per DAIDE s0003: DumbBot Algorithm). Movement phase only.
Implements the key ideas from the spec:
  • Province valuation per unit type (A/F), using owner/strength heuristics.
  • Value diffusion ("board averaging") over adjacency for a few iterations.
  • One-ply tactical choice: move toward highest-value adjacent region; avoid
    vacating owned SCs in Fall; simple, legal support when two units target the
    same destination.

References:
  - DAIDE: DumbBot Algorithm (s0003).  
"""

# ---------- helpers ----------
_norm = lambda s: s.split('/')[0] if s else s

# book helpers
_origin = lambda o: (o.split()+["",""])[1]

def _maybe_match(intended, poss):
    if intended in poss:
        return intended
    if " VIA CONVOY" in intended and intended.replace(" VIA CONVOY", "") in poss:
        return intended.replace(" VIA CONVOY", "")
    vc = intended + " VIA CONVOY" if (" - " in intended and intended.startswith("A ")) else None
    if vc and vc in poss:
        return vc
    n = " ".join(intended.split())
    for p in poss:
        if " ".join(p.split()) == n:
            return p
    return None

def _weighted_build_choice(orders, fleet_ratio):
    """Prefer fleets vs armies when choosing a BUILD order from `orders`.
    If no builds are present, fall back to uniform random from `orders`.
    """
    builds = [o for o in orders if o.endswith(' B')]
    if not builds:
        return random.choice(orders)
    f_builds = [o for o in builds if o.startswith('F ')]
    a_builds = [o for o in builds if o.startswith('A ')]
    if f_builds and not a_builds:
        return random.choice(f_builds)
    if a_builds and not f_builds:
        return random.choice(a_builds)
    return random.choice(f_builds) if random.random() < fleet_ratio else random.choice(a_builds)

def _is_adj(game, utype, a, b):
    a, b = _norm(a), _norm(b)
    return game.map.abuts(utype, a, '-', b)

def _legal_nonconvoy_moves(game, orders):
    """Filter out convoy-related orders and any non-adjacent moves that would
    imply a convoy (e.g., A HOL - NWY VIA). HOLDs and supports are kept.
    """
    out = []
    for o in orders or []:
        if ' C ' in o or ' VIA' in o:
            continue
        parts = o.split()
        if len(parts) >= 4 and parts[2] == '-':
            utype = parts[0]
            src = _norm(parts[1])
            dest = _norm(parts[3])
            if not game.map.abuts(utype, src, '-', dest):
                continue
        out.append(o)
    return out

# ---------- agent ----------
class StudentAgent(Agent):
    def __init__(self, agent_name='DumbBotAgent'):
        super().__init__(agent_name)
        self.A = None  # army graph
        self.F = None  # fleet graph

    # ---- init / graphs ----
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self._build_graphs()

    def _build_graphs(self):
        if not self.game:
            raise Exception('Game Not Initialised.')
        A, F = nx.Graph(), nx.Graph()
        raw = list(self.game.map.loc_type.keys())
        for k in raw:
            t = self.game.map.loc_type[k]; u = k.upper()
            if t in ('LAND','COAST'): A.add_node(u)
            if t in ('WATER','COAST'): F.add_node(u)
        locs = [k.upper() for k in raw]
        for i in locs:
            for j in locs:
                if self.game.map.abuts('A', i, '-', j): A.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j): F.add_edge(i, j)
        self.A, self.F = A, F

    # ---- DAIDE s0003-inspired valuation ----
    def _power_sizes(self):
        sizes = {p: len(self.game.get_centers(p)) for p in self.game.powers}
        return sizes

    def _base_values(self, utype):
        """Assign initial province values for unit type (A/F) with explicit
        distinctions between SC/non‑SC and mine/neutral/enemy.

        Values (requested):
          My SC:            0.5 + 0.5 * (# adjacent enemy units)
          Neutral SC:       8.0
          Enemy SC:         max(3.0, 8.0 − 0.5 * owner_size)
          My non‑SC:        0.5
          Neutral non‑SC:   3.0
          Enemy non‑SC:     1.5
        """
        sizes = self._power_sizes()
        my = self.power_name
        scs = set(self.game.map.scs)
        vals = {}

        # Unit presence (for non‑SC classification)
        my_units = { _norm(l) for l in self.game.get_orderable_locations(my) }
        enemy_units_list = []
        for p in self.game.powers:
            if p != my:
                enemy_units_list += self.game.get_orderable_locations(p)
        enemy_units = { _norm(l) for l in enemy_units_list }

        # Enemy adjacency counts (for My SC formula)
        G = self.A if utype=='A' else self.F
        enemy_adj = {n: 0 for n in G.nodes}
        for n in enemy_adj:
            for e in enemy_units:
                if _is_adj(self.game, utype, n, e):
                    enemy_adj[n] += 1

        my_scs = { _norm(c) for c in self.game.get_centers(my) }

        for n in G.nodes:
            b = _norm(n)
            base = 0.0
            if b in scs:
                # Supply center valuations
                if b in my_scs:
                    base = 0.5 + 0.5 * enemy_adj[n]
                else:
                    # Determine owner if any
                    owner = None
                    for p in self.game.powers:
                        if b in self.game.get_centers(p):
                            owner = p
                            break
                    if owner is None:
                        base = 8.0  # neutral SC
                    else:
                        owner_size = sizes.get(owner, 0)
                        base = max(3.0, 8.0 - 0.5 * owner_size)
            else:
                # Non‑SC valuations by current unit occupancy
                if b in my_units:
                    base = 0.5     # my non‑SC (where my unit sits)
                elif b in enemy_units:
                    base = 1.5     # enemy non‑SC (occupied by enemy unit)
                else:
                    base = 3.0     # neutral non‑SC
            vals[n] = float(base)
        return vals

    def _diffuse(self, utype, values, iters=2, decay=0.6):
        """Average values over neighbors to propagate front importance."""
        G = self.A if utype=='A' else self.F
        v = values.copy()
        for _ in range(iters):
            nv = v.copy()
            for n in G.nodes:
                neigh = list(G.neighbors(n))
                if not neigh: continue
                avg = sum(v[m] for m in neigh) / len(neigh)
                nv[n] = (1-decay)*v[n] + decay*avg
            v = nv
        return v

    # ---- movement only ----
    def get_actions(self):
        if self.game.phase_type != 'M':
            all_poss = self.game.get_all_possible_orders()
            orderable = self.game.get_orderable_locations(self.power_name)
            # Fleet-vs-Army build bias by power (rough "pro" tendencies)
            fleet_bias = {
                'ENGLAND': 0.80,  # very fleet heavy
                'AUSTRIA': 0.20,  # army heavy
                'GERMANY': 0.25,  # army leaning
                'FRANCE':  0.50,  # balanced
                'ITALY':   0.40,  # slight army lean
                'RUSSIA':  0.50,  # balanced (north/south split)
                'TURKEY':  0.45,  # slightly more fleets early but mixed
            }
            r = fleet_bias.get(self.power_name, 0.5)
            outs = []
            for loc in orderable:
                poss = all_poss.get(loc, [])
                if not poss:
                    continue
                # If we have any build orders, choose with bias; else fallback random (retreats/disbands)
                if any(o.endswith(' B') for o in poss):
                    outs.append(_weighted_build_choice(poss, r))
                else:
                    outs.append(random.choice(poss))
            return outs
        all_poss = self.game.get_all_possible_orders()
        orderable = self.game.get_orderable_locations(self.power_name)
        my_scs = set(self.game.get_centers(self.power_name))
        season = (self.game.get_current_phase() or 'S')[0]
        scs = set(self.game.map.scs)
        owned_all = set(u for p in self.game.powers for u in self.game.get_centers(p))
        neutral_scs = scs - owned_all

        # Opening book override for movement phases in 1901 only.
        tag = self.game.get_current_phase() or ''
        try:
            year = int(tag[1:5])
        except Exception:
            year = None
        pre_orders, used = {}, set()
        if year == 1901 and tag.endswith('M'):
            key = (self.power_name, tag)
            if key in OPENING_BOOK:
                for intended in OPENING_BOOK[key]:
                    origin = _origin(intended)
                    loc = origin if origin in orderable else next((c for c in orderable if c == origin or _norm(c) == _norm(origin)), None)
                    if loc and loc not in used:
                        poss = all_poss.get(loc, [])
                        m = _maybe_match(intended, poss)
                        if m:
                            pre_orders[loc] = m
                            used.add(loc)
        orderable_remain = [loc for loc in orderable if loc not in used]


        # 1) province values per type + diffusion
        Avals = self._diffuse('A', self._base_values('A'))
        Fvals = self._diffuse('F', self._base_values('F'))

        # 2) initial move choice per unit = best valued destination among legal moves
        orders = dict(pre_orders)
        intents = {}  # dest -> list of movers (only for non-book moves)
        for loc in orderable_remain:
            poss = _legal_nonconvoy_moves(self.game, all_poss.get(loc, []))
            if not poss: continue
            utype = None
            for o in poss:
                if o and o[0] in 'AF': utype = o[0]; break
            if not utype: continue

            # Fall: if on a NEUTRAL SC, prefer HOLD to secure it for winter
            if season == 'F' and _norm(loc) in neutral_scs:
                hold = next((o for o in poss if o.endswith(' H')), (poss[0] if poss else None))
                if hold:
                    orders[loc] = hold
                    continue

            # Evaluate moves; include HOLD as a candidate with current tile value
            best_o, best_v = None, -1e9
            cur_v = (Avals if utype=='A' else Fvals).get(loc.upper(), 0.0)
            hold_o = next((o for o in poss if o.endswith(' H')), None)
            if hold_o and cur_v > best_v:
                best_o, best_v = hold_o, cur_v
            for o in poss:
                p = o.split()
                if len(p) >= 4 and p[2] == '-':
                    dest = p[3].upper()
                    v = (Avals if utype=='A' else Fvals).get(dest, 0.0)
                    if v > best_v:
                        best_o, best_v = o, v
            if best_o:
                orders[loc] = best_o
                # record intent if it's a move
                if ' - ' in best_o:
                    dest = best_o.split()[3]
                    intents.setdefault(_norm(dest), []).append(loc)

        # 3) simple legal support when two of ours attack same province
        for dest, movers in intents.items():
            if len(movers) < 2: continue
            # choose a spearhead (first) and convert others to support if adjacent
            spear = movers[0]
            spear_order = orders[spear]
            for s_loc in movers[1:]:
                # can s_loc legally support spear into dest?
                poss = _legal_nonconvoy_moves(self.game, all_poss.get(s_loc, []))
                stype = next((o[0] for o in poss if o and o[0] in 'AF'), None)
                if stype and _is_adj(self.game, stype, s_loc, dest):
                    orders[s_loc] = f"{stype} {s_loc} S {spear_order.split(' VIA')[0]}"

        # 4) emit one order per unit (book + algorithm for extra units)
        return [orders[loc] for loc in orderable if loc in orders and orders[loc]]
