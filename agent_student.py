import random
import networkx as nx
from agent_baselines import Agent

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

# ---------- helpers ----------
_norm = lambda s: s.split('/')[0] if s else s

def _era_from_year(y: int | None):
    if y is None:
        return 'early'
    if y <= 1905:
        return 'early'
    if y <= 1915:
        return 'mid'
    return 'late'

_origin = lambda o: (o.split()+["",""])[1]

def _book_helper(intended, poss):
    ni = " ".join(intended.split())
    for p in (poss or []):
        if " ".join(p.split()) == ni:
            return p
    return None

def _weighted_build_choice(orders, fleet_ratio):
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
    def __init__(self, agent_name='StudentAgent'):
        super().__init__(agent_name)
        self.A = None  
        self.F = None  
        self.static_powers = False

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

    # This is used only in scenario 1 to prevent Italy from getting stuck and for England to get a higher SC count
    def _owner_multiplier(self, province: str) -> float:
        if not province:
            return 1.0
        prov = _norm(province)
        owner = None
        for p in self.game.powers:
            if prov in { _norm(c) for c in self.game.get_centers(p) }:
                owner = p; break
        if self.power_name == 'ITALY' and owner == 'AUSTRIA':
            return 10
        if self.power_name == 'ENGLAND' and owner == 'GERMANY':
            return 3
        return 1.0

    # Called once and determines if the game is scenario 1
    def _check_static_powers(self):
        hist = self.game.get_phase_history() or []
        mv_phases = [ph for ph in hist
                     if ph and getattr(ph, 'name', '').endswith('M')
                     and ph.name[1:5].isdigit()
                     and 1901 <= int(ph.name[1:5]) <= 1902]
        all_others_static = True
        for p in self.game.powers:
            if p == self.power_name:
                continue
            total = 0
            for ph in mv_phases:
                orders = (getattr(ph, 'orders', None) or {}).get(p, [])
                total += len(orders)
            if total != 0:
                all_others_static = False
                break
        self.static_powers = all_others_static

    def _power_sizes(self):
        sizes = {p: len(self.game.get_centers(p)) for p in self.game.powers}
        return sizes

    def _base_values(self, utype):
        tag = self.game.get_current_phase() 
        year = int(tag[1:5])
        if tag == "S1902M": self._check_static_powers()
        stage = _era_from_year(year) 
        neutral_mul = 1.5 if stage == 'early' else (1.25 if stage == 'mid' else 1.0)

        sizes = self._power_sizes()
        my = self.power_name
        scs = set(self.game.map.scs)
        vals = {}
        my_units = { _norm(l) for l in self.game.get_orderable_locations(my) }
        enemy_units_list = []
        for p in self.game.powers:
            if p != my:
                enemy_units_list += self.game.get_orderable_locations(p)
        enemy_units = { _norm(l) for l in enemy_units_list }

        G = self.A if utype=='A' else self.F
        enemy_adj = {n: 0 for n in G.nodes}
        for n in enemy_adj:
            for e in enemy_units:
                if _is_adj(self.game, utype, n, e):
                    enemy_adj[n] += 1

        my_scs = { _norm(c) for c in self.game.get_centers(my) }
        for n in G.nodes:
            b = _norm(n)
            m = 1.0 
            if b in scs:
                # Heuristic for my, enemy, neutral SC
                if b in my_scs:
                    base = (1 + enemy_adj[n]) / neutral_mul
                    if self.static_powers:
                        m *= -1
                else:
                    owner = None
                    for p in self.game.powers:
                        if b in self.game.get_centers(p):
                            owner = p; break
                    if owner is None:
                        base = 8 * neutral_mul
                        if self.static_powers:
                            m *= 5
                    else:
                        base = 8.0 + (1.5 * sizes.get(owner,0))
                        if self.static_powers:
                            m *= 1
                            m *= self._owner_multiplier(b)
            else:
                # Heuristic for my, enemy, neutral non-SC
                if b in my_units:
                    base = 1
                    if self.static_powers:
                        m *= -1
                elif b in enemy_units:
                    base = 3
                    if self.static_powers:
                        m *= 1
                else:
                    base = 3 * neutral_mul
                    if self.static_powers:
                        m *= 5
            vals[n] = float(base * m)
        return vals

    def _diffuse(self, utype, values, iters=2, decay=0.5):
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

    def get_actions(self):
        # 1) Non-movement phases: builds are based on fleet / army ratio, disbands are just random
        if self.game.phase_type != 'M':
            all_poss = self.game.get_all_possible_orders()
            orderable = self.game.get_orderable_locations(self.power_name)
            # Higher number means more fleets
            fleet_bias = {
                'ENGLAND': 0.85,  
                'AUSTRIA': 0.20, 
                'GERMANY': 0.25,  
                'FRANCE':  0.50,  
                'ITALY':   0.40,  
                'RUSSIA':  0.50,  
                'TURKEY':  0.45,  
            }
            r = fleet_bias.get(self.power_name, 0.5)
            outs = []
            for loc in orderable:
                poss = all_poss.get(loc, [])
                if not poss:
                    continue
                if any(o.endswith(' B') for o in poss):
                    outs.append(_weighted_build_choice(poss, r))
                else:
                    outs.append(random.choice(poss))
            return outs

        # 2) Opening book fast-path: only in 1901 movement phases; return immediately if fully valid
        tag = self.game.get_current_phase()
        year = int(tag[1:5])
        if year == 1901 and tag.endswith('M'):
            key = (self.power_name, tag)
            if key in OPENING_BOOK:
                all_poss = self.game.get_all_possible_orders()
                orderable = self.game.get_orderable_locations(self.power_name)
                intended_list = OPENING_BOOK[key]
                origin_to_intended = {}
                for intended in intended_list:
                    origin = _origin(intended)
                    origin_to_intended[_norm(origin)] = intended
                pre_orders = {}
                # For every move in the book, check that it also exists in the orderable moves
                for loc in orderable:
                    base = _norm(loc)
                    if base in origin_to_intended:
                        poss = all_poss.get(loc, [])
                        m = _book_helper(origin_to_intended[base], poss)
                        if m is None:
                            pre_orders = None
                            break
                        pre_orders[loc] = m
                # If it is not valid, this will fail and so fall through to the greedy algorithm
                if pre_orders:
                    return [pre_orders[loc] for loc in orderable if loc in pre_orders]
        
        # 3) Greedyyyyyyy Algorithm
        all_poss = self.game.get_all_possible_orders()
        orderable = self.game.get_orderable_locations(self.power_name)
        my_scs = set(self.game.get_centers(self.power_name))
        season = (self.game.get_current_phase() or 'S')[0]
        scs = set(self.game.map.scs)
        owned_all = set(u for p in self.game.powers for u in self.game.get_centers(p))
        neutral_scs = scs - owned_all
        enemy_scs = scs - my_scs - neutral_scs

        # 3A) province values per type + diffusion
        Avals = self._diffuse('A', self._base_values('A'))
        Fvals = self._diffuse('F', self._base_values('F'))

        # 3B) initial move choice per unit = best valued destination among legal moves
        orders = {}
        intents = {} 
        for loc in orderable:
            poss = _legal_nonconvoy_moves(self.game, all_poss.get(loc, []))
            if not poss: continue
            utype = None
            for o in poss:
                if o and o[0] in 'AF': utype = o[0]; break
            if not utype: continue

            # Fall: if on a Neutral or Enemy SC, prefer HOLD to secure it for winter
            if season == 'F' and (_norm(loc) in neutral_scs or _norm(loc) in enemy_scs):
                hold = next((o for o in poss if o.endswith(' H')), (poss[0] if poss else None))
                if hold:
                    orders[loc] = hold
                    continue

            # Evaluate best moves
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
                if ' - ' in best_o:
                    dest = best_o.split()[3]
                    intents.setdefault(_norm(dest), []).append(loc)

        # SPRING: recruit supports for attacks into enemy-held SCs
        if season == 'S':
            enemy_sc_attacks = [] 
            for loc, ord_str in list(orders.items()):
                if ' - ' in ord_str:
                    parts = ord_str.split()
                    if len(parts) >= 4 and parts[2] == '-':
                        dest_base = _norm(parts[3])
                        if dest_base in enemy_scs:
                            enemy_sc_attacks.append((loc, ord_str, dest_base))
            used_supporters = set()
            for spear_loc, spear_order, dest_base in enemy_sc_attacks:
                spear_clean = spear_order.split(' VIA')[0]
                for s_loc in orderable:
                    if s_loc == spear_loc or s_loc in used_supporters:
                        continue
                    poss_raw = all_poss.get(s_loc, [])
                    stype = next((o[0] for o in poss_raw if o and o[0] in 'AF'), None)
                    if not stype:
                        continue
                    if not _is_adj(self.game, stype, s_loc, dest_base):
                        continue
                    support_order = f"{stype} {s_loc} S {spear_clean}"
                    if support_order in poss_raw:
                        orders[s_loc] = support_order
                        used_supporters.add(s_loc)

        # Rebuild intents from current orders (moves only) to avoid supporting a support
        intents = {}
        for loc_o, ord_str_o in orders.items():
            pts = ord_str_o.split()
            if len(pts) >= 4 and pts[2] == '-':
                intents.setdefault(_norm(pts[3]), []).append(loc_o)

        # simple legal support when two of ours attack same province
        for dest, movers in intents.items():
            if len(movers) < 2: continue
            spear = movers[0]
            spear_order = orders[spear]
            for s_loc in movers[1:]:
                poss = _legal_nonconvoy_moves(self.game, all_poss.get(s_loc, []))
                stype = next((o[0] for o in poss if o and o[0] in 'AF'), None)
                if stype and _is_adj(self.game, stype, s_loc, dest):
                    orders[s_loc] = f"{stype} {s_loc} S {spear_order.split(' VIA')[0]}"

        # Emit one order per unit
        return [orders[loc] for loc in orderable if loc in orders and orders[loc]]
