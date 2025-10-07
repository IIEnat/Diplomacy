import random, re, networkx as nx, timeout_decorator
from agent_baselines import Agent

# ===== Opening book through end of 1902 (incl. Winters) =====
OPENING_BOOK = {
    ("ENGLAND","S1901M"): ["F LON - ENG","F EDI - NTH","A LVP - YOR"],
    ("ENGLAND","F1901M"): ["F NTH - NWY","F ENG - BEL","A YOR - EDI"],
    ("ENGLAND","W1901A"): ["F LON B","F EDI B","A LVP B"],
    ("ENGLAND","S1902M"): ["F NWY H","F BEL - ENG","A EDI - NTH"],
    ("ENGLAND","F1902M"): ["F ENG - BEL","A NTH H","F NWY H"],
    ("ENGLAND","W1902A"): ["F LON B","F EDI B","A LVP B"],

    ("FRANCE","S1901M"): ["A PAR - BUR","A MAR - SPA","F BRE - MAO"],
    ("FRANCE","F1901M"): ["F MAO - POR","A BUR - BEL","A SPA - GAS"],
    ("FRANCE","W1901A"): ["F BRE B","A PAR B","A MAR B"],
    ("FRANCE","S1902M"): ["A GAS - SPA","A BEL H","F POR - MAO"],
    ("FRANCE","F1902M"): ["A BEL - HOL","F MAO - SPA","A SPA - MAR"],
    ("FRANCE","W1902A"): ["A PAR B","F MAR B","F BRE B"],

    ("GERMANY","S1901M"): ["A MUN - RUH","A BER - KIE","F KIE - DEN"],
    ("GERMANY","F1901M"): ["A RUH - BEL","A KIE - HOL","F DEN - SWE"],
    ("GERMANY","W1901A"): ["F KIE B","A MUN B","A BER B"],
    ("GERMANY","S1902M"): ["A BEL H","A HOL - KIE","F SWE H"],
    ("GERMANY","F1902M"): ["A KIE - DEN","A BEL - HOL","F SWE H"],
    ("GERMANY","W1902A"): ["A BER B","A MUN B","F KIE B"],

    ("ITALY","S1901M"): ["A VEN - APU","A ROM - VEN","F NAP - ION"],
    ("ITALY","F1901M"): ["F ION C A APU - TUN","A APU - TUN VIA CONVOY","A VEN - PIE"],
    ("ITALY","W1901A"): ["F NAP B","A ROM B","A VEN B"],
    ("ITALY","S1902M"): ["F ION - EMS","A TUN H","A PIE - MAR"],
    ("ITALY","F1902M"): ["F EMS - AEG","A MAR H","A TUN H"],
    ("ITALY","W1902A"): ["A ROM B","F NAP B","A VEN B"],

    ("AUSTRIA","S1901M"): ["A VIE - GAL","A BUD - SER","F TRI - ALB"],
    ("AUSTRIA","F1901M"): ["A SER - GRE","F ALB S A SER - GRE","A GAL - RUM"],
    ("AUSTRIA","W1901A"): ["F TRI B","A VIE B","A BUD B"],
    ("AUSTRIA","S1902M"): ["A GRE H","F ALB - ION","A RUM - BUL"],
    ("AUSTRIA","F1902M"): ["A BUL H","F ION - AEG","A GRE - SER"],
    ("AUSTRIA","W1902A"): ["F TRI B","A VIE B","A BUD B"],

    ("RUSSIA","S1901M"): ["A MOS - UKR","A WAR - GAL","F SEV - BLA","F STP/SC - BOT"],
    ("RUSSIA","F1901M"): ["F BOT - SWE","A UKR S F BLA - RUM","F BLA - RUM","A GAL - WAR"],
    ("RUSSIA","W1901A"): ["F STP/SC B","A MOS B","F SEV B","A WAR B"],
    ("RUSSIA","S1902M"): ["F SWE H","A WAR - UKR","A RUM - SEV","F RUM H"],
    ("RUSSIA","F1902M"): ["F SWE - BAL","A SEV - UKR","A UKR - GAL","F RUM - SEV"],
    ("RUSSIA","W1902A"): ["A MOS B","F SEV B","F STP/SC B","A WAR B"],

    ("TURKEY","S1901M"): ["F ANK - BLA","A CON - BUL","A SMY - ARM"],
    ("TURKEY","F1901M"): ["A BUL - GRE","F BLA S A BUL - GRE","A ARM - SEV"],
    ("TURKEY","W1901A"): ["F ANK B","A CON B","A SMY B"],
    ("TURKEY","S1902M"): ["F BLA - RUM","A GRE - BUL","A SEV H"],
    ("TURKEY","F1902M"): ["F RUM - BLA","A BUL - CON","A SEV - UKR"],
    ("TURKEY","W1902A"): ["A SMY B","F ANK B","A CON B"],
}

# ===== utils =====
_log = lambda *_: None
_norm = lambda s: s.split('/')[0] if s else s
_origin = lambda o: (o.split()+["",""])[1]
_dest = lambda o: (lambda p: p[p.index('-')+1] if '-' in p else None)(o.split())
_build_site = lambda o: (o.split()+["","",""])[1].split("/")[0] if len(o.split())>=3 else ""
_hold_from = lambda poss: next((o for o in poss if o.endswith(" H")), (poss[0] if poss else None))
_adjust_delta = lambda g,p: len(g.get_centers(p)) - len(g.get_units(power_name=p))

def _maybe_match(intended, poss):
    if intended in poss: return intended
    if " VIA CONVOY" in intended and intended.replace(" VIA CONVOY","") in poss: return intended.replace(" VIA CONVOY","")
    vc = intended+" VIA CONVOY" if (" - " in intended and intended.startswith("A ")) else None
    if vc in poss: return vc
    n = " ".join(intended.split())
    return next((p for p in poss if " ".join(p.split())==n), None)

def _is_adj(game, a, b): 
    s, d = _norm(a), _norm(b)
    return game.map.abuts('A',s,'-',d) or game.map.abuts('F',s,'-',d)

def _nearest_step(G, src_base, targets, pick=lambda x:x):
    if not G: return None
    src = next((n for n in G if _norm(n)==_norm(src_base)), None)
    tnodes = [n for tb in targets for n in G if _norm(n)==_norm(tb)]
    if not src or not tnodes: return None
    try: paths = nx.shortest_path(G, source=src)
    except: return None
    best = min((t for t in tnodes if t in paths and len(paths[t])>=2), key=lambda t: len(paths[t]), default=None)
    return paths[best][1] if best else None

def _find_move_to(poss, loc, dest_node):
    if not poss or not dest_node: return None
    bl, bd = _norm(loc), _norm(dest_node)
    for o in poss:
        p=o.split()
        if len(p)>=4 and _norm(p[1])==bl and p[2]=='-' and _norm(p[3])==bd: return o
    for o in poss:
        p=o.split()
        if len(p)>=4 and _norm(p[1])==bl and " - " in o and _norm(p[3])==bd and " VIA CONVOY" in o: return o
    return None

MC_TOTAL_SAMPLES = 8
CANDIDATE_LIMIT  = 10
OPP_MODEL_P = {"static": .35, "greedy": .45, "random": .20}

class StudentAgent(Agent):
    """Book through 1902; Winters/Retreats handled; post-1902 uses 1-ply Monte Carlo with graph routing.
       Enhanced: opponent model inference from 1901–1902 history at 1903+.
    """
    def __init__(self, agent_name='Bot'):
        super().__init__(agent_name)
        self._last_orders=[]
        self.map_graph_army=None
        self.map_graph_navy=None
        self._opp_model_fixed=None  # set at 1903 via _infer_opp_models

    def build_map_graphs(self):
        if not self.game:
            raise Exception('Game Not Initialised.')
        A, N = nx.Graph(), nx.Graph()

        raw = list(self.game.map.loc_type.keys())             # original keys (may not be upper)
        for k in raw:
            t = self.game.map.loc_type[k]                     # read type from raw key
            name = k.upper()                                  # nodes stored uppercased
            if t in ('LAND', 'COAST'):  A.add_node(name)
            if t in ('WATER', 'COAST'): N.add_node(name)

        locs = [k.upper() for k in raw]
        for i in locs:
            for j in locs:
                if self.game.map.abuts('A', i, '-', j): A.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j): N.add_edge(i, j)

        self.map_graph_army, self.map_graph_navy = A, N

    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game, self.power_name, self._last_orders = game, power_name, []
        self._opp_model_fixed=None
        self.build_map_graphs()

    @timeout_decorator.timeout(1)
    def update_game(self, all_power_orders):
        for p,os in all_power_orders.items(): self.game.set_orders(p, os)
        self.game.process()
        try: status = self.game.get_order_status(power_name=self.power_name)
        except: status=None
        if status and self._last_orders:
            for o in self._last_orders:
                res = status.get(" ".join(o.split()[:2]), None)
                _log(f"{o}: {'SUCCEEDED' if res==[] else ('FAILED ['+', '.join(map(str,res))+']' if res else '(no status)')}") 
        self._last_orders=[]

    # -------- Opponent model inference (run once at 1903) --------
    def _infer_opp_models(self):
        """Classify opponents using 1901–1902 movement history.
        If no orders in 1901–1902 => 'static'.
        Otherwise, 'greedy' iff there exists a destination with at least one move
        to it AND at least one support-to-move into it (across 1901–1902 movement phases).
        Else 'other'.
        took some time to develop but is pretty accurate from testing. 
        """
        fixed = {}
        hist = self.game.get_phase_history()
        mv_phases = [ph for ph in hist if ph.name.endswith("M") and 1901 <= int(ph.name[1:5]) <= 1902]

        for p in self.game.powers:
            if p == self.power_name:
                continue
            total = 0
            convoys = 0
            only_plain_moves = True  # flipped off if we see support/convoy
            sensible_support = False  
            
            for ph in mv_phases:
                orders = (ph.orders or {}).get(p, [])
                total += len(orders)
                
                #print(orders)
                for o in orders:
                    dests = []
                    sdests = []
                    s = " ".join(o.split())
                    dest = s.split(" - ")[-1].split()[0]
                    
                    if " S " in s:
                        sdests.append(dest)
                        only_plain_moves = False
                    else:
                        dests.append(dest)

                    if " VIA CONVOY" in s or " C " in s:
                        convoys += 1
                        only_plain_moves = False
                
                if total == 0:
                    #print(p, "is static")
                    fixed[p] = "static"
                    continue

                for i in sdests:
                    if i in dests:
                        sensible_support = True

            if (convoys == 0 and sensible_support == True) or only_plain_moves == True:
                #print(p, "is greedy")
                fixed[p] = "greedy"
            else:
                #print(p, "is other")
                fixed[p] = "other"

        self._opp_model_fixed = fixed

    def _pick_builds(self, tag):
        delta = _adjust_delta(self.game, self.power_name)
        if delta<=0: return []
        flat = [o for loc in self.game.get_orderable_locations(self.power_name) for o in self.game.get_all_possible_orders().get(loc,[])]
        legal = [o for o in flat if o.endswith(" B")]
        if not legal: return []
        key=(self.power_name,tag); chosen=[]; used=set()
        if key in OPENING_BOOK:
            for intended in OPENING_BOOK[key]:
                if intended in legal:
                    site=_build_site(intended)
                    if site and site not in used and len(chosen)<delta: chosen.append(intended); used.add(site)
        for o in legal:
            if len(chosen)>=delta: break
            site=_build_site(o)
            if site and site not in used: chosen.append(o); used.add(site)
        return chosen

    def _pick_removals(self):
        need = -_adjust_delta(self.game, self.power_name)
        if need<=0: return []
        flat = [o for loc in self.game.get_orderable_locations(self.power_name) for o in self.game.get_all_possible_orders().get(loc,[])]
        dis = [o for o in flat if o.endswith(" D")]
        return dis[:need] if dis else []

    def _pick_retreats(self):
        scs=set(self.game.map.scs); outs=[]
        ap=self.game.get_all_possible_orders()
        for loc in self.game.get_orderable_locations(self.power_name):
            poss=ap.get(loc,[]); pick=next((o for o in poss if _dest(o) in scs), None)
            outs.append(pick or next((o for o in poss if not o.endswith(" D")), (poss[0] if poss else None)))
        return outs

    def _candidate_sets(self, all_possible, orderable):
        scs, myscs = set(self.game.map.scs), set(self.game.get_centers(self.power_name))
        def utype(loc): 
            for o in all_possible.get(loc,[]): 
                if o and o[0] in "AF": return o[0]
        def step(loc, targets):
            G = self.map_graph_navy if utype(loc)=='F' else self.map_graph_army
            return _nearest_step(G, loc, targets)
        greedy=[]
        for loc in orderable:
            poss=all_possible.get(loc,[]); mv=None
            sn=step(loc, [s for s in scs if s not in myscs])
            if sn: mv=_find_move_to(poss, loc, sn)
            greedy.append(mv if mv else _hold_from(poss))
        enemy_units=[l for p in self.game.powers if p!=self.power_name for l in self.game.get_orderable_locations(p)]
        threatened={_norm(sc) for sc in myscs if any(_is_adj(self.game,e,sc) for e in enemy_units)}
        cover=[_hold_from(all_possible.get(loc,[])) for loc in orderable]  # (heuristic collapsed to holds)
        probe=[]
        for loc in orderable:
            poss=all_possible.get(loc,[]); best=next((o for o in poss if (_dest(o) in scs and _dest(o) not in myscs)), None)
            probe.append(best or _hold_from(poss))
        holds=[_hold_from(all_possible.get(loc,[])) for loc in orderable]
        cand=[]; seen=set()
        def add(v): 
            k=tuple(sorted(v))
            if k not in seen and len(v)==len(orderable): seen.add(k); cand.append(list(v))
        for s in (greedy,cover,probe,holds): add(s)
        idx=list(range(len(orderable))); random.shuffle(idx)
        for i in idx[:6]:
            loc=orderable[i]; poss=all_possible.get(loc,[])
            alt=next((o for o in poss if not o.endswith(" H") and o!=greedy[i]), None)
            if alt: g2=list(greedy); g2[i]=alt; add(g2)
            if len(cand)>=CANDIDATE_LIMIT: break
        return cand

    def _sample_opponents(self, all_possible):
        scs = set(self.game.map.scs); joint = {}
        fixed = self._opp_model_fixed or {}
        for p in self.game.powers:
            if p == self.power_name:
                continue
            orderable = self.game.get_orderable_locations(p)
            model = fixed.get(p)
            if model == "other" or model is None:
                model = "random"
            my_scs = set(self.game.get_centers(p)); out = []
            for loc in orderable:
                poss = all_possible.get(loc, [])
                if model == "static":
                    out.append(_hold_from(poss))
                elif model == "random":
                    out.append(random.choice(poss) if poss else None)
                else:  # greedy
                    ut = next((o[0] for o in poss if o and o[0] in "AF"), None)
                    G = self.map_graph_navy if ut == 'F' else self.map_graph_army
                    step = _nearest_step(G, loc, [s for s in scs if s not in my_scs]) if G else None
                    mv = _find_move_to(poss, loc, step) if step else None
                    out.append(mv if mv else _hold_from(poss))
            joint[p] = [o for o in out if o]
        return joint

    def _threat_prob(self, all_possible, samples=MC_TOTAL_SAMPLES):
        counts={}; total=0
        for _ in range(samples):
            opp=self._sample_opponents(all_possible); total+=1; seen=set()
            for plist in opp.values():
                for o in plist:
                    d=_dest(o)
                    if d: seen.add(_norm(d))
            for d in seen: counts[d]=counts.get(d,0)+1
        return {d:c/total for d,c in counts.items()} if total else {}

    def _score(self, orders, threat_prob):
        myscs={_norm(s) for s in self.game.get_centers(self.power_name)}
        scs={_norm(s) for s in self.game.map.scs}
        vac=set(); gains=0.0
        for o in orders:
            p=o.split()
            if len(p)>=4 and p[2]=='-':
                o0,d0=_norm(p[1]),_norm(p[3])
                if o0 in myscs: vac.add(o0)
                if d0 in scs and d0 not in myscs: gains += (1.0 - threat_prob.get(d0,0.0))
        losses=sum(threat_prob.get(s,0.0) for s in vac)
        exposure=.2*sum(threat_prob.get(s,0.0) for s in vac)
        return gains - losses - exposure

    @timeout_decorator.timeout(1)
    def get_actions(self):
        tag = self.game.get_current_phase()
        phase_type = self.game.phase_type
        power= self.power_name
        year = int(tag[1:5])
        all_possible = self.game.get_all_possible_orders()
        orderable = self.game.get_orderable_locations(power)

        # one-time inference when leaving the book (>= 1903)
        if year >= 1903 and self._opp_model_fixed is None:
            self._infer_opp_models()

        if phase_type=="M":
            if year and year<=1902:
                used=set(); outs=[]
                key=(power,tag)
                if key in OPENING_BOOK:
                    for intended in OPENING_BOOK[key]:
                        origin= _origin(intended)
                        loc = origin if origin in orderable else next((c for c in orderable if c==origin or _norm(c)==_norm(origin)), None)
                        if loc and loc not in used:
                            m=_maybe_match(intended, all_possible.get(loc,[]))
                            if m: outs.append(m); used.add(loc)
                for loc in orderable:
                    if loc not in used:
                        poss=all_possible.get(loc,[])
                        if poss: outs.append(_hold_from(poss))
                outs=list(dict.fromkeys(outs)); self._last_orders=outs[:]; return outs

            cand_sets=self._candidate_sets(all_possible, orderable)
            threat=self._threat_prob(all_possible, samples=MC_TOTAL_SAMPLES)
            best=max(cand_sets or [[_hold_from(all_possible.get(loc,[])) for loc in orderable]], key=lambda orders:self._score(orders, threat))
            outs=list(dict.fromkeys(best)); self._last_orders=outs[:]; return outs

        if phase_type=="A":
            delta=_adjust_delta(self.game, power)
            outs = self._pick_builds(tag) if delta>0 else (self._pick_removals() if delta<0 else [])
            outs=list(dict.fromkeys(outs)); self._last_orders=outs[:]; return outs

        if phase_type=="R":
            outs=list(dict.fromkeys(self._pick_retreats())); self._last_orders=outs[:]; return outs

        self._last_orders=[]; return []
