import json
from dataclasses import replace
import numpy as np
from pages.key_generation.compute.nist.runner import run_all
from pages.offline_keygen_v2 import calibration as C, growing_policy as G, ldpc as L, privacy as P
from research import linear_key_scheme as core
from research.universal_matrix import calibrate, stream
m=json.loads(core.PARENT.read_text())
for case in ("kazan_v4_00","kazan_long_00","july_kaiser9_02","july_kaiser9_03"):
    raw=core.read(m["caches"][case]["decor"]["path"]); raw.acf_pooled=raw.acf_profiles=raw.cca_curves=None
    sel=G.select(raw); times=[s.elapsed_s for s in raw.steps]; dur=float(times[-1]-times[0])
    print(f"== {case} b={sorted(set(sel['profile_bits'].values()))} fallback={sel['fallback_bits']}", flush=True)
    for cdf in ("per_profile","pooled","hybrid","rare_own"):
        kw=dict(cdf_cdf=cdf, n_target=1000 if cdf=="pooled" else 200, preprocess="rms",
                rms_window=50, checks=1, rebalance="per_profile", widths="pilot")
        res=calibrate(raw, sel, **kw)
        lr=int(res.widths[res.kept].sum()); n=G.block_bits(lr)
        used=int(np.unique(res.profile).size)
        res=replace(res, policy_selection=sel|{"block_bits":n,"public_decision_bits":used+2})
        rec=L.run(res, L.LdpcConfig(block_bits=n, soft=True, public_reliability=False), session_id=case+cdf, times=times)
        pa=P.run(rec, res, P.PAConfig(eve_from_frames=False))
        fails={p:[t.name for t in run_all(stream(res,p)) if t.passed is False] for p in ("alice","bob")}
        print(f"   {cdf:12s} L_R={lr:6d} used={used:2d} BDR={100*(C.summary(res)['BDR p̂_e'] or 0):5.2f}% NIST={len(fails['alice'])}/{len(fails['bob'])} N={pa.n:6d} ell={pa.length:6d} bit/s={(pa.length/dur if dur and pa.length else 0):5.1f}", flush=True)
