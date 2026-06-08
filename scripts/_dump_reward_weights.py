import sys, yaml
d = yaml.unsafe_load(open(sys.argv[1]))
r = d["rewards"]
for k, v in r.items():
    if isinstance(v, dict) and "weight" in v:
        w = v["weight"]
        func = v.get("func", "")
        fname = func.split(":")[-1] if isinstance(func, str) else ""
        print(f"{k:40s} weight={w:<10} func={fname}")
