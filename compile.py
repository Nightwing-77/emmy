"""python compile.py path/to/kernel.ir.json [--arch sm_XX] > out.ptx   (needs only emmy-ml==0.3.11 to load the IR).

``--arch`` picks the GPU generation, ``sm_50`` (Maxwell) through ``sm_120`` (Blackwell); the default is ``sm_89``, what
the problems are scheduled for. PTX for an older GPU also runs on every newer one."""

import argparse
import json
import pathlib

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import KernelOp

from ptx import emit

ap = argparse.ArgumentParser(description="Emmy Kernel IR → PTX")
ap.add_argument("ir", type=pathlib.Path, help="a kernel's .ir.json; meta.json must sit beside it")
ap.add_argument("--arch", default="sm_89", help="target GPU, e.g. sm_70, sm_80, sm_90, sm_120 (default sm_89)")
args = ap.parse_args()

meta = json.loads((args.ir.parent / "meta.json").read_text())
g = Graph.from_dict(json.loads(args.ir.read_text()))
kernel = next(n.op for n in g.nodes.values() if isinstance(n.op, KernelOp))
arg_order = next(k for k in meta["kernels"] if k["entry"] == kernel.name)["arg_order"]
print(emit(kernel, {n: g.buffer(n) for n in arg_order}, target=args.arch))
