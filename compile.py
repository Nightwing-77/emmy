"""python compile.py path/to/kernel.ir.json > out.ptx  (needs only emmy-ml==0.3.11 to load the IR)."""
import json
import pathlib
import sys

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel import KernelOp

from ptx import emit

ir = pathlib.Path(sys.argv[1])
meta = json.loads((ir.parent / "meta.json").read_text())
g = Graph.from_dict(json.loads(ir.read_text()))
kernel = next(n.op for n in g.nodes.values() if isinstance(n.op, KernelOp))
arg_order = next(k for k in meta["kernels"] if k["entry"] == kernel.name)["arg_order"]
print(emit(kernel, {n: g.buffer(n) for n in arg_order}, cap=(8, 9)))
