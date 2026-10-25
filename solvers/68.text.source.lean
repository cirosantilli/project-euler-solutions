import «.formalsw».solvers.«68.py.ast»
import ProjectEulerStatements.Lib.MagicRingTextCalls

namespace solvers.«68»
open FormalSW.PyAst ProjectEulerStatements MagicRingTextCalls
set_option Elab.async false
abbrev program := Solvers.P68Ast.module
private abbrev textArgs := ((findFunctionIn program.body "ring_string").getD ([], [])).1
private abbrev outerArg := textArgs.getD 0 ⟨"", none, none⟩
private abbrev innerArg := textArgs.getD 1 ⟨"", none, none⟩
private abbrev startArg := textArgs.getD 2 ⟨"", none, none⟩

theorem ring_string.result {n : Nat} (ring : MagicRing.Ring n) (start : Nat)
    (heap : Mutable.Heap) (outerAddress innerAddress : Nat)
    (getOuter : heap[outerAddress]? = some (.list (encode ring.outer)))
    (getInner : heap[innerAddress]? = some (.tuple (encode ring.inner))) :
    HeapCalls.ReturnsValue program "ring_string"
      [.ref outerAddress, .ref innerAddress, number start] heap
      (string (MagicRing.textFrom ring.outer ring.inner start))
      (working heap (pieces ring start n)) :=
  MagicRingTextCalls.result program outerArg innerArg startArg rfl rfl rfl rfl
    ring start heap outerAddress innerAddress getOuter getInner

end solvers.«68»
