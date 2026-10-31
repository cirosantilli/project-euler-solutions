import solvers.«90.py.lean.d».Inputs

namespace solvers.«90»
open FormalSW.PyAst

/-- Both complete Python functions terminate and agree on arbitrary valid caller states. -/
theorem solve_source_equivalence (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (required : source_equivalence.Required) :
    source_equivalence.Domain heap globals required →
      source_equivalence.Verified count heap globals required.address :=
  source_equivalence.verified count heap globals required

/-- In particular, every finite list of decimal digit pairs is covered, with no bound on its length. -/
theorem solve_standard_source_equivalence (count : Int) (pairs : List (Nat × Nat)) :
    source_equivalence.Verified count (source_equivalence.standardHeap pairs) [] pairs.length :=
  solve_source_equivalence count _ [] _ (source_equivalence.standard_domain pairs)

theorem solve.«Returns» (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (required : source_equivalence.Required) (domain : source_equivalence.Domain heap globals required) :
    (source_equivalence.optimizedCall count heap globals required.address).Returns :=
  (solve_source_equivalence count heap globals required domain).1

theorem solve.«Supported» (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (required : source_equivalence.Required) (domain : source_equivalence.Domain heap globals required) :
    (source_equivalence.optimizedCall count heap globals required.address).Supported :=
  (solve.«Returns» count heap globals required domain).supported

theorem solve.«Equiv(naive)» (count : Int) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (required : source_equivalence.Required) (domain : source_equivalence.Domain heap globals required) :
    (source_equivalence.optimizedCall count heap globals required.address).Equiv
      (source_equivalence.referenceCall count heap globals required.address) :=
  (solve_source_equivalence count heap globals required domain).2.2.2.2

end solvers.«90»
