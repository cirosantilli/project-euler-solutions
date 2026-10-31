import solvers.«89.py.lean.d».Inputs

namespace solvers.«89»
open FormalSW.PyAst

/-- Complete source equivalence on the generalized Roman-symbol input domain. -/
theorem solve_source_equivalence (heap : Mutable.Heap) (globals : Mutable.Globals)
    (address : Nat) (lines : List String) :
    source_equivalence.Domain heap globals address lines →
      source_equivalence.Verified heap globals address :=
  source_equivalence.verified heap globals address lines

/-- The contract covers every valid list with the standard table contents. -/
theorem solve_standard_source_equivalence (lines : List String) (valid : ∀ text ∈ lines, ValidText text) :
    source_equivalence.Verified (source_equivalence.standardHeap lines)
      source_equivalence.standardGlobals source_equivalence.standardBase.length :=
  solve_source_equivalence _ _ _ lines (source_equivalence.standard_domain lines valid)

end solvers.«89»
