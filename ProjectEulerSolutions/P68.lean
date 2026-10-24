import ProjectEulerStatements.P68
import ProjectEulerStatements.Lib.MagicRingPruning

set_option linter.style.header false

/-! Magic-ring search with sound width and line-sum pruning for arbitrary signed arguments. -/
namespace ProjectEulerSolutions.P68
open ProjectEulerStatements.MagicRing FormalSW.PyAst

def solve (n : Int := 5) (digits : Int := 16) : String :=
  if n < 3 ∨ digits < 3 * n then ""
  else prunedSelectionScan n.toNat digits.toNat (Combinations.rows n.toNat (labels n.toNat)) ""

end ProjectEulerSolutions.P68
