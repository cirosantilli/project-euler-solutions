import ProjectEulerSolutions.Lib.MillerRabinFast
import ProjectEulerSolutions.Lib.PrimePairGraph

namespace ProjectEulerSolutions.PrimePairCertificate.Fast

/-- A reduction-friendly evaluator for the exact finite source relation. -/
def relation (p q : Nat) : Bool :=
  if (p + q) % 3 = 0 then false else
    MillerRabin.Fast.model (p * DecimalShift.fiveDigits q + q) &&
      MillerRabin.Fast.model (q * DecimalShift.fiveDigits p + p)

theorem relation_eq_source (p q : Nat) : relation p q = PrimePairCertificate.relation p q := by
  simp only [relation, PrimePairCertificate.relation, MillerRabin.Fast.model_eq_source]

def upperNeighbors (tree : VertexTree) (p : Nat) : List Nat :=
  tree.filter (fun q => decide (p < q) && relation p q)

theorem upperNeighbors_eq_source (tree : VertexTree) (p : Nat) :
    upperNeighbors tree p = PrimePairCertificate.upperNeighbors tree p := by
  unfold upperNeighbors PrimePairCertificate.upperNeighbors
  congr 1
  funext q
  rw [relation_eq_source]

end ProjectEulerSolutions.PrimePairCertificate.Fast
