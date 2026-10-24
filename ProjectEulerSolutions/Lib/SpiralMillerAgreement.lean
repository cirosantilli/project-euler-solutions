import ProjectEulerSolutions.Lib.SpiralPrimes
import ProjectEulerSolutions.Lib.MillerRabin

/-! Finite agreement between the exact Miller-Rabin model and genuine spiral prime counts.
Actual Python source execution is a separate theorem consuming this agreement. -/
namespace ProjectEulerSolutions.SpiralPrimes

variable {first middle last : Nat}

def modelRow (layer : Nat) : Nat := ((corners layer).filter MillerRabin.isPrimeModel).length

def RowAgreement (first last : Nat) : Prop :=
  ∀ layer, first < layer → layer ≤ last → modelRow layer = row layer

/-- One opaque model-row equality supplies the entire unit interval. -/
theorem RowAgreement.single (checked : modelRow (first + 1) = row (first + 1)) :
    RowAgreement first (first + 1) := by
  intro layer lower upper
  have same : layer = first + 1 := by omega
  subst layer
  exact checked

/-- Opaque interval facts compose without evaluating any primality or Miller-Rabin call. -/
theorem RowAgreement.join (left : RowAgreement first middle) (right : RowAgreement middle last) :
    RowAgreement first last := by
  intro layer lower upper
  by_cases before : layer ≤ middle
  · exact left layer lower before
  · exact right layer (by omega) upper

end ProjectEulerSolutions.SpiralPrimes
