import Mathlib.Data.Rat.Lemmas
import Mathlib.Data.Int.GCD
import Mathlib.Tactic

/-! Canonical integer pairs for exact rational expression evaluation. -/
namespace ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev Pair := Int × Int

def value (pair : Pair) : Rat := Rat.divInt pair.1 pair.2

def reduced (numerator denominator : Int) : Pair :=
  let divisor : Int := numerator.gcd denominator
  (numerator / divisor, denominator / divisor)

def normalize (numerator denominator : Int) : Pair :=
  if denominator < 0 then reduced (-numerator) (-denominator)
  else reduced numerator denominator

/-- Reduction produces exactly Lean's canonical numerator and positive denominator. -/
theorem reduced_canonical (numerator : Int) (denominator : Nat) (positive : 0 < denominator) :
    reduced numerator denominator =
      ((Rat.divInt numerator denominator).num, ((Rat.divInt numerator denominator).den : Int)) := by
  simp [reduced, Rat.divInt, mkRat, positive.ne', Rat.normalize_eq, Int.gcd]

/-- Normalization covers either denominator sign, without a numeric bound. -/
theorem normalize_canonical (numerator denominator : Int) (nonzero : denominator ≠ 0) :
    normalize numerator denominator =
      ((Rat.divInt numerator denominator).num, ((Rat.divInt numerator denominator).den : Int)) := by
  by_cases negative : denominator < 0
  · have positive : 0 < -denominator := by omega
    lift -denominator to Nat using positive.le with d hd
    have equal : Rat.divInt (-numerator) (d : Int) = Rat.divInt numerator denominator := by
      rw [hd, Rat.divInt_neg, neg_neg]
    rw [normalize, if_pos negative, ← hd, reduced_canonical (-numerator) d (by omega), equal]
  · have positive : 0 < denominator := by omega
    lift denominator to Nat using positive.le
    simpa [normalize, negative] using reduced_canonical numerator denominator (by omega)

theorem normalize_value (numerator denominator : Int) (nonzero : denominator ≠ 0) :
    value (normalize numerator denominator) = Rat.divInt numerator denominator := by
  rw [normalize_canonical numerator denominator nonzero]
  exact Rat.num_divInt_den _

theorem normalize_positive (numerator denominator : Int) (nonzero : denominator ≠ 0) :
    0 < (normalize numerator denominator).2 := by
  rw [normalize_canonical numerator denominator nonzero]
  change (0 : Int) < ((Rat.divInt numerator denominator).den : Int)
  exact_mod_cast (Rat.divInt numerator denominator).pos

/-- Equal rational values have identical normalized tuple keys. -/
theorem normalize_eq_iff (a b c d : Int) (hb : b ≠ 0) (hd : d ≠ 0) :
    normalize a b = normalize c d ↔ Rat.divInt a b = Rat.divInt c d := by
  constructor
  · intro equal
    simpa only [normalize_value _ _ hb, normalize_value _ _ hd] using congrArg value equal
  · intro equal
    simp only [normalize_canonical _ _ hb, normalize_canonical _ _ hd, equal]

end ProjectEulerSolutions.RationalPair
