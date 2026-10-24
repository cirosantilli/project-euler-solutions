import ProjectEulerStatements.P71

set_option linter.style.header false

/-! The denominator scan tests the greatest numerator below the target at each step. -/
namespace ProjectEulerSolutions.P71
open ProjectEulerStatements.FractionSearch
set_option Elab.async false

abbrev gcd := Nat.gcd

def bestFractionLeftOf (targetN targetD maxD : Nat) : Pair :=
  loop 1 (0, 1)
where
  loop (d : Nat) (best : Pair) : Pair :=
    if d ≤ maxD then
      loop (d + 1) (optimizedStep targetN targetD d best)
    else best
  termination_by maxD + 1 - d

/-- The scan advances through precisely the remaining denominator interval. -/
theorem loop_correct (a b limit first count : Nat) (positive : 0 < first)
    (length : first + count = limit + 1) :
    bestFractionLeftOf.loop a b limit first (optimizedUpTo a b (first - 1)) =
      optimizedUpTo a b limit := by
  induction count generalizing first with
  | zero =>
      rw [bestFractionLeftOf.loop, if_neg (by omega)]
      congr 1
      omega
  | succ count ih =>
      rw [bestFractionLeftOf.loop, if_pos (by omega)]
      have step : optimizedStep a b first (optimizedUpTo a b (first - 1)) =
          optimizedUpTo a b first := by
        have same : first - 1 + 1 = first := by omega
        conv_rhs => rw [← same, optimizedUpTo]
        rw [same]
      rw [step]
      have next := ih (first + 1) (by omega) (by omega)
      simpa only [Nat.add_sub_cancel] using next

theorem bestFractionLeftOf_correct (a b limit : Nat) :
    bestFractionLeftOf a b limit = optimizedUpTo a b limit :=
  loop_correct a b limit 1 limit (by decide) (by omega)

def solve (limit : Int) (num : Int := 3) (den : Int := 7) : Nat :=
  (bestFractionLeftOf num.toNat den.toNat limit.toNat).1

/-- The public signed interface agrees with the exhaustive reference for every proper target. -/
theorem solve_eq_reference (limit num den : Int) (positive : 0 < num) (proper : num < den) :
    solve limit num den = ProjectEulerStatements.P71.naive limit num den := by
  unfold solve ProjectEulerStatements.P71.naive ProjectEulerStatements.P71.leftOf
  rw [bestFractionLeftOf_correct, optimized_eq_exhaustive num.toNat den.toNat limit.toNat
    (by omega) (by omega)]

end ProjectEulerSolutions.P71
