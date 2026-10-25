import ProjectEulerStatements.P73

set_option linter.style.header false

/-! Finite coprimality scans over the complete, input-derived numerator intervals. -/
namespace ProjectEulerSolutions.P73
open ProjectEulerStatements.FractionIntervalCount
set_option Elab.async false

def coprimes (d first stop : Int) (total : Nat) : Nat :=
  if first < stop then
    coprimes d (first + 1) stop (if Int.gcd first d = 1 then total + 1 else total)
  else total
termination_by (stop - first).toNat

theorem coprimes_correct (d first stop : Int) (total : Nat) :
    coprimes d first stop total = total + count (fun n => Int.gcd n d = 1) first stop := by
  by_cases active : first < stop
  · rw [coprimes, if_pos active, count_step _ first stop active]
    by_cases chosen : Int.gcd first d = 1
    · rw [if_pos chosen, if_pos chosen, coprimes_correct d (first + 1) stop (total + 1)]
      omega
    · rw [if_neg chosen, if_neg chosen, coprimes_correct d (first + 1) stop total]
      omega
  · rw [coprimes, if_neg active, count_empty _ first stop (by omega), Nat.add_zero]
termination_by (stop - first).toNat
decreasing_by all_goals omega

def countBetween (p : Bounds) (d : Int) : Nat :=
  let first := lower p d
  let last := upper p d
  if last < first then 0 else coprimes d first (last + 1) 0

theorem countBetween_correct (p : Bounds) (positive : Positive p) (d : Int) :
    countBetween p d = row p d := by
  rw [← trimmed_eq_row p positive d]
  unfold countBetween
  dsimp only
  split
  · exact (count_empty _ _ _ (by omega)).symm
  · rw [coprimes_correct, Nat.zero_add]

def scan (p : Bounds) (limit d : Int) (total : Nat) : Nat :=
  if d ≤ limit then scan p limit (d + 1) (total + countBetween p d) else total
termination_by (limit + 1 - d).toNat

theorem scan_correct (p : Bounds) (positive : Positive p) (limit first : Int)
    (count total : Nat) (length : (limit + 1 - first).toNat = count) :
    scan p limit first total = total + rows p first count := by
  induction count generalizing first total with
  | zero => rw [scan, if_neg (by omega), rows, Nat.add_zero]
  | succ count ih =>
      rw [scan, if_pos (by omega), countBetween_correct p positive first]
      rw [ih (first + 1) (total + row p first) (by omega), rows]
      omega

def solve (limit : Int) (lowerNum : Int := 1) (lowerDen : Int := 3)
    (upperNum : Int := 1) (upperDen : Int := 2) : Nat :=
  scan ⟨lowerNum, lowerDen, upperNum, upperDen⟩ limit 2 0

theorem solve_eq_result (limit : Int) (p : Bounds) (positive : Positive p) :
    solve limit p.lowerNum p.lowerDen p.upperNum p.upperDen = result p limit := by
  simpa only [solve, result, Nat.zero_add] using
    scan_correct p positive limit 2 (limit - 1).toNat 0 (by omega)

end ProjectEulerSolutions.P73
