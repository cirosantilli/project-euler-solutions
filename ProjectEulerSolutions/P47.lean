import ProjectEulerSolutions.Lib.P47FactorSieve
import ProjectEulerSolutions.Lib.ConsecutiveFactors
namespace ProjectEulerSolutions.P47
open ProjectEulerStatements.P47 ConsecutiveFactors P21DivisorSieve
set_option Elab.async false

/-- Scan the completed sieve once, retaining the current run length. -/
def scan (k maximum next run : Nat) (cells : Array Nat) : Option Nat :=
  if next ≤ maximum then
    let updated := if cells[next]! = k then run+1 else 0
    if updated = k then some (next-k+1)
    else scan k maximum (next+1) updated cells
  else none
termination_by maximum+1-next
decreasing_by omega

theorem scan_correct (k maximum next run : Nat) (positive : 0 < k)
    (before : next ≤ lastIndex k) (invariant : run = streak (good k) next) :
    scan k maximum next run (encode (maximum+1) numDistinctPrimeFactors) =
      if lastIndex k ≤ maximum then some (least k) else none := by
  rw [scan]
  split
  · rename_i inside
    have get : (encode (maximum+1) numDistinctPrimeFactors)[next]! = numDistinctPrimeFactors next := by
      simp [encode, show next < maximum+1 by omega]
    rw [get]
    have update : (if numDistinctPrimeFactors next = k then run+1 else 0) = streak (good k) (next+1) := by
      simp [streak, good, invariant]
    rw [update]
    by_cases last : next = lastIndex k
    · subst next
      rw [streak_winner k positive, if_pos rfl, if_pos inside]
      congr 1
      have lower := least_lower k positive
      simp only [lastIndex]
      omega
    · have earlier : next+1 < least k+k := by dsimp [lastIndex] at before last; omega
      have shorter := streak_before k (next+1) earlier
      rw [if_neg (by omega)]
      exact scan_correct k maximum (next+1) _ positive (by omega) rfl
  · rename_i done
    rw [if_neg (by omega : ¬ lastIndex k ≤ maximum)]
termination_by maximum+1-next
decreasing_by omega

theorem scan_initial (k maximum : Nat) (positive : 0 < k) :
    scan k maximum 2 0 (P47FactorSieve.compute maximum) =
      if lastIndex k ≤ maximum then some (least k) else none := by
  rw [P47FactorSieve.compute_correct]
  apply scan_correct _ _ _ _ positive
  · have := least_lower k positive
    dsimp [lastIndex]; omega
  · exact (streak_two k positive).symm

theorem missing_bound (k maximum : Nat) (positive : 0 < k)
    (missing : scan k maximum 2 0 (P47FactorSieve.compute maximum) = none) : maximum < lastIndex k := by
  rw [scan_initial k maximum positive] at missing
  split at missing
  · contradiction
  · omega

/-- Recompute the factor-count sieve at twice the bound after an unsuccessful scan. -/
def search (k maximum : Nat) (positive : 0 < k) (nonzero : 0 < maximum) : Nat :=
  match found : scan k maximum 2 0 (P47FactorSieve.compute maximum) with
  | some answer => answer
  | none => search k (2*maximum) positive (by omega)
termination_by lastIndex k + 1 - maximum
decreasing_by
  have missing : scan k maximum 2 0 (P47FactorSieve.compute maximum) = none := by assumption
  have := missing_bound k maximum positive missing
  omega

def solve (k startLimit : Nat) (positive : 0 < k) (nonzero : 0 < startLimit) : Nat :=
  search k startLimit positive nonzero
end ProjectEulerSolutions.P47
