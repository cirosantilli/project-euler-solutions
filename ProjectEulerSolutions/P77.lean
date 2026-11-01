import ProjectEulerStatements.P77
import ProjectEulerSolutions.Lib.CoinChangeArray
import ProjectEulerSolutions.Lib.PrimeList
import Mathlib.Data.List.TakeWhile

/-! A proved sieve and mutable partition table drive an unbounded, terminating search. -/
namespace ProjectEulerSolutions.P77
open ProjectEulerStatements.P31 ProjectEulerStatements.CoinChange
set_option Elab.async false

def sievePrimes (limit : Nat) : List Nat := PrimeList.sievePrimes limit

theorem sievePrimes_correct (limit : Nat) :
    sievePrimes limit = ProjectEulerStatements.PrimePartitions.primesUpTo limit :=
  PrimeList.sievePrimes_correct limit

def active (n : Nat) (parts : List Nat) : List Nat := parts.takeWhile (fun p => decide (p ≤ n))

def countPrimeSummations (n : Nat) (parts : List Nat) : Nat :=
  (CoinChangeArray.compute n (active n parts)).getD n 0

theorem countPrimeSummations_correct (n : Nat) (parts : List Nat)
    (positive : ∀ p ∈ parts, 0 < p) :
    countPrimeSummations n parts = countWaysBound n (active n parts) := by
  have selected : ∀ p ∈ active n parts, 0 < p := fun p member => positive p (List.takeWhile_subset _ member)
  rw [countPrimeSummations, CoinChangeArray.compute_encode _ _ selected]
  simp [CoinChangeArray.encode, Array.getD, countWaysBound_reverse n (active n parts) selected]

theorem candidate_correct (n : Nat) :
    countPrimeSummations n (sievePrimes n) = ProjectEulerStatements.PrimePartitions.count n := by
  rw [sievePrimes_correct, countPrimeSummations_correct _ _
    (ProjectEulerStatements.PrimePartitions.primesUpTo_positive n)]
  have bounded : ∀ p ∈ ProjectEulerStatements.PrimePartitions.primesUpTo n, p ≤ n := by
    intro p member
    have inside : p < n+1 := List.mem_range.mp (List.mem_filter.mp member).1
    omega
  have unchanged : active n (ProjectEulerStatements.PrimePartitions.primesUpTo n) =
      ProjectEulerStatements.PrimePartitions.primesUpTo n := by
    apply List.takeWhile_eq_self_iff.mpr
    intro p member
    simpa using bounded p member
  rw [unchanged]
  rfl

private theorem next_le (target : Int) (n : Nat)
    (upper : n ≤ ProjectEulerStatements.PrimePartitions.answer target)
    (notOver : ¬ target < (countPrimeSummations n (sievePrimes n) : Int)) :
    n+1 ≤ ProjectEulerStatements.PrimePartitions.answer target := by
  rw [candidate_correct] at notOver
  have exceeds := (ProjectEulerStatements.PrimePartitions.answer_spec target).2
  by_contra after
  have same : n = ProjectEulerStatements.PrimePartitions.answer target := by omega
  exact notOver (by simpa [same] using exceeds)

def search (target : Int) (n : Nat) (lower : 2 ≤ n)
    (upper : n ≤ ProjectEulerStatements.PrimePartitions.answer target) : Nat :=
  if exceeds : target < (countPrimeSummations n (sievePrimes n) : Int) then n
  else search target (n+1) (by omega) (next_le target n upper exceeds)
termination_by ProjectEulerStatements.PrimePartitions.answer target-n
decreasing_by
  have := next_le target n upper exceeds
  omega

theorem search_result (target : Int) (n : Nat) (lower : 2 ≤ n)
    (upper : n ≤ ProjectEulerStatements.PrimePartitions.answer target) :
    search target n lower upper = ProjectEulerStatements.PrimePartitions.answer target := by
  rw [search]
  split
  · rename_i exceeds
    rw [candidate_correct] at exceeds
    have least := ProjectEulerStatements.PrimePartitions.answer_le target n ⟨lower, exceeds⟩
    omega
  · rename_i notOver
    exact search_result target (n+1) (by omega) (next_le target n upper notOver)
termination_by ProjectEulerStatements.PrimePartitions.answer target-n
decreasing_by
  have := next_le target n upper (by assumption)
  omega

def firstValueOver (target : Int) : Nat :=
  search target 2 (by omega) (ProjectEulerStatements.PrimePartitions.answer_spec target).1

theorem firstValueOver_correct (target : Int) :
    firstValueOver target = ProjectEulerStatements.PrimePartitions.answer target :=
  search_result target 2 (by omega) (ProjectEulerStatements.PrimePartitions.answer_spec target).1

def solve (target : Int) : Int := firstValueOver target

theorem solve_result (target : Int) : solve target = ProjectEulerStatements.P77.naive target := by
  simp only [solve, firstValueOver_correct, ProjectEulerStatements.P77.naive, ProjectEulerStatements.P77.firstOver]

end ProjectEulerSolutions.P77
