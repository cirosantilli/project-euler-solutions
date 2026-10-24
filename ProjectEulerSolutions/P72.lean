import ProjectEulerStatements.P72
import ProjectEulerSolutions.Lib.TotientArraySieveCorrect

set_option linter.style.header false

/-! The linear array sieve supplies every term of the reduced-fraction count. -/
namespace ProjectEulerSolutions.P72
open ProjectEulerStatements
set_option Elab.async false

abbrev computePhiLinear := TotientArraySieve.computePhiLinear

def sumFrom (phi : Array Nat) (index total : Nat) : Nat :=
  if index < phi.size then sumFrom phi (index + 1) (total + phi[index]!) else total
termination_by phi.size - index

/-- The indexed loop sums exactly the unvisited array suffix. -/
theorem sumFrom_correct (phi : Array Nat) (index total : Nat) (remaining : List Nat)
    (suffix : phi.toList.drop index = remaining) :
    sumFrom phi index total = total + remaining.sum := by
  induction remaining generalizing index total with
  | nil =>
      have missing : phi[index]? = none := by
        simpa only [suffix, List.head?_nil, Array.getElem?_toList] using
          (List.head?_drop (l := phi.toList) (i := index)).symm
      have exhausted : phi.size ≤ index := Array.getElem?_eq_none_iff.mp missing
      rw [sumFrom, if_neg (by omega), List.sum_nil, Nat.add_zero]
  | cons head tail ih =>
      have item : phi[index]? = some head := by
        simpa only [suffix, List.head?_cons, Array.getElem?_toList] using
          (List.head?_drop (l := phi.toList) (i := index)).symm
      obtain ⟨bounded, value⟩ := Array.getElem?_eq_some_iff.mp item
      have read : phi[index]! = head := by simp [getElem!_pos, bounded, value]
      rw [sumFrom, if_pos bounded, read,
        ih (index + 1) (total + head) (by simp [List.drop_add_one_eq_tail_drop, suffix])]
      simp [Nat.add_assoc]

def countReducedProperFractions (limit : Nat) : Nat :=
  if limit < 2 then 0 else sumFrom (computePhiLinear limit) 2 0

def solve (limit : Int) : Nat :=
  if limit < 2 then 0 else countReducedProperFractions limit.toNat

theorem solve_eq_result (limit : Int) : solve limit = ReducedFractionCount.result limit := by
  by_cases small : limit < 2
  · rw [solve, if_pos small, ReducedFractionCount.result_small limit small]
  · have large : ¬ limit.toNat < 2 := by omega
    rw [solve, if_neg small, countReducedProperFractions, if_neg large]
    have table : (computePhiLinear limit.toNat).toList =
        (List.range (limit.toNat + 1)).map Nat.totient := by
      apply List.ext_getElem
      · simp [TotientArraySieve.computePhiLinear_size]
      · intro n leftBound rightBound
        have bound : n ≤ limit.toNat := by
          simpa [TotientArraySieve.computePhiLinear_size] using leftBound
        have arrayBound : n < (computePhiLinear limit.toNat).size := by simpa using leftBound
        have read := TotientArraySieve.computePhiLinear_correct limit.toNat n bound
        rw [getElem!_pos (TotientArraySieve.computePhiLinear limit.toNat) n arrayBound] at read
        simpa only [Array.getElem_toList, List.getElem_map, List.getElem_range] using read
    rw [sumFrom_correct _ 2 0 _ rfl, Nat.zero_add, table, ← List.map_drop]
    rw [ReducedFractionCount.tail_sum]
    have size : (limit + 1).toNat = limit.toNat + 1 := by omega
    rw [ReducedFractionCount.result, size]

end ProjectEulerSolutions.P72
