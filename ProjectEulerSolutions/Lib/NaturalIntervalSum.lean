import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Order.Interval.Finset.Nat

namespace ProjectEulerSolutions.NaturalIntervalSum
open scoped BigOperators
set_option Elab.async false

theorem step (f : Nat → Nat) (first last : Nat) (inside : first ≤ last) :
    (∑ n ∈ Finset.Icc first last, f n) = f first + ∑ n ∈ Finset.Icc (first + 1) last, f n := by
  have interval : Finset.Icc first last = insert first (Finset.Icc (first + 1) last) := by
    ext n
    simp only [Finset.mem_Icc, Finset.mem_insert]
    omega
  rw [interval, Finset.sum_insert (by simp)]

theorem empty (f : Nat → Nat) (first last : Nat) (outside : last < first) :
    (∑ n ∈ Finset.Icc first last, f n) = 0 := by
  have interval : Finset.Icc first last = ∅ := by
    ext n
    simp only [Finset.mem_Icc, Finset.notMem_empty, iff_false, not_and]
    omega
  rw [interval, Finset.sum_empty]

end ProjectEulerSolutions.NaturalIntervalSum
