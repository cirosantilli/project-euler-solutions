import ProjectEulerSolutions.Lib.ProductSumTable
import ProjectEulerSolutions.Lib.ListEraseDups
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-! The observed table suffix contains exactly the bounded minima whose distinct values are summed. -/
namespace ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

theorem table_suffix (maxSize : Nat) :
    (table maxSize).drop 2 = (List.range' 2 (maxSize - 1)).map (minimum maxSize) := by
  apply List.ext_getElem
  · simp only [List.length_drop, table_length, List.length_map, List.length_range']
    omega
  · intro index leftBound rightBound
    have inside : 2 + index ≤ maxSize := by
      simp only [List.length_drop, table_length] at leftBound
      omega
    have value := table_lookup maxSize (2 + index) inside
    have tableBound : 2 + index < (table maxSize).length := by rw [table_length]; omega
    simpa [List.getElem_drop, List.getElem_map, List.getElem_range', tableBound] using value

theorem suffix_bounded (maxSize value : Nat) (member : value ∈ (table maxSize).drop 2) :
    value ≤ 2 * maxSize := by
  rw [table_suffix] at member
  obtain ⟨size, within, rfl⟩ := List.mem_map.mp member
  obtain ⟨offset, bound, rfl⟩ := List.mem_range'.mp within
  have lower : 2 ≤ 2 + offset := by omega
  have upper : 2 + offset ≤ maxSize := by omega
  have witness := minimum_le_witness maxSize (2 + offset) lower upper
  simpa only [one_mul] using witness.trans (Nat.mul_le_mul_left 2 upper)

theorem sentinel_absent (maxSize : Nat) : 2 * maxSize + 1 ∉ (table maxSize).drop 2 := by
  intro member
  have := suffix_bounded maxSize (2 * maxSize + 1) member
  omega

theorem answer_table (maxSize : Nat) :
    ((table maxSize).drop 2).eraseDups.sum = answer maxSize := by
  rw [table_suffix]
  rfl

theorem answer_small (maxSize : Nat) (small : maxSize < 2) : answer maxSize = 0 := by
  simp [answer, show maxSize - 1 = 0 by omega]

end ProjectEulerSolutions.ProductSumNumbers
