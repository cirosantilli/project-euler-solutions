import ProjectEulerSolutions.Lib.ProductSumNumbers
import Mathlib.Data.List.Range

/-! A total factor-search trace shared by the two product-sum implementations. -/
namespace ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

def nextFactors (limit start product : Nat) : List Nat :=
  List.range' start (limit / product + 1 - start)

theorem mem_nextFactors (limit start product factor : Nat) (positive : 0 < product) :
    factor ∈ nextFactors limit start product ↔ start ≤ factor ∧ product * factor ≤ limit := by
  simp only [nextFactors, List.mem_range']
  have quotient : factor ≤ limit / product ↔ product * factor ≤ limit := by
    rw [Nat.le_div_iff_mul_le positive, Nat.mul_comm factor product]
  rw [← quotient]
  constructor
  · rintro ⟨offset, inside, rfl⟩
    omega
  · rintro ⟨lower, upper⟩
    exact ⟨factor - start, by omega, by omega⟩

/-- Preorder trace of the actual updates: each accepted node precedes its descendants. -/
def search (maxSize limit start product summ terms : Nat) : List (Nat × Nat) :=
  if _active : 0 < product ∧ 2 ≤ start then
    (nextFactors limit start product).attach.flatMap fun factor =>
      let nextProduct := product * factor.val
      let nextSum := summ + factor.val
      let nextTerms := terms + 1
      let size := nextTerms + nextProduct - nextSum
      if size ≤ maxSize then
        (size, nextProduct) :: search maxSize limit factor.val nextProduct nextSum nextTerms
      else []
  else []
termination_by limit + 1 - product
decreasing_by
  have bounds := (mem_nextFactors limit start product factor.val _active.1).mp factor.property
  exact descent limit product factor.val (by omega) (by omega) bounds.2

theorem search_inactive (maxSize limit start product summ terms : Nat)
    (inactive : ¬ (0 < product ∧ 2 ≤ start)) :
    search maxSize limit start product summ terms = [] := by
  rw [search, dif_neg inactive]

theorem mem_search (maxSize limit start product summ terms size value : Nat)
    (positive : 0 < product) (startBound : 2 ≤ start) :
    (size, value) ∈ search maxSize limit start product summ terms ↔
      ∃ factor ∈ nextFactors limit start product,
        terms + 1 + product * factor - (summ + factor) ≤ maxSize ∧
        ((size = terms + 1 + product * factor - (summ + factor) ∧ value = product * factor) ∨
          (size, value) ∈ search maxSize limit factor (product * factor) (summ + factor) (terms + 1)) := by
  rw [search, dif_pos ⟨positive, startBound⟩]
  simp only [List.mem_flatMap, List.mem_attach, true_and, Subtype.exists, exists_prop]
  apply exists_congr
  intro factor
  apply and_congr_right
  intro _
  split_ifs with accepted
  · simp [accepted, Prod.mk.injEq]
  · simp [accepted]

end ProjectEulerSolutions.ProductSumNumbers
