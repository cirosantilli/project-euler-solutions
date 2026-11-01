import ProjectEulerSolutions.Lib.LatticeOriginCount
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise

/-! Ordered exhaustive counting equals the sum over the unique right-angle vertex. -/
namespace ProjectEulerSolutions.LatticeRightTriangles
set_option Elab.async false
open scoped BigOperators

def rightWeight (p q : Int × Int) : Nat :=
  if nondegenerate p q ∧ isRight p.1 p.2 q.1 q.2 then 1 else 0

def originWeight (p q : Int × Int) : Nat :=
  if nondegenerate p q ∧ atOrigin p.1 p.2 q.1 q.2 then 1 else 0

def firstWeight (p q : Int × Int) : Nat :=
  if nondegenerate p q ∧ atFirst p.1 p.2 q.1 q.2 then 1 else 0

theorem nondegenerate_swap (p q : Int × Int) : nondegenerate q p ↔ nondegenerate p q := by
  constructor <;> rintro ⟨hp, hq, distinct⟩ <;> exact ⟨hq, hp, Ne.symm distinct⟩

theorem rightWeight_split (p q : Int × Int) :
    rightWeight p q = originWeight p q + firstWeight p q + firstWeight q p := by
  by_cases valid : nondegenerate p q
  · have pn : p.1 ≠ 0 ∨ p.2 ≠ 0 := by
      by_contra missing
      push Not at missing
      exact valid.1 (Prod.ext missing.1 missing.2)
    have qn : q.1 ≠ 0 ∨ q.2 ≠ 0 := by
      by_contra missing
      push Not at missing
      exact valid.2.1 (Prod.ext missing.1 missing.2)
    have distinct : p.1 ≠ q.1 ∨ p.2 ≠ q.2 := by
      by_contra missing
      push Not at missing
      exact valid.2.2 (Prod.ext missing.1.symm missing.2.symm)
    simpa only [rightWeight, originWeight, firstWeight, nondegenerate_swap p q, valid, true_and, atSecond, atFirst] using
      right_indicator p.1 p.2 q.1 q.2 pn qn distinct
  · simp only [rightWeight, originWeight, firstWeight, nondegenerate_swap p q, valid, false_and, ↓reduceIte,
      Nat.zero_add]

theorem firstWeight_sum (n : Nat) (p : Int × Int) :
    (∑ q ∈ grid n, firstWeight p q) = (partners n p).card := by
  rw [partners, Finset.card_eq_sum_ones, Finset.sum_filter]
  simp only [firstWeight, nondegenerate, and_assoc]

theorem originWeight_sum (n : Nat) :
    (∑ p ∈ grid n, ∑ q ∈ grid n, originWeight p q) = 2 * (n * n) := by
  rw [← originPairs_card, originPairs, Finset.card_eq_sum_ones, Finset.sum_filter, Finset.sum_product]
  rfl

def orderedCount (n : Nat) : Nat := ∑ p ∈ grid n, ∑ q ∈ grid n, rightWeight p q

/-- Exhaustive ordered counting includes two orders of each triangle, at every right-angle vertex. -/
theorem orderedCount_eq (n : Nat) :
    orderedCount n = 2 * (n * n + ∑ p ∈ grid n, (partners n p).card) := by
  have swapped : (∑ p ∈ grid n, ∑ q ∈ grid n, firstWeight q p) =
      ∑ p ∈ grid n, ∑ q ∈ grid n, firstWeight p q := Finset.sum_comm
  unfold orderedCount
  simp_rw [rightWeight_split, Finset.sum_add_distrib]
  rw [swapped, originWeight_sum]
  simp_rw [firstWeight_sum]
  omega

/-- Natural coordinates enumerate the entire integer grid, in the source loops' order. -/
theorem grid_sum (n : Nat) (f : Int × Int → Nat) :
    (∑ p ∈ grid n, f p) = ∑ x ∈ Finset.range (n + 1), ∑ y ∈ Finset.range (n + 1), f ((x : Int), (y : Int)) := by
  have equal : (∑ p ∈ Finset.range (n + 1) ×ˢ Finset.range (n + 1), f ((p.1 : Int), (p.2 : Int))) =
      ∑ p ∈ grid n, f p := by
    apply Finset.sum_bij (fun p _ => ((p.1 : Int), (p.2 : Int)))
    · intro p member
      rcases Finset.mem_product.mp member with ⟨hx, hy⟩
      rw [Finset.mem_range] at hx hy
      exact (mem_grid _ _).mpr ⟨by omega, by omega, by omega, by omega⟩
    · intro p _ q _ same
      apply Prod.ext
      · have equal : (p.1 : Int) = (q.1 : Int) := congrArg Prod.fst same
        exact_mod_cast equal
      · have equal : (p.2 : Int) = (q.2 : Int) := congrArg Prod.snd same
        exact_mod_cast equal
    · intro p member
      have bounds := (mem_grid n p).mp member
      refine ⟨(p.1.toNat, p.2.toNat), ?_, ?_⟩
      · simp only [Finset.mem_product, Finset.mem_range]
        omega
      · exact Prod.ext (Int.toNat_of_nonneg bounds.1) (Int.toNat_of_nonneg bounds.2.2.1)
    · intros
      rfl
  simpa only [Finset.sum_product] using equal.symm

def optimizedCount (n : Nat) : Nat :=
  n * n + ∑ x ∈ Finset.range (n + 1), ∑ y ∈ Finset.range (n + 1), pointContribution n x y

def referenceCount (n : Nat) : Nat :=
  (∑ a ∈ Finset.range (n + 1), ∑ b ∈ Finset.range (n + 1),
    ∑ c ∈ Finset.range (n + 1), ∑ d ∈ Finset.range (n + 1),
      rightWeight ((a : Int), (b : Int)) ((c : Int), (d : Int))) / 2

/-- Both complete mathematical counts agree for every square size, including the empty triangle domain. -/
theorem referenceCount_eq (n : Nat) : referenceCount n = optimizedCount n := by
  have partnersSum : (∑ p ∈ grid n, (partners n p).card) =
      ∑ x ∈ Finset.range (n + 1), ∑ y ∈ Finset.range (n + 1), pointContribution n x y := by
    rw [grid_sum]
    apply Finset.sum_congr rfl
    intro x hx
    apply Finset.sum_congr rfl
    intro y hy
    exact partners_card n x y (Nat.le_of_lt_succ (Finset.mem_range.mp hx)) (Nat.le_of_lt_succ (Finset.mem_range.mp hy))
  have ordered : orderedCount n =
      ∑ a ∈ Finset.range (n + 1), ∑ b ∈ Finset.range (n + 1),
        ∑ c ∈ Finset.range (n + 1), ∑ d ∈ Finset.range (n + 1),
          rightWeight ((a : Int), (b : Int)) ((c : Int), (d : Int)) := by
    unfold orderedCount
    rw [grid_sum]
    simp_rw [grid_sum]
  rw [referenceCount, ← ordered, orderedCount_eq, partnersSum]
  simp [optimizedCount]

end ProjectEulerSolutions.LatticeRightTriangles
