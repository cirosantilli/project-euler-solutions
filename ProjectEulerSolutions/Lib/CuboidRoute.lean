import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Finset.Prod
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Tactic.Linarith

/-! Cuboid surface routes and exact counting by the sum of the two smaller sides. -/
namespace ProjectEulerSolutions.CuboidRoute

set_option Elab.async false
open scoped BigOperators

def square (n : Nat) : Prop := Nat.sqrt n * Nat.sqrt n = n
instance (n : Nat) : Decidable (square n) := inferInstanceAs (Decidable (_ = _))

theorem square_iff (n : Nat) : square n ↔ ∃ k : Nat, k * k = n := by
  exact (Nat.exists_mul_self n).symm

/-- Sorting the side lengths makes this unfolding no longer than either alternative. -/
theorem shortest_unfolding (a b c : Nat) (ab : a ≤ b) (bc : b ≤ c) :
    (a + b) ^ 2 + c ^ 2 ≤ (a + c) ^ 2 + b ^ 2 ∧
    (a + b) ^ 2 + c ^ 2 ≤ (b + c) ^ 2 + a ^ 2 := by
  constructor
  · nlinarith [Nat.mul_le_mul_left (2 * a) bc]
  · nlinarith [Nat.mul_le_mul_left (2 * b) (ab.trans bc)]

def pairs (c s : Nat) : Finset (Nat × Nat) :=
  ((Finset.Icc 1 c) ×ˢ (Finset.Icc 1 c)).filter fun p => p.1 ≤ p.2 ∧ p.1 + p.2 = s

theorem mem_pairs (c s : Nat) (p : Nat × Nat) :
    p ∈ pairs c s ↔ 1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ c ∧ p.1 + p.2 = s := by
  rcases p with ⟨a, b⟩
  simp only [pairs, Finset.mem_filter, Finset.mem_product, Finset.mem_Icc]
  omega

def pairCount (c s : Nat) : Nat := s / 2 + 1 - max 1 (s - c)

/-- Choosing the smaller side in this exact interval is a bijection with ordered side pairs. -/
theorem pairs_card (c s : Nat) : (pairs c s).card = pairCount c s := by
  have equal : (Finset.Icc (max 1 (s - c)) (s / 2)).card = (pairs c s).card := by
    apply Finset.card_bij (fun a _ => (a, s - a))
    · intro a member
      rw [mem_pairs]
      simp only [Finset.mem_Icc] at member
      dsimp
      omega
    · intro a ha b hb same
      exact congrArg Prod.fst same
    · intro p member
      rw [mem_pairs] at member
      refine ⟨p.1, ?_, ?_⟩
      · simp only [Finset.mem_Icc]
        omega
      · exact Prod.ext rfl (by dsimp; omega)
  rw [Nat.card_Icc] at equal
  exact equal.symm

theorem pair_interval_nonempty {c s : Nat} (lower : 2 ≤ s) (upper : s ≤ 2 * c) :
    max 1 (s - c) ≤ s / 2 := by omega

theorem pairCount_int_formula {c s : Nat} (lower : 2 ≤ s) (upper : s ≤ 2 * c) :
    (pairCount c s : Int) = (s : Int) / 2 - max 1 ((s : Int) - c) + 1 := by
  have nonempty := pair_interval_nonempty lower upper
  unfold pairCount
  omega

def cuboids (c : Nat) : Finset (Nat × Nat) :=
  ((Finset.Icc 1 c) ×ˢ (Finset.Icc 1 c)).filter fun p =>
    p.1 ≤ p.2 ∧ square ((p.1 + p.2) ^ 2 + c ^ 2)

def exhaustiveLayer (c : Nat) : Nat := (cuboids c).card

def groupedLayer (c : Nat) : Nat :=
  ∑ s ∈ Finset.Icc 2 (2 * c), if square (s ^ 2 + c ^ 2) then pairCount c s else 0

theorem mem_cuboids (c : Nat) (p : Nat × Nat) :
    p ∈ cuboids c ↔ 1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ c ∧ square ((p.1 + p.2) ^ 2 + c ^ 2) := by
  rcases p with ⟨a, b⟩
  simp only [cuboids, Finset.mem_filter, Finset.mem_product, Finset.mem_Icc]
  constructor
  · intro h
    exact ⟨h.1.1.1, h.2.1, h.1.2.2, h.2.2⟩
  · rintro ⟨first, ordered, bounded, good⟩
    exact ⟨⟨⟨first, ordered.trans bounded⟩, ⟨first.trans ordered, bounded⟩⟩, ordered, good⟩

theorem fiber_card (c s : Nat) :
    ((cuboids c).filter fun p => p.1 + p.2 = s).card =
      if square (s ^ 2 + c ^ 2) then pairCount c s else 0 := by
  by_cases accepted : square (s ^ 2 + c ^ 2)
  · rw [if_pos accepted, ← pairs_card]
    congr 1
    ext p
    simp only [Finset.mem_filter, mem_cuboids, mem_pairs]
    constructor
    · rintro ⟨⟨first, ordered, bounded, good⟩, sum⟩
      exact ⟨first, ordered, bounded, sum⟩
    · rintro ⟨first, ordered, bounded, sum⟩
      exact ⟨⟨first, ordered, bounded, by simpa only [sum] using accepted⟩, sum⟩
  · rw [if_neg accepted, Finset.card_eq_zero]
    apply Finset.eq_empty_iff_forall_notMem.mpr
    intro p member
    simp only [Finset.mem_filter, mem_cuboids] at member
    exact accepted (by simpa only [member.2] using member.1.2.2.2)

/-- Grouping by a+b preserves the complete exhaustive cuboid count. -/
theorem layer_equivalence (c : Nat) : exhaustiveLayer c = groupedLayer c := by
  have grouped := Finset.card_eq_sum_card_fiberwise
    (f := fun p : Nat × Nat => p.1 + p.2) (s := cuboids c) (t := Finset.Icc 2 (2 * c))
    (by
      intro p member
      have bounds := (mem_cuboids c p).mp member
      simp only [Finset.mem_coe, Finset.mem_Icc]
      omega)
  simpa only [exhaustiveLayer, groupedLayer, fiber_card] using grouped

/-- The ordered nested loops count exactly the finite set of cuboids in one layer. -/
theorem exhaustiveLayer_nested (c : Nat) : exhaustiveLayer c =
    ∑ a ∈ Finset.Icc 1 c, ∑ b ∈ Finset.Icc a c,
      if square ((a + b) ^ 2 + c ^ 2) then 1 else 0 := by
  simp only [exhaustiveLayer, cuboids, Finset.card_eq_sum_ones, Finset.sum_filter, Finset.sum_product]
  apply Finset.sum_congr rfl
  intro a member
  have positive : 1 ≤ a := (Finset.mem_Icc.mp member).1
  have interval : (Finset.Icc 1 c).filter (fun b => a ≤ b) = Finset.Icc a c := by
    ext b
    simp only [Finset.mem_filter, Finset.mem_Icc]
    omega
  rw [← interval, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro b bounded
  by_cases ordered : a ≤ b <;> by_cases good : square ((a + b) ^ 2 + c ^ 2) <;>
    simp [ordered, good]

end ProjectEulerSolutions.CuboidRoute
