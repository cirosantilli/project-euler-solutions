import lang.py.FormalSW.Permutations
import Mathlib.Data.List.Nodup
import Mathlib.Data.List.Perm.Subperm
import Mathlib.Algebra.BigOperators.Group.List.Basic

/-! A right-to-left search enumerates constrained permutations without duplicates. -/
namespace ProjectEulerSolutions.ConstraintRows
open FormalSW.PyAst

set_option Elab.async false

variable {α : Type} [DecidableEq α]

/-- Prepend an unused pool entry satisfying the next constraint. -/
def prepend (pool : List α) (suffixes : List (List α)) (test : α → List α → Bool) :
    List (List α) :=
  suffixes.flatMap fun suffix =>
    (pool.filter fun digit => decide (digit ∉ suffix) && test digit suffix).map (· :: suffix)

@[simp] theorem mem_prepend (pool : List α) (suffixes : List (List α))
    (test : α → List α → Bool) (row : List α) :
    row ∈ prepend pool suffixes test ↔
      ∃ digit suffix, digit ∈ pool ∧ suffix ∈ suffixes ∧ digit ∉ suffix ∧
        test digit suffix = true ∧ row = digit :: suffix := by
  simp only [prepend, List.mem_flatMap, List.mem_map, List.mem_filter,
    Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨suffix, member, digit, ⟨inside, fresh, passed⟩, rfl⟩
    exact ⟨digit, suffix, inside, member, fresh, passed, rfl⟩
  · rintro ⟨digit, suffix, inside, member, fresh, passed, rfl⟩
    exact ⟨suffix, member, digit, ⟨inside, fresh, passed⟩, rfl⟩

theorem prepend_nodup (pool : List α) (suffixes : List (List α))
    (test : α → List α → Bool) (poolNodup : pool.Nodup) (suffixesNodup : suffixes.Nodup) :
    (prepend pool suffixes test).Nodup := by
  apply List.nodup_flatMap.mpr
  constructor
  · intro suffix _
    exact (poolNodup.filter _).map (fun _ _ equal => (List.cons.inj equal).1)
  · refine suffixesNodup.pairwise_of_forall_ne fun first _ second _ different => ?_
    change List.Disjoint
      ((pool.filter fun digit => decide (digit ∉ first) && test digit first).map (· :: first))
      ((pool.filter fun digit => decide (digit ∉ second) && test digit second).map (· :: second))
    apply List.disjoint_left.mpr
    intro row firstMember secondMember
    obtain ⟨a, _, firstEqual⟩ := List.mem_map.mp firstMember
    obtain ⟨b, _, secondEqual⟩ := List.mem_map.mp secondMember
    exact different (List.cons.inj (firstEqual.trans secondEqual.symm)).2

/-- The first test checks the first digit, and may inspect its entire suffix. -/
def search (pool : List α) : List (α → List α → Bool) → List (List α)
  | [] => [[]]
  | test :: tests => prepend pool (search pool tests) test

/-- Length mismatches are rejected, including extra digits after the last test. -/
def accepts : List (α → List α → Bool) → List α → Bool
  | [], [] => true
  | test :: tests, digit :: suffix => test digit suffix && accepts tests suffix
  | _, _ => false

omit [DecidableEq α] in
theorem accepts_length (tests : List (α → List α → Bool)) (row : List α)
    (accepted : accepts tests row = true) : row.length = tests.length := by
  induction tests generalizing row with
  | nil => cases row <;> simp_all [accepts]
  | cons test tests ih =>
      cases row with
      | nil => simp [accepts] at accepted
      | cons digit suffix =>
          have passed : test digit suffix = true ∧ accepts tests suffix = true := by
            simpa only [accepts, Bool.and_eq_true] using accepted
          have tail := ih suffix passed.2
          simp [tail]

theorem mem_search (pool : List α) (tests : List (α → List α → Bool)) (row : List α) :
    row ∈ search pool tests ↔ row.Nodup ∧ row ⊆ pool ∧ accepts tests row = true := by
  induction tests generalizing row with
  | nil =>
      cases row <;> simp [search, accepts]
  | cons test tests ih =>
      cases row with
      | nil => simp [search, accepts]
      | cons digit suffix =>
          simp only [search, mem_prepend]
          constructor
          · rintro ⟨head, tail, inside, member, fresh, passed, equal⟩
            have identified := List.cons.inj equal
            rcases identified with ⟨rfl, rfl⟩
            obtain ⟨nodup, subset, accepted⟩ := (ih suffix).mp member
            exact ⟨List.nodup_cons.mpr ⟨fresh, nodup⟩,
              List.cons_subset.mpr ⟨inside, subset⟩, by simp [accepts, passed, accepted]⟩
          · rintro ⟨nodup, subset, accepted⟩
            obtain ⟨fresh, tailNodup⟩ := List.nodup_cons.mp nodup
            obtain ⟨inside, tailSubset⟩ := List.cons_subset.mp subset
            have passed : test digit suffix = true ∧ accepts tests suffix = true := by
              simpa only [accepts, Bool.and_eq_true] using accepted
            exact ⟨digit, suffix, inside, (ih suffix).mpr ⟨tailNodup, tailSubset, passed.2⟩,
              fresh, passed.1, rfl⟩

theorem search_nodup (pool : List α) (tests : List (α → List α → Bool))
    (poolNodup : pool.Nodup) : (search pool tests).Nodup := by
  induction tests with
  | nil => simp [search]
  | cons test tests ih => exact prepend_nodup pool _ test poolNodup ih

omit [DecidableEq α] in
/-- Distinct positional choices cannot generate the same row from a duplicate-free pool. -/
theorem permutation_rows_nodup (length : Nat) (pool : List α) (poolNodup : pool.Nodup) :
    (Permutations.rows length pool).Nodup := by
  have heads (items : List α) : (Permutations.choices items).map Prod.fst = items := by
    induction items with
    | nil => rfl
    | cons first rest ih => simp [Permutations.choices, List.map_map, Function.comp_def, ih]
  induction length generalizing pool with
  | zero => simp [Permutations.rows]
  | succ length ih =>
      apply List.nodup_flatMap.mpr
      constructor
      · intro pair member
        have remainingNodup := ((Permutations.choice_perm member).nodup_iff.mpr poolNodup).of_cons
        exact (ih pair.2 remainingNodup).map (fun _ _ equal => (List.cons.inj equal).2)
      · have differentHeads : (Permutations.choices pool).Pairwise (fun a b => a.1 ≠ b.1) := by
          have mapped := poolNodup
          rw [← heads pool] at mapped
          exact List.pairwise_map.mp mapped
        apply differentHeads.imp
        intro first second different
        change List.Disjoint
          ((Permutations.rows length first.2).map (first.1 :: ·))
          ((Permutations.rows length second.2).map (second.1 :: ·))
        apply List.disjoint_left.mpr
        intro row firstMember secondMember
        obtain ⟨a, _, firstEqual⟩ := List.mem_map.mp firstMember
        obtain ⟨b, _, secondEqual⟩ := List.mem_map.mp secondMember
        exact different (List.cons.inj (firstEqual.trans secondEqual.symm)).1

/-- A complete constrained search agrees with exhaustive positional enumeration, including multiplicity. -/
theorem search_perm_filter (pool : List α) (tests : List (α → List α → Bool))
    (poolNodup : pool.Nodup) (width : tests.length = pool.length) :
    (search pool tests).Perm ((Permutations.rows pool.length pool).filter (accepts tests)) := by
  apply (List.perm_ext_iff_of_nodup (search_nodup pool tests poolNodup)
    ((permutation_rows_nodup pool.length pool poolNodup).filter _)).mpr
  intro row
  rw [mem_search, List.mem_filter, Permutations.mem_rows_full_iff]
  constructor
  · rintro ⟨nodup, subset, accepted⟩
    exact ⟨(nodup.subperm subset).perm_of_length_le
      (by rw [accepts_length tests row accepted, width]; exact Nat.le_refl _),
      accepted⟩
  · rintro ⟨permuted, accepted⟩
    exact ⟨permuted.nodup_iff.mpr poolNodup, fun _ member => permuted.mem_iff.mp member, accepted⟩

/-- Summing does not depend on the enumeration order used by the search. -/
theorem search_sum_eq {β : Type} [AddCommMonoid β] (pool : List α)
    (tests : List (α → List α → Bool)) (poolNodup : pool.Nodup)
    (width : tests.length = pool.length) (value : List α → β) :
    ((search pool tests).map value).sum =
      (((Permutations.rows pool.length pool).filter (accepts tests)).map value).sum :=
  ((search_perm_filter pool tests poolNodup width).map value).sum_eq

end ProjectEulerSolutions.ConstraintRows
