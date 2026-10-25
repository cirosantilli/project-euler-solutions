import ProjectEulerSolutions.Lib.PokerStraights
import lang.py.FormalSW.ReverseSortingCalls
import Batteries.Data.List.Perm

namespace ProjectEulerSolutions.PokerUnique
open FormalSW.PyAst

def addUnique (values : List Int) (value : Int) : List Int :=
  if value ∈ values then values else values ++ [value]

def unique (values : List Int) : List Int := values.foldl addUnique []

def descending (values : List Int) : List Int := HeapCalls.sortedInts true values

def orderedUnique (values : List Int) : List Int := ScalarSort.ints (unique (descending values))

theorem mem_addUnique (values : List Int) (value query : Int) :
    query ∈ addUnique values value ↔ query = value ∨ query ∈ values := by
  by_cases present : value ∈ values
  · rw [addUnique, if_pos present]
    constructor
    · intro member; exact .inr member
    · intro member
      cases member with
      | inl same => subst query; exact present
      | inr old => exact old
  · rw [addUnique, if_neg present]
    simp only [List.mem_append, List.mem_singleton]
    exact or_comm

theorem addUnique_nodup (values : List Int) (value : Int) (distinct : values.Nodup) :
    (addUnique values value).Nodup := by
  by_cases present : value ∈ values
  · simpa only [addUnique, if_pos present] using distinct
  · rw [addUnique, if_neg present]
    apply List.nodup_append.mpr
    refine ⟨distinct, by simp, ?_⟩
    intro left member right singleton same
    have only : right = value := List.mem_singleton.mp singleton
    exact present ((same.trans only) ▸ member)

private theorem nodup_fold (values initial : List Int) (distinct : initial.Nodup) :
    (values.foldl addUnique initial).Nodup := by
  induction values generalizing initial with
  | nil => exact distinct
  | cons value rest ih =>
      rw [List.foldl_cons]
      exact ih (addUnique initial value) (addUnique_nodup initial value distinct)

theorem unique_nodup (values : List Int) : (unique values).Nodup :=
  nodup_fold values [] (by simp)

private theorem mem_fold (values initial : List Int) (query : Int) :
    query ∈ values.foldl addUnique initial ↔ query ∈ values ∨ query ∈ initial := by
  induction values generalizing initial with
  | nil => simp
  | cons value rest ih =>
      rw [List.foldl_cons, ih, mem_addUnique]
      simp only [List.mem_cons, or_assoc, or_left_comm]

theorem mem_unique (values : List Int) (query : Int) : query ∈ unique values ↔ query ∈ values := by
  simpa [unique] using mem_fold values [] query

theorem descending_perm (values : List Int) : (descending values).Perm values := by
  change (ScalarSort.ints values.reverse).reverse.Perm values
  exact (List.reverse_perm _).trans
    ((ScalarSort.ints_perm values.reverse).trans (List.reverse_perm values))

theorem mem_orderedUnique (values : List Int) (query : Int) :
    query ∈ orderedUnique values ↔ query ∈ values :=
  (ScalarSort.ints_perm (unique (descending values))).mem_iff.trans
    ((mem_unique (descending values) query).trans (descending_perm values).mem_iff)

theorem orderedUnique_nodup (values : List Int) : (orderedUnique values).Nodup :=
  (ScalarSort.ints_perm (unique (descending values))).nodup_iff.mpr (unique_nodup (descending values))

theorem orderedUnique_ordered (values : List Int) :
    (orderedUnique values).Pairwise (fun left right : Int => left ≤ right) :=
  ScalarSort.ints_ordered (unique (descending values))

/-- Sorting unique ranks strengthens nondecreasing order to strict order. -/
theorem orderedUnique_strict (values : List Int) :
    (orderedUnique values).Pairwise (fun left right : Int => left < right) := by
  have combined : (orderedUnique values).Pairwise (fun left right : Int => left ≤ right ∧ left ≠ right) :=
    List.pairwise_and_iff.mpr ⟨orderedUnique_ordered values, orderedUnique_nodup values⟩
  apply combined.imp
  intro left right conditions
  omega

theorem orderedUnique_length_le (values : List Int) : (orderedUnique values).length ≤ values.length :=
  (List.subperm_of_subset (orderedUnique_nodup values) (fun query member => (mem_orderedUnique values query).mp member)).length_le

theorem orderedUnique_perm_of_length (values : List Int)
    (sameLength : (orderedUnique values).length = values.length) : (orderedUnique values).Perm values :=
  (List.subperm_of_subset (orderedUnique_nodup values)
    (fun query member => (mem_orderedUnique values query).mp member)).perm_of_length_le (by omega)

theorem orderedUnique_length_eq_iff (values : List Int) :
    (orderedUnique values).length = values.length ↔ values.Nodup := by
  constructor
  · intro sameLength
    exact (orderedUnique_perm_of_length values sameLength).nodup (orderedUnique_nodup values)
  · intro distinct
    have same : (orderedUnique values).Perm values :=
      (List.perm_ext_iff_of_nodup (orderedUnique_nodup values) distinct).mpr (mem_orderedUnique values)
    exact same.length_eq

/-- A unique-rank collection of unchanged length loses no ranks, so both ascending sorts agree. -/
theorem sorted_eq_orderedUnique_of_length (values : List Int)
    (sameLength : (orderedUnique values).length = values.length) :
    ScalarSort.ints values = orderedUnique values := by
  have same : (ScalarSort.ints values).Perm (orderedUnique values) :=
    (ScalarSort.ints_perm values).trans (orderedUnique_perm_of_length values sameLength).symm
  exact List.Perm.eq_of_pairwise (le := fun left right : Int => left ≤ right)
    (fun left right _ _ forward backward => by omega)
    (ScalarSort.ints_ordered values) (orderedUnique_ordered values) same

theorem sorted_eq_orderedUnique_five (values : List Int) (five : values.length = 5)
    (fiveUnique : (orderedUnique values).length = 5) : ScalarSort.ints values = orderedUnique values :=
  sorted_eq_orderedUnique_of_length values (by omega)

/-- Fewer than five unique ranks in a five-card list rules out every reference straight. -/
theorem reference_not_straight_of_short_unique (values : List Int) (five : values.length = 5)
    (short : (orderedUnique values).length < 5) :
    PokerStraights.referenceStraight (ScalarSort.ints values) = false := by
  apply PokerStraights.not_straight_of_duplicates
  intro sortedDistinct
  have distinct : values.Nodup := (ScalarSort.ints_perm values).nodup sortedDistinct
  have sameLength := (orderedUnique_length_eq_iff values).mpr distinct
  omega

theorem five_referenceStraight (values : List Int) (five : values.length = 5) :
    PokerStraights.referenceStraight (ScalarSort.ints values) =
      if (orderedUnique values).length = 5 then PokerStraights.referenceStraight (orderedUnique values) else false := by
  by_cases complete : (orderedUnique values).length = 5
  · rw [if_pos complete, sorted_eq_orderedUnique_five values five complete]
  · rw [if_neg complete]
    apply reference_not_straight_of_short_unique values five
    have bounded := orderedUnique_length_le values
    omega

end ProjectEulerSolutions.PokerUnique
