import ProjectEulerSolutions.Lib.CubicWindows

/-! An exact count for one digit class supplies an unbounded-search termination witness. -/
namespace ProjectEulerSolutions.CubicClassWitness
open CubicPermutations CubicGroups CubicGroupCounts CubicSearch CubicWindows
set_option Elab.async false

def rootsWith (digits : List Nat) (first count : Nat) : List Nat :=
  (List.range' first count).filter (fun root => decide (signature (cube root) = digits))

theorem rootsWith_append (digits : List Nat) (first left right : Nat) :
    rootsWith digits first (left + right) =
      rootsWith digits first left ++ rootsWith digits (first + left) right := by
  simp only [rootsWith, ← List.range'_append_1, List.filter_append]

theorem rootsWith_join (digits : List Nat) (first left right : Nat)
    (before after : List Nat)
    (leftChecked : rootsWith digits first left = before)
    (rightChecked : rootsWith digits (first + left) right = after) :
    rootsWith digits first (left + right) = before ++ after := by
  rw [rootsWith_append, leftChecked, rightChecked]

theorem occurrences_prior (digits : List Nat) (stop : Nat) :
    occurrences digits (prior stop) = (rootsWith digits 1 (stop - 1)).length := by
  simp only [occurrences, prior, List.filter_map, List.length_map, rootsWith,
    Function.comp_def]

/-- A class containing a processed positive cube makes the selected result positive. -/
theorem choice_positive (stop root : Nat) (target : Int)
    (positive : 0 < root) (before : root < stop)
    (counted : ((rootsWith (signature (cube root)) 1 (stop - 1)).length : Int) = target) :
    0 < choose target (collect (prior stop)) := by
  have summary := collect_summary (prior stop)
  have eligible : (cube root : Int) ∈ eligible target (prior stop) := by
    apply (mem_eligible _ _ _).mpr
    refine ⟨cube root, mem_prior.mpr ⟨root, positive, before, rfl⟩, ?_, rfl⟩
    simpa only [occurrences_prior] using counted
  obtain ⟨candidate, member, _⟩ := summary.matching_below_eligible target _ eligible
  have nonempty : matching target (collect (prior stop)) ≠ [] := by
    intro empty
    simp [empty] at member
  have selected : choose target (collect (prior stop)) ∈
      matching target (collect (prior stop)) := by
    cases found : matching target (collect (prior stop)) with
    | nil => exact False.elim (nonempty found)
    | cons first rest =>
        simpa only [choose, found, List.min?_cons', Option.getD_some] using
          (List.min?_mem (xs := first :: rest) rfl)
  obtain ⟨value, present, _, same⟩ := (mem_eligible _ _ _).mp
    (summary.eligible_of_matching target _ selected)
  obtain ⟨n, nPositive, _, rfl⟩ := mem_prior.mp present
  rw [← same]
  exact Int.natCast_pos.mpr (Nat.mul_pos (Nat.mul_pos nPositive nPositive) nPositive)

/-- The width is closed mathematically; it does not become a parameter of either source. -/
theorem returns_of_count (stop root : Nat) (target : Int)
    (positive : 0 < root) (before : root < stop)
    (closed : width (stop - 1) < width stop)
    (counted : ((rootsWith (signature (cube root)) 1 (stop - 1)).length : Int) = target) :
    Returns target start (choose target (collect (prior stop))) :=
  returns_closed target stop (by omega) closed
    (ne_of_gt (choice_positive stop root target positive before counted))

end ProjectEulerSolutions.CubicClassWitness
