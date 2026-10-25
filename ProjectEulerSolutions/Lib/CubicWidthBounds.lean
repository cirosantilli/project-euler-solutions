import ProjectEulerSolutions.Lib.CubicWindows

/-! Input-derived boundaries for exhaustive searches through complete decimal widths. -/
namespace ProjectEulerSolutions.CubicWidthBounds
open CubicPermutations CubicGroups CubicGroupCounts CubicSearch CubicWindows
set_option Elab.async false

private theorem boundary_exists (digits : Nat) : ∃ n : Nat, 10 ^ digits ≤ cube n := by
  refine ⟨10 ^ digits, ?_⟩
  have positive : 0 < 10 ^ digits := Nat.pow_pos (by decide)
  have squared : 10 ^ digits ≤ 10 ^ digits * 10 ^ digits := Nat.le_mul_of_pos_right _ positive
  exact squared.trans (Nat.le_mul_of_pos_right _ positive)

/-- The first root whose cube reaches the next decimal width. -/
noncomputable def stop (digits : Nat) : Nat := Nat.find (boundary_exists digits)

theorem stop_reaches (digits : Nat) : 10 ^ digits ≤ cube (stop digits) :=
  Nat.find_spec (boundary_exists digits)

theorem before_stop (digits root : Nat) (before : root < stop digits) : cube root < 10 ^ digits :=
  Nat.lt_of_not_ge (Nat.find_min (boundary_exists digits) before)

theorem stop_le (digits root : Nat) (reached : 10 ^ digits ≤ cube root) : stop digits ≤ root :=
  Nat.find_min' (boundary_exists digits) reached

theorem stop_positive (digits : Nat) : 0 < stop digits := by
  have reached := stop_reaches digits
  have positive : 0 < 10 ^ digits := Nat.pow_pos (by decide)
  by_contra zero
  have eq : stop digits = 0 := by omega
  simp [eq, cube] at reached

theorem stop_mono {left right : Nat} (ordered : left ≤ right) : stop left ≤ stop right :=
  stop_le left (stop right) ((Nat.pow_le_pow_right (by decide : 1 ≤ 10) ordered).trans (stop_reaches right))

theorem stop_two (digits : Nat) (positive : 1 ≤ digits) : 2 ≤ stop digits := by
  have reached := stop_reaches digits
  have ten : 10 ≤ 10 ^ digits := by
    simpa using Nat.pow_le_pow_right (by decide : 1 ≤ 10) positive
  by_contra small
  have bound : cube (stop digits) ≤ 1 := by
    simpa [cube] using cube_mono (show stop digits ≤ 1 by omega)
  omega

theorem width_positive (root : Nat) (positive : 0 < root) : 1 ≤ width root := by
  have nonzero : cube root ≠ 0 := Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos positive positive) positive)
  simpa [width, decimal, nonzero] using List.length_pos_iff.mpr (Nat.digits_ne_nil_iff_ne_zero.mpr nonzero)

theorem width_le (root digits : Nat) (positive : 0 < root) (below : cube root < 10 ^ digits) : width root ≤ digits := by
  have nonzero : cube root ≠ 0 := Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos positive positive) positive)
  simpa [width, decimal, nonzero] using (Nat.digits_length_le_iff (by decide : 1 < 10) (cube root)).mpr below

theorem after_root (root : Nat) (positive : 0 < root) : root < stop (width root) := by
  have nonzero : cube root ≠ 0 := Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos positive positive) positive)
  have upper : cube root < 10 ^ width root :=
    (Nat.digits_length_le_iff (by decide : 1 < 10) (cube root)).mp (by simp [width, decimal, nonzero])
  by_contra late
  have lower := (stop_reaches (width root)).trans (cube_mono (show stop (width root) ≤ root by omega))
  omega

/-- Every completed width ends at a genuine gap between cube digit lengths. -/
theorem closed (digits : Nat) (positive : 1 ≤ digits) : width (stop digits - 1) < width (stop digits) := by
  have two := stop_two digits positive
  have lower : width (stop digits - 1) ≤ digits :=
    width_le _ _ (by omega) (before_stop digits _ (by omega))
  have upper : digits < width (stop digits) := by
    have nonzero : cube (stop digits) ≠ 0 := by
      have pos := stop_positive digits
      exact Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos pos pos) pos)
    simpa [width, decimal, nonzero] using
      (Nat.lt_digits_length_iff (by decide : 1 < 10) (cube (stop digits))).mpr (stop_reaches digits)
  omega

private theorem choice_positive (target : Int) (inputs : List Nat) (groups : List Group)
    (summary : Summary inputs groups) (positive : ∀ value ∈ inputs, 0 < value)
    (nonempty : matching target groups ≠ []) : 0 < choose target groups := by
  have member : choose target groups ∈ matching target groups := by
    cases selected : matching target groups with
    | nil => exact False.elim (nonempty selected)
    | cons first rest =>
        simpa only [choose, selected, List.min?_cons', Option.getD_some] using
          (List.min?_mem (xs := first :: rest) rfl)
  obtain ⟨value, member, _, same⟩ := (mem_eligible target _ inputs).mp
    (summary.eligible_of_matching target _ member)
  exact same ▸ Int.natCast_pos.mpr (positive value member)

/-- Any finite model return implies a successful complete width, without a global search bound. -/
theorem returns_has_width {target value : Int} {state : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) (returned : Returns target state value) :
    ∃ digits : Nat, 1 ≤ digits ∧ choose target (collect (prior (stop digits))) ≠ 0 := by
  induction returned generalizing retired with
  | @done state value checked =>
      have growing : state.currentWidth < width state.next := by
        by_contra absent
        simp [step, absent] at checked
      have nonempty : matching target state.groups ≠ [] := by
        intro empty
        simp [step, growing, empty] at checked
      have allNonempty : matching target (retired ++ state.groups) ≠ [] := by
        simp [matching_append, invariant.retiredEmpty, nonempty]
      have pickedPositive := choice_positive target (prior state.next) (retired ++ state.groups)
        invariant.summary (by
          intro cube member
          obtain ⟨root, positive, _, rfl⟩ := mem_prior.mp member
          exact Nat.mul_pos (Nat.mul_pos positive positive) positive) allNonempty
      have equal := invariant.return_value (Nat.le_refl state.next) checked
      have same : choose target (collect (prior state.next)) = choose target (retired ++ state.groups) :=
        (invariant.summary.choice_unique (collect_summary _) target).symm
      rw [same] at equal
      have resultPositive : 0 < value := equal ▸ pickedPositive
      refine ⟨width state.next, width_positive _ invariant.positive, ?_⟩
      rw [← invariant.return_value (after_root _ invariant.positive).le checked]
      omega
  | next checked rest ih =>
      obtain ⟨retired', nextInvariant, _⟩ := invariant.continues checked
      exact ih nextInvariant

end ProjectEulerSolutions.CubicWidthBounds
