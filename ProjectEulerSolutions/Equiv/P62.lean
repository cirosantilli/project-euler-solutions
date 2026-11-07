import ProjectEulerSolutions.P62

namespace ProjectEulerSolutions.P62
open ProjectEulerStatements.P62 ProjectEulerStatements.UnboundedSearch

private theorem cube_positive (limit cube : Nat) (member : cube ∈ cubes limit) : 0 < cube := by
  obtain ⟨root, present, rfl⟩ := List.mem_map.mp member
  exact Nat.pow_pos (by have := (List.mem_range'_1.mp present).1; omega)

private theorem count_eq (limit cube : Nat) (positive : 0 < cube) :
    CubicGroupCounts.occurrences (CubicPermutations.signature cube) (cubes limit) =
      cubePermCount limit cube := by
  unfold CubicGroupCounts.occurrences cubePermCount
  congr 1
  apply List.filter_congr
  intro other member
  simp only [CubicPermutations.signature_eq_sorted other cube (cube_positive limit other member) positive]

theorem window_eq (limit : Nat) (target : Int) :
    windowAnswer limit target = CubicGroupCounts.choose target (CubicGroupCounts.collect (cubes limit)) := by
  rw [CubicGroupCounts.choose_collect]
  unfold windowAnswer CubicGroupCounts.eligible
  congr 2
  apply List.filterMap_congr
  intro cube member
  rw [count_eq limit cube (cube_positive limit cube member)]

theorem width_eq (target : Int) (digits : Nat) :
    widthAnswer target digits = CubicGroupCounts.choose target
      (CubicGroupCounts.collect (CubicWindows.prior (CubicWidthBounds.stop digits))) := by
  rw [widthAnswer, window_eq]
  congr 2
  simp [cubes, CubicWindows.prior, CubicSearch.cube, pow_succ]

private theorem model_finishes {target value : Int} {state : CubicSearch.State} :
    CubicSearch.Returns target state value ↔ Finishes (CubicSearch.step target) state value := by
  constructor
  · intro returned
    induction returned with
    | done checked => exact .done checked
    | next checked rest ih => exact .next checked ih
  · intro returned
    induction returned with
    | done checked => exact .done checked
    | next checked rest ih => exact .next checked ih

private theorem reference_success (target : Int) (digits current : Nat)
    (positive : 1 ≤ current) (before : current ≤ digits)
    (success : widthAnswer target digits ≠ 0)
    (earlier : ∀ width, 1 ≤ width → width < digits → widthAnswer target width = 0) :
    run (step target) current = some (widthAnswer target digits) := by
  rw [run.eq_1]
  by_cases last : current = digits
  · subst current
    simp [step, success]
  · have zero := earlier current positive (by omega)
    simp only [step, zero]
    exact reference_success target digits (current + 1) (by omega) (by omega) success earlier
termination_by digits - current

private theorem reference_has_width {target value : Int} {current : Nat}
    (positive : 1 ≤ current) (finished : Finishes (step target) current value) :
    ∃ digits, 1 ≤ digits ∧ widthAnswer target digits ≠ 0 := by
  induction finished with
  | @done current value checked =>
      refine ⟨current, positive, ?_⟩
      intro zero
      simp [step, zero] at checked
  | @next current following value checked rest ih =>
      have next : following = current + 1 := by
        unfold step at checked
        dsimp only at checked
        split at checked
        · exact (Sum.inr.inj checked).symm
        · contradiction
      apply ih
      omega

theorem equiv (target : Int) : ProjectEulerStatements.P62.naive target = solve target := by
  by_cases existsWidth : ∃ digits, 1 ≤ digits ∧ widthAnswer target digits ≠ 0
  · let digits := Nat.find existsWidth
    have positive := (Nat.find_spec existsWidth).1
    have successful := (Nat.find_spec existsWidth).2
    have earlier : ∀ width, 1 ≤ width → width < digits → widthAnswer target width = 0 := by
      intro width positive before
      by_contra success
      exact Nat.find_min existsWidth before ⟨positive, success⟩
    have ref := reference_success target digits 1 (by omega) positive successful earlier
    have returned := CubicWindows.returns_closed target (CubicWidthBounds.stop digits)
      (CubicWidthBounds.stop_positive digits) (CubicWidthBounds.closed digits positive)
      (by simpa only [← width_eq] using successful)
    have opt := (run_some_iff _ _ _).mpr (model_finishes.mp returned)
    rw [← width_eq] at opt
    exact ref.trans opt.symm
  · have ref : naive target = none := by
      cases result : naive target with
      | none => rfl
      | some value =>
          exact False.elim (existsWidth (reference_has_width (by omega)
            ((run_some_iff _ _ _).mp result)))
    have opt : solve target = none := by
      cases result : solve target with
      | none => rfl
      | some value =>
          have returned := model_finishes.mpr ((run_some_iff _ _ _).mp result)
          obtain ⟨digits, positive, success⟩ := CubicWidthBounds.returns_has_width
            (CubicWindows.invariant_start target) returned
          exact False.elim (existsWidth ⟨digits, positive, by simpa only [width_eq] using success⟩)
    exact ref.trans opt.symm

end ProjectEulerSolutions.P62
