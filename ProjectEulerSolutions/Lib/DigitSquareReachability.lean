import ProjectEulerSolutions.Lib.DigitSquareChains

namespace ProjectEulerSolutions.DigitSquareChains
open ProjectEulerStatements.BoundedOrbit
set_option Elab.async false

@[simp] theorem nextTerm_one : nextTerm 1 = 1 := rfl

@[simp] theorem endsAt89_zero : endsAt89 0 = false := by
  apply Bool.eq_false_iff.mpr
  intro hit
  obtain ⟨steps, reached⟩ := (endsAt89_iff 0).mp hit
  have fixed := Function.iterate_fixed nextTerm_zero steps
  omega

@[simp] theorem endsAt89_one : endsAt89 1 = false := by
  apply Bool.eq_false_iff.mpr
  intro hit
  obtain ⟨steps, reached⟩ := (endsAt89_iff 1).mp hit
  have fixed := Function.iterate_fixed nextTerm_one steps
  omega

@[simp] theorem endsAt89_target : endsAt89 89 = true :=
  (endsAt89_iff 89).mpr ⟨0, rfl⟩

/-- The target itself lies on a cycle, so advancing from it preserves target reachability. -/
theorem target_returns : ∃ steps : Nat, nextTerm^[steps] (nextTerm 89) = 89 := by
  have a : nextTerm 89 = 145 := rfl
  have b : nextTerm 145 = 42 := rfl
  have c : nextTerm 42 = 20 := rfl
  have d : nextTerm 20 = 4 := rfl
  have e : nextTerm 4 = 16 := rfl
  have f : nextTerm 16 = 37 := rfl
  have g : nextTerm 37 = 58 := rfl
  have h : nextTerm 58 = 89 := rfl
  refine ⟨7, ?_⟩
  change nextTerm (nextTerm (nextTerm (nextTerm (nextTerm (nextTerm (nextTerm (nextTerm 89))))))) = 89
  rw [a, b, c, d, e, f, g, h]

/-- Grouping starts by their first digit-square sum preserves the classification. -/
theorem endsAt89_next (n : Nat) : endsAt89 (nextTerm n) = endsAt89 n := by
  apply Bool.eq_iff_iff.mpr
  rw [endsAt89_iff, endsAt89_iff]
  constructor
  · rintro ⟨steps, reached⟩
    exact ⟨steps + 1, by simpa only [Function.iterate_succ_apply] using reached⟩
  · rintro ⟨steps, reached⟩
    cases steps with
    | zero =>
        have same : n = 89 := reached
        simpa only [same] using target_returns
    | succ steps =>
        exact ⟨steps, by simpa only [Function.iterate_succ_apply] using reached⟩

/-- A visited prefix avoids the target, and each recorded successor is either recorded or current. -/
structure Frontier (current : Nat) (seen : List Nat) : Prop where
  avoids : 89 ∉ seen
  links : ∀ value ∈ seen, nextTerm value ∈ seen ∨ nextTerm value = current

theorem Frontier.empty (current : Nat) : Frontier current [] := by
  constructor <;> simp

theorem Frontier.advance {current : Nat} {seen : List Nat}
    (frontier : Frontier current seen) (notTarget : current ≠ 89) :
    Frontier (nextTerm current) (seen ++ [current]) := by
  constructor
  · simpa only [List.mem_append, List.mem_singleton, not_or] using
      And.intro frontier.avoids (Ne.symm notTarget)
  · intro value member
    rcases List.mem_append.mp member with previous | last
    · rcases frontier.links value previous with earlier | here
      · exact Or.inl (List.mem_append_left _ earlier)
      · exact Or.inl (by simp [here])
    · exact Or.inr (by simp only [List.mem_singleton.mp last])

/-- Once the frontier repeats, its whole forward orbit stays among the visited non-target values. -/
theorem Frontier.repeated {current : Nat} {seen : List Nat}
    (frontier : Frontier current seen) (repeated : current ∈ seen) : endsAt89 current = false := by
  have closed : ∀ value ∈ seen, nextTerm value ∈ seen := by
    intro value member
    rcases frontier.links value member with previous | here
    · exact previous
    · simpa only [here] using repeated
  have iterateInside : ∀ steps : Nat, nextTerm^[steps] current ∈ seen := by
    intro steps
    induction steps with
    | zero => exact repeated
    | succ steps ih => simpa only [Function.iterate_succ_apply'] using closed _ ih
  apply Bool.eq_false_iff.mpr
  intro hit
  obtain ⟨steps, reached⟩ := (endsAt89_iff current).mp hit
  exact frontier.avoids (reached ▸ iterateInside steps)

/-- Exact control flow of the reference's endpoint-or-repeat loop. -/
inductive ReferenceTrace : Nat → List Nat → Bool → Prop
  | done {current seen} : (current = 1 ∨ current = 89 ∨ current ∈ seen) →
      ReferenceTrace current seen (decide (current = 89))
  | step {current seen answer} : current ≠ 1 → current ≠ 89 → current ∉ seen →
      ReferenceTrace (nextTerm current) (seen ++ [current]) answer →
      ReferenceTrace current seen answer

/-- The finite invariant supplies a decreasing measure, never an exhausted-search branch. -/
theorem reference_trace (start current : Nat) (seen : List Nat)
    (bounded : current ≤ bound start) (distinct : seen.Nodup) (inside : Within (bound start) seen)
    (frontier : Frontier current seen) : ReferenceTrace current seen (endsAt89 current) := by
  by_cases stopped : current = 1 ∨ current = 89 ∨ current ∈ seen
  · have answer : endsAt89 current = decide (current = 89) := by
      rcases stopped with one | target | repeated
      · subst current; simp
      · subst current; simp
      · have notTarget : current ≠ 89 := by intro same; exact frontier.avoids (same ▸ repeated)
        simp only [frontier.repeated repeated, notTarget, decide_false]
    rw [answer]
    exact .done stopped
  · have notOne : current ≠ 1 := by tauto
    have notTarget : current ≠ 89 := by tauto
    have fresh : current ∉ seen := by tauto
    apply ReferenceTrace.step notOne notTarget fresh
    rw [← endsAt89_next current]
    exact reference_trace start (nextTerm current) (seen ++ [current])
      (nextTerm_bounded start current bounded) (append_distinct seen current distinct fresh)
      (append_within (bound start) seen current inside bounded) (frontier.advance notTarget)
termination_by bound start + 1 - seen.length
decreasing_by
  have enough := room (bound start) current seen distinct inside bounded fresh
  simp only [List.length_append, List.length_singleton]
  omega

theorem initial_reference_trace (start : Nat) : ReferenceTrace start [] (endsAt89 start) :=
  reference_trace start start [] (Nat.le_max_left _ _) (by simp) (by simp [Within]) (Frontier.empty start)

end ProjectEulerSolutions.DigitSquareChains
