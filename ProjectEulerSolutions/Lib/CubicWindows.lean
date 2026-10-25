import ProjectEulerSolutions.Lib.CubicGroupCounts
import ProjectEulerSolutions.Lib.CubicSearch

/-! Digit-width resets retain a summary of every cube already visited by the search. -/
namespace ProjectEulerSolutions.CubicWindows
open CubicPermutations CubicGroups CubicGroupCounts CubicSearch
set_option Elab.async false

theorem histogram_sum (digits : List Nat) (bounded : ∀ digit ∈ digits, digit < 10) :
    (DigitHistogram.counts digits).sum = digits.length := by
  induction digits with
  | nil => decide
  | cons digit rest ih =>
      have small := bounded digit (by simp)
      have tail := ih (fun value member => bounded value (by simp [member]))
      interval_cases digit <;>
        simp [DigitHistogram.counts, List.range_succ, List.count_cons] at tail ⊢ <;> omega

theorem signature_sum (n : Nat) : (signature n).sum = (decimal n).length :=
  histogram_sum _ (decimal_bounded n)

def groupWidth (group : Group) : Nat := group.digits.sum

@[simp] theorem fresh_width (n : Nat) : groupWidth (fresh (cube n)) = width n :=
  signature_sum _

@[simp] theorem update_width (n : Nat) (group : Group) :
    groupWidth (update (cube n) group) = groupWidth group := by
  simp [groupWidth]

theorem cube_mono {left right : Nat} (ordered : left ≤ right) : cube left ≤ cube right :=
  Nat.mul_le_mul (Nat.mul_le_mul ordered ordered) ordered

theorem cube_strict {left right : Nat} (ordered : left < right) : cube left < cube right := by
  have positive : 0 < right := by omega
  exact (Nat.mul_le_mul_right left (Nat.mul_le_mul ordered.le ordered.le)).trans_lt
    (Nat.mul_lt_mul_of_pos_left ordered (Nat.mul_pos positive positive))

theorem width_mono {left right : Nat} (positive : 0 < left) (ordered : left ≤ right) :
    width left ≤ width right := by
  have leftPositive : cube left ≠ 0 := Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos positive positive) positive)
  have rightPositive : 0 < right := positive.trans_le ordered
  have rightNonzero : cube right ≠ 0 := Nat.ne_of_gt
    (Nat.mul_pos (Nat.mul_pos rightPositive rightPositive) rightPositive)
  simpa [width, decimal, leftPositive, rightNonzero] using Nat.le_length_digits_le 10 _ _ (cube_mono ordered)

/-- The processed roots are exactly the positive integers strictly below `next`. -/
def prior (next : Nat) : List Nat := (List.range' 1 (next - 1)).map cube

theorem mem_prior {value next : Nat} : value ∈ prior next ↔
    ∃ root, 0 < root ∧ root < next ∧ cube root = value := by
  simp only [prior, List.mem_map, List.mem_range'_1]
  constructor
  · rintro ⟨root, ⟨lower, upper⟩, equal⟩
    exact ⟨root, by omega, by omega, equal⟩
  · rintro ⟨root, positive, upper, equal⟩
    exact ⟨root, ⟨by omega, by omega⟩, equal⟩

theorem prior_succ (next : Nat) (positive : 0 < next) : prior (next + 1) = prior next ++ [cube next] := by
  have split : next = (next - 1) + 1 := by omega
  simp only [prior, Nat.add_sub_cancel]
  conv_lhs => rw [split, List.range'_1_concat]
  simp only [List.map_append, List.map_cons, List.map_nil, show 1 + (next - 1) = next by omega]

theorem record_away (n : Nat) (retired active : List Group)
    (different : ∀ group ∈ retired, group.digits ≠ signature (cube n)) :
    CubicGroups.record (cube n) (retired ++ active) = retired ++ CubicGroups.record (cube n) active := by
  have absent : contains (cube n) retired = false := by
    apply Bool.eq_false_iff.mpr
    intro present
    obtain ⟨group, member, same⟩ := (contains_iff _ _).mp present
    exact different group member same
  have unchanged : retired.map (update (cube n)) = retired := by
    calc
      retired.map (update (cube n)) = retired.map id := by
        apply List.map_congr_left
        intro group member
        simp [update, different group member]
      _ = retired := List.map_id retired
  have lookup : contains (cube n) (retired ++ active) = contains (cube n) active := by
    simpa only [contains, List.any_append, Bool.false_or] using
      congrArg (fun present => present || contains (cube n) active) absent
  simp only [CubicGroups.record, lookup]
  split <;> simp only [List.map_append, unchanged, List.append_assoc]

theorem record_width (n expected : Nat) (groups : List Group)
    (same : width n = expected) (old : ∀ group ∈ groups, groupWidth group = expected) :
    ∀ group ∈ CubicGroups.record (cube n) groups, groupWidth group = expected := by
  unfold CubicGroups.record
  split
  · intro group member
    obtain ⟨before, beforeMember, rfl⟩ := List.mem_map.mp member
    simpa only [update_width] using old before beforeMember
  · intro group member
    rcases List.mem_append.mp member with oldMember | newMember
    · exact old group oldMember
    · have equal : group = fresh (cube n) := by simpa using newMember
      simpa only [equal, fresh_width] using same

theorem matching_append (target : Int) (left right : List Group) :
    matching target (left ++ right) = matching target left ++ matching target right :=
  List.filterMap_append

/-- Retired buckets are a proof object; the executable keeps only the active width. -/
structure Invariant (target : Int) (state : CubicSearch.State) (retired : List Group) : Prop where
  positive : 0 < state.next
  previousWidth : state.currentWidth = width (state.next - 1)
  nextWidth : state.currentWidth ≤ width state.next
  summary : Summary (prior state.next) (retired ++ state.groups)
  retiredWidth : ∀ group ∈ retired, groupWidth group < state.currentWidth
  activeWidth : ∀ group ∈ state.groups, groupWidth group = state.currentWidth
  retiredEmpty : matching target retired = []

theorem invariant_start (target : Int) : Invariant target start [] := by
  constructor
  · decide
  · decide
  · decide
  · exact collect_summary []
  · simp
  · simp [start]
  · rfl

theorem Invariant.past_width {target : Int} {state : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) {root : Nat} (positive : 0 < root) (before : root < state.next) :
    width root ≤ state.currentWidth := by
  rw [invariant.previousWidth]
  exact width_mono positive (by omega)

theorem Invariant.retired_away {target : Int} {state : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) :
    ∀ group ∈ retired, group.digits ≠ signature (cube state.next) := by
  intro group member same
  have old := invariant.retiredWidth group member
  have nextWidth := invariant.nextWidth
  have equal : groupWidth group = width state.next := by
    simpa only [groupWidth, same, width] using signature_sum (cube state.next)
  omega

theorem Invariant.same_width {target : Int} {state : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) (same : state.currentWidth = width state.next) :
    Invariant target (advance state state.currentWidth state.groups) retired := by
  constructor
  · simp only [advance]
    omega
  · simpa only [advance, Nat.add_sub_cancel] using same
  · simpa only [advance, same] using width_mono invariant.positive (Nat.le_succ state.next)
  · have extended := invariant.summary.record (cube state.next)
    rw [record_away _ _ _ invariant.retired_away] at extended
    simpa only [advance, prior_succ _ invariant.positive] using extended
  · exact invariant.retiredWidth
  · exact record_width _ _ _ same.symm invariant.activeWidth
  · exact invariant.retiredEmpty

theorem Invariant.new_width {target : Int} {state : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) (growing : state.currentWidth < width state.next)
    (empty : matching target state.groups = []) :
    Invariant target (advance state (width state.next) []) (retired ++ state.groups) := by
  have allSmaller : ∀ group ∈ retired ++ state.groups, groupWidth group < width state.next := by
    intro group member
    rcases List.mem_append.mp member with old | active
    · exact (invariant.retiredWidth group old).trans growing
    · rw [invariant.activeWidth group active]
      exact growing
  have away : ∀ group ∈ retired ++ state.groups, group.digits ≠ signature (cube state.next) := by
    intro group member equal
    have smaller := allSmaller group member
    have same : groupWidth group = width state.next := by
      simpa only [groupWidth, equal, width] using signature_sum (cube state.next)
    omega
  constructor
  · simp only [advance]
    omega
  · simp only [advance, Nat.add_sub_cancel]
  · exact width_mono invariant.positive (Nat.le_succ state.next)
  · have extended := invariant.summary.record (cube state.next)
    have separated := record_away state.next (retired ++ state.groups) [] away
    simp only [List.append_nil] at separated
    rw [separated] at extended
    simpa only [advance, prior_succ _ invariant.positive] using extended
  · exact allSmaller
  · simpa only [advance, record_nil, List.mem_singleton] using
      (fun group (equal : group = fresh (cube state.next)) => equal ▸ fresh_width state.next)
  · simp only [matching, List.filterMap_append] at *
    simpa only [matching, empty, List.append_nil] using invariant.retiredEmpty

theorem Invariant.continues {target : Int} {state following : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) (checked : step target state = .inr following) :
    ∃ retired', Invariant target following retired' ∧ following.next = state.next + 1 := by
  by_cases growing : state.currentWidth < width state.next
  · cases selected : matching target state.groups with
    | nil =>
        simp only [step, if_pos growing, selected, Sum.inr.injEq] at checked
        subst following
        exact ⟨_, invariant.new_width growing selected, rfl⟩
    | cons first rest => simp [step, growing, selected] at checked
  · have same : state.currentWidth = width state.next := by have bounded := invariant.nextWidth; omega
    simp only [step, if_neg growing, Sum.inr.injEq] at checked
    subst following
    exact ⟨_, invariant.same_width same, rfl⟩

/-- Later widths cannot change an earlier class or improve a successful earlier minimum. -/
theorem eligible_min_append (target : Int) (inputs later : List Nat)
    (separated : ∀ before ∈ inputs, ∀ after ∈ later, (decimal after).length ≠ (decimal before).length)
    (ordered : ∀ before ∈ inputs, ∀ after ∈ later, before ≤ after)
    (nonempty : eligible target inputs ≠ []) :
    (eligible target inputs).min? = (eligible target (inputs ++ later)).min? := by
  apply minima_eq
  · intro value member
    obtain ⟨before, beforeMember, counted, attained⟩ := (mem_eligible target value inputs).mp member
    apply (mem_eligible target value (inputs ++ later)).mpr
    refine ⟨before, List.mem_append_left _ beforeMember, ?_, attained⟩
    rw [occurrences_separated before inputs later (separated before beforeMember)]
    exact counted
  · intro value member
    obtain ⟨after, afterMember, counted, attained⟩ := (mem_eligible target value (inputs ++ later)).mp member
    rcases List.mem_append.mp afterMember with before | laterMember
    · refine ⟨value, ?_, le_refl _⟩
      apply (mem_eligible target value inputs).mpr
      refine ⟨after, before, ?_, attained⟩
      rw [occurrences_separated after inputs later (separated after before)] at counted
      exact counted
    · cases selected : eligible target inputs with
      | nil => exact False.elim (nonempty selected)
      | cons first rest =>
          have firstMember : first ∈ eligible target inputs := by rw [selected]; simp
          obtain ⟨before, beforeMember, _, firstEq⟩ := (mem_eligible target first inputs).mp firstMember
          refine ⟨first, by simp, ?_⟩
          rw [← firstEq, ← attained]
          exact_mod_cast ordered before beforeMember after laterMember

def between (first stop : Nat) : List Nat := (List.range' first (stop - first)).map cube

theorem mem_between {value first stop : Nat} : value ∈ between first stop ↔
    ∃ root, first ≤ root ∧ root < stop ∧ cube root = value := by
  simp only [between, List.mem_map, List.mem_range'_1]
  constructor
  · rintro ⟨root, ⟨lower, upper⟩, equal⟩
    exact ⟨root, lower, by omega, equal⟩
  · rintro ⟨root, lower, upper, equal⟩
    exact ⟨root, ⟨lower, by omega⟩, equal⟩

theorem prior_split {next stop : Nat} (positive : 0 < next) (bounded : next ≤ stop) :
    prior stop = prior next ++ between next stop := by
  have sum : stop - 1 = (next - 1) + (stop - next) := by omega
  have start : 1 + (next - 1) = next := by omega
  simp only [prior, between, ← List.map_append]
  congr 1
  rw [sum, ← List.range'_append_1, start]

theorem Invariant.return_value {target value : Int} {state : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) {stop : Nat} (bounded : state.next ≤ stop)
    (checked : step target state = .inl value) : value = choose target (collect (prior stop)) := by
  have growing : state.currentWidth < width state.next := by
    by_contra absent
    simp [step, absent] at checked
  cases selected : matching target state.groups with
  | nil => simp [step, growing, selected] at checked
  | cons first rest =>
      have result : rest.foldl min first = value := by simpa [step, growing, selected] using checked
      have candidates : matching target (retired ++ state.groups) = first :: rest := by
        rw [matching_append, invariant.retiredEmpty, selected]
        rfl
      have oldMember : first ∈ eligible target (prior state.next) :=
        invariant.summary.eligible_of_matching target first (by rw [candidates]; simp)
      have nonempty : eligible target (prior state.next) ≠ [] := by
        intro empty
        simp [empty] at oldMember
      have separated : ∀ before ∈ prior state.next, ∀ after ∈ between state.next stop,
          (decimal after).length ≠ (decimal before).length := by
        intro before beforeMember after afterMember
        obtain ⟨left, leftPositive, leftBound, rfl⟩ := mem_prior.mp beforeMember
        obtain ⟨right, rightLower, _, rfl⟩ := mem_between.mp afterMember
        have small := invariant.past_width leftPositive leftBound
        have large := width_mono invariant.positive rightLower
        change width right ≠ width left
        omega
      have ordered : ∀ before ∈ prior state.next, ∀ after ∈ between state.next stop, before ≤ after := by
        intro before beforeMember after afterMember
        obtain ⟨left, _, leftBound, rfl⟩ := mem_prior.mp beforeMember
        obtain ⟨right, rightLower, _, rfl⟩ := mem_between.mp afterMember
        exact cube_mono (by omega)
      rw [choose_collect, prior_split invariant.positive bounded,
        ← eligible_min_append target _ _ separated ordered nonempty,
        ← invariant.summary.minimum_eq_eligible target, candidates, List.min?_cons']
      exact result.symm

/-- A successful exhaustive result at a closed digit width supplies a finite search witness. -/
theorem Invariant.returns_before {target : Int} {state : CubicSearch.State} {retired : List Group}
    (invariant : Invariant target state retired) (stop : Nat) (bounded : state.next ≤ stop)
    (closed : width (stop - 1) < width stop) (success : choose target (collect (prior stop)) ≠ 0) :
    Returns target state (choose target (collect (prior stop))) := by
  cases checked : step target state with
  | inl value =>
      have equal := invariant.return_value bounded checked
      rw [equal] at checked
      exact Returns.done checked
  | inr following =>
      obtain ⟨retired', nextInvariant, nextEq⟩ := invariant.continues checked
      have room : state.next < stop := by
        by_contra full
        have atEnd : state.next = stop := by omega
        have growing : state.currentWidth < width state.next := by
          rw [invariant.previousWidth, atEnd]
          exact closed
        have empty : matching target state.groups = [] := by
          cases selected : matching target state.groups with
          | nil => rfl
          | cons first rest => simp [step, growing, selected] at checked
        have allEmpty : matching target (retired ++ state.groups) = [] := by
          rw [matching_append, invariant.retiredEmpty, empty]
          rfl
        have zero : choose target (collect (prior stop)) = 0 := by
          rw [← atEnd, ← invariant.summary.choice_unique (collect_summary _) target]
          simp only [choose, allEmpty, List.min?_nil, Option.getD_none]
        exact success zero
      exact Returns.next checked (nextInvariant.returns_before stop (by omega) closed success)
termination_by stop - state.next
decreasing_by omega

theorem returns_closed (target : Int) (stop : Nat) (positive : 0 < stop)
    (closed : width (stop - 1) < width stop) (success : choose target (collect (prior stop)) ≠ 0) :
    Returns target start (choose target (collect (prior stop))) :=
  (invariant_start target).returns_before stop (by simpa [start] using positive) closed success

end ProjectEulerSolutions.CubicWindows
