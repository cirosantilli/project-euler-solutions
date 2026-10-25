import ProjectEulerSolutions.Lib.CubicGroups
import Mathlib.Data.List.Nodup

/-! Counts and minima of ordered groups agree with exhaustive signature filtering. -/
namespace ProjectEulerSolutions.CubicGroupCounts
open CubicPermutations CubicGroups
set_option Elab.async false

def occurrences (digits : List Nat) (inputs : List Nat) : Nat :=
  (inputs.filter (fun cube => decide (signature cube = digits))).length

structure Summary (inputs : List Nat) (groups : List Group) : Prop where
  distinct : (groups.map Group.digits).Nodup
  complete : ∀ cube ∈ inputs, ∃ group ∈ groups, group.digits = signature cube
  count : ∀ group ∈ groups, group.count = (occurrences group.digits inputs : Int)
  minimum : ∀ group ∈ groups, ∃ cube ∈ inputs, signature cube = group.digits ∧ group.minimum = (cube : Int) ∧
    ∀ other ∈ inputs, signature other = group.digits → group.minimum ≤ (other : Int)

theorem occurrences_append (digits : List Nat) (inputs : List Nat) (cube : Nat) :
    occurrences digits (inputs ++ [cube]) = occurrences digits inputs + if signature cube = digits then 1 else 0 := by
  by_cases same : signature cube = digits <;> simp [occurrences, List.filter_append, same]

theorem contains_iff (cube : Nat) (groups : List Group) :
    contains cube groups = true ↔ ∃ group ∈ groups, group.digits = signature cube := by
  simp [contains, List.any_eq_true]

theorem update_minimum (cube : Nat) (inputs : List Nat) (group : Group)
    (old : ∃ before ∈ inputs, signature before = group.digits ∧ group.minimum = (before : Int) ∧
      ∀ other ∈ inputs, signature other = group.digits → group.minimum ≤ (other : Int)) :
    ∃ before ∈ inputs ++ [cube], signature before = (update cube group).digits ∧
      (update cube group).minimum = (before : Int) ∧
      ∀ other ∈ inputs ++ [cube], signature other = (update cube group).digits →
        (update cube group).minimum ≤ (other : Int) := by
  obtain ⟨before, member, key, attained, lower⟩ := old
  by_cases same : group.digits = signature cube
  · have bound : ∀ other ∈ inputs ++ [cube], signature other = group.digits →
        min group.minimum (cube : Int) ≤ (other : Int) := by
      intro other otherMember otherKey
      rcases List.mem_append.mp otherMember with old | newMember
      · exact (min_le_left _ _).trans (lower other old otherKey)
      · have equal : other = cube := by simpa using newMember
        subst other
        exact min_le_right _ _
    by_cases smaller : group.minimum ≤ (cube : Int)
    · refine ⟨before, List.mem_append_left _ member, ?_, ?_, ?_⟩
      · simpa only [update_digits] using key
      · simp only [update, if_pos same, bump]
        rw [min_eq_left smaller]
        exact attained
      · simpa only [update, if_pos same, bump] using bound
    · refine ⟨cube, by simp, ?_, ?_, ?_⟩
      · simpa only [update_digits] using same.symm
      · simp [update, bump, same, min_eq_right (by omega : (cube : Int) ≤ group.minimum)]
      · simpa only [update, if_pos same, bump] using bound
  · refine ⟨before, List.mem_append_left _ member, ?_, ?_, ?_⟩
    · simpa only [update_digits] using key
    · simpa only [update, if_neg same] using attained
    · intro other otherMember otherKey
      rcases List.mem_append.mp otherMember with old | newMember
      · simpa only [update, if_neg same, update_digits] using lower other old (by simpa only [update_digits] using otherKey)
      · have equal : other = cube := by simpa using newMember
        subst other
        exact False.elim (same (by simpa only [update_digits] using otherKey.symm))

theorem Summary.record {inputs : List Nat} {groups : List Group} (summary : Summary inputs groups) (cube : Nat) :
    Summary (inputs ++ [cube]) (CubicGroups.record cube groups) := by
  by_cases present : contains cube groups = true
  · have selected := (contains_iff cube groups).mp present
    constructor
    · simpa [record_digits, present] using summary.distinct
    · intro value member
      rcases List.mem_append.mp member with old | newMember
      · obtain ⟨group, groupMember, key⟩ := summary.complete value old
        refine ⟨update cube group, ?_, ?_⟩
        · simp only [CubicGroups.record, if_pos present]
          exact List.mem_map.mpr ⟨group, groupMember, rfl⟩
        · simpa only [update_digits] using key
      · have equal : value = cube := by simpa using newMember
        subst value
        obtain ⟨group, groupMember, key⟩ := selected
        refine ⟨update cube group, ?_, ?_⟩
        · simp only [CubicGroups.record, if_pos present]
          exact List.mem_map.mpr ⟨group, groupMember, rfl⟩
        · simpa only [update_digits] using key
    · intro group member
      simp only [CubicGroups.record, if_pos present] at member
      obtain ⟨old, oldMember, rfl⟩ := List.mem_map.mp member
      rw [update_digits, occurrences_append]
      by_cases same : old.digits = signature cube
      · simp [update, bump, same, summary.count old oldMember]
      · simp [update, same, Ne.symm same, summary.count old oldMember]
    · intro group member
      simp only [CubicGroups.record, if_pos present] at member
      obtain ⟨old, oldMember, rfl⟩ := List.mem_map.mp member
      exact update_minimum cube inputs old (summary.minimum old oldMember)
  · have absent : ∀ group ∈ groups, group.digits ≠ signature cube := by
      simpa only [contains_iff, not_exists, not_and] using present
    have noKey : signature cube ∉ groups.map Group.digits := by
      intro member
      obtain ⟨group, groupMember, same⟩ := List.mem_map.mp member
      exact absent group groupMember same
    have oldEmpty : occurrences (signature cube) inputs = 0 := by
      unfold occurrences
      rw [List.filter_eq_nil_iff.mpr]
      · rfl
      · intro value member
        obtain ⟨group, groupMember, key⟩ := summary.complete value member
        simpa only [decide_eq_true_eq] using fun same => absent group groupMember (key.trans same)
    constructor
    · simpa [CubicGroups.record, present, fresh, List.nodup_append] using And.intro summary.distinct noKey
    · intro value member
      rcases List.mem_append.mp member with old | newMember
      · obtain ⟨group, groupMember, key⟩ := summary.complete value old
        exact ⟨group, by simp [CubicGroups.record, present, groupMember], key⟩
      · have equal : value = cube := by simpa using newMember
        subst value
        exact ⟨fresh cube, by simp [CubicGroups.record, present], rfl⟩
    · intro group member
      simp only [CubicGroups.record, if_neg present] at member
      rcases List.mem_append.mp member with old | newMember
      · rw [occurrences_append]
        simp [Ne.symm (absent group old), summary.count group old]
      · have equal : group = fresh cube := by simpa using newMember
        subst group
        simp [fresh, occurrences_append, oldEmpty]
    · intro group member
      simp only [CubicGroups.record, if_neg present] at member
      rcases List.mem_append.mp member with old | newMember
      · obtain ⟨before, beforeMember, key, attained, lower⟩ := summary.minimum group old
        refine ⟨before, List.mem_append_left _ beforeMember, key, attained, ?_⟩
        intro other otherMember otherKey
        rcases List.mem_append.mp otherMember with oldInput | freshInput
        · exact lower other oldInput otherKey
        · have equal : other = cube := by simpa using freshInput
          subst other
          exact False.elim (absent group old otherKey.symm)
      · have equal : group = fresh cube := by simpa using newMember
        subst group
        refine ⟨cube, by simp, rfl, rfl, ?_⟩
        intro other otherMember otherKey
        rcases List.mem_append.mp otherMember with oldInput | freshInput
        · obtain ⟨oldGroup, oldMember, key⟩ := summary.complete other oldInput
          exact False.elim (absent oldGroup oldMember (key.trans otherKey))
        · have equal : other = cube := by simpa using freshInput
          subst other
          exact le_refl _

def collect (inputs : List Nat) : List Group := inputs.foldl (fun groups cube => CubicGroups.record cube groups) []

theorem Summary.fold {inputs : List Nat} {groups : List Group} (summary : Summary inputs groups) (rest : List Nat) :
    Summary (inputs ++ rest) (rest.foldl (fun groups cube => CubicGroups.record cube groups) groups) := by
  induction rest generalizing inputs groups with
  | nil => simpa using summary
  | cons cube rest ih =>
      simpa only [List.foldl_cons, List.append_assoc, List.singleton_append] using (ih (summary.record cube))

theorem collect_summary (inputs : List Nat) : Summary inputs (collect inputs) := by
  have empty : Summary [] [] := ⟨by simp, by simp, by simp, by simp⟩
  simpa only [List.nil_append, collect] using empty.fold inputs

def eligible (target : Int) (inputs : List Nat) : List Int :=
  inputs.filterMap (fun cube => if (occurrences (signature cube) inputs : Int) = target then some (cube : Int) else none)
def choose (target : Int) (groups : List Group) : Int := (matching target groups).min?.getD 0

theorem mem_matching (target value : Int) (groups : List Group) :
    value ∈ matching target groups ↔ ∃ group ∈ groups, group.count = target ∧ group.minimum = value := by
  simp only [matching, List.mem_filterMap]
  constructor
  · rintro ⟨group, member, selected⟩
    by_cases counted : group.count = target
    · exact ⟨group, member, counted, by simpa only [if_pos counted, Option.some.injEq] using selected⟩
    · simp [counted] at selected
  · rintro ⟨group, member, counted, valueEq⟩
    exact ⟨group, member, by simp [counted, valueEq]⟩

theorem mem_eligible (target value : Int) (inputs : List Nat) :
    value ∈ eligible target inputs ↔
      ∃ cube ∈ inputs, (occurrences (signature cube) inputs : Int) = target ∧ (cube : Int) = value := by
  simp only [eligible, List.mem_filterMap]
  constructor
  · rintro ⟨cube, member, selected⟩
    by_cases counted : (occurrences (signature cube) inputs : Int) = target
    · exact ⟨cube, member, counted, by simpa only [if_pos counted, Option.some.injEq] using selected⟩
    · simp [counted] at selected
  · rintro ⟨cube, member, counted, valueEq⟩
    exact ⟨cube, member, by simp [counted, valueEq]⟩

theorem Summary.eligible_of_matching {inputs : List Nat} {groups : List Group} (summary : Summary inputs groups)
    (target value : Int) (member : value ∈ matching target groups) : value ∈ eligible target inputs := by
  obtain ⟨group, groupMember, counted, valueEq⟩ := (mem_matching target value groups).mp member
  obtain ⟨cube, cubeMember, key, attained, lower⟩ := summary.minimum group groupMember
  apply (mem_eligible target value inputs).mpr
  refine ⟨cube, cubeMember, ?_, attained.symm.trans valueEq⟩
  rw [key, ← summary.count group groupMember]
  exact counted

theorem Summary.matching_below_eligible {inputs : List Nat} {groups : List Group} (summary : Summary inputs groups)
    (target value : Int) (member : value ∈ eligible target inputs) :
    ∃ candidate ∈ matching target groups, candidate ≤ value := by
  obtain ⟨cube, cubeMember, counted, valueEq⟩ := (mem_eligible target value inputs).mp member
  obtain ⟨group, groupMember, key⟩ := summary.complete cube cubeMember
  obtain ⟨before, beforeMember, beforeKey, attained, lower⟩ := summary.minimum group groupMember
  refine ⟨group.minimum, ?_, ?_⟩
  · apply (mem_matching target group.minimum groups).mpr
    refine ⟨group, groupMember, ?_, rfl⟩
    simpa only [summary.count group groupMember, key] using counted
  · rw [← valueEq]
    exact lower cube cubeMember key.symm

theorem minima_eq (left right : List Int) (subset : ∀ value ∈ left, value ∈ right)
    (below : ∀ value ∈ right, ∃ candidate ∈ left, candidate ≤ value) : left.min? = right.min? := by
  cases selected : left.min? with
  | none =>
      have empty : left = [] := List.min?_eq_none_iff.mp selected
      cases right with
      | nil => rfl
      | cons head tail =>
          obtain ⟨candidate, member, lower⟩ := below head (by simp)
          simp [empty] at member
  | some value =>
      obtain ⟨member, lower⟩ := List.min?_eq_some_iff.mp selected
      have smallest : right.min? = some value := List.min?_eq_some_iff.mpr ⟨subset value member, by
        intro other otherMember
        obtain ⟨candidate, candidateMember, bounded⟩ := below other otherMember
        exact (lower candidate candidateMember).trans bounded⟩
      exact smallest.symm

/-- Choosing the least matching bucket is the exhaustive minimum over all cubes with the target frequency. -/
theorem Summary.minimum_eq_eligible {inputs : List Nat} {groups : List Group} (summary : Summary inputs groups)
    (target : Int) : (matching target groups).min? = (eligible target inputs).min? :=
  minima_eq _ _ (summary.eligible_of_matching target) (summary.matching_below_eligible target)

theorem choose_collect (target : Int) (inputs : List Nat) :
    choose target (collect inputs) = (eligible target inputs).min?.getD 0 := by
  rw [choose, (collect_summary inputs).minimum_eq_eligible target]

/-- A certificate may use a different group order from the source's insertion order. -/
theorem Summary.choice_unique {inputs : List Nat} {left right : List Group}
    (first : Summary inputs left) (second : Summary inputs right) (target : Int) :
    choose target left = choose target right := by
  rw [choose, first.minimum_eq_eligible, choose, second.minimum_eq_eligible]

def lookup (digits : List Nat) (groups : List Group) : Option Group :=
  groups.find? (fun group => decide (group.digits = digits))
def countOf (digits : List Nat) (groups : List Group) : Int := ((lookup digits groups).map Group.count).getD 0

theorem Summary.countOf_eq {inputs : List Nat} {groups : List Group} (summary : Summary inputs groups)
    (digits : List Nat) : countOf digits groups = (occurrences digits inputs : Int) := by
  cases selected : lookup digits groups with
  | none =>
      have absent : ∀ group ∈ groups, group.digits ≠ digits := by simpa [lookup] using List.find?_eq_none.mp selected
      have noInputs : occurrences digits inputs = 0 := by
        unfold occurrences
        rw [List.filter_eq_nil_iff.mpr]
        · rfl
        · intro cube cubeMember
          obtain ⟨group, groupMember, key⟩ := summary.complete cube cubeMember
          simpa only [decide_eq_true_eq] using fun same => absent group groupMember (key.trans same)
      simp [countOf, selected, noInputs]
  | some group =>
      have member := List.mem_of_find?_eq_some selected
      have key : group.digits = digits := of_decide_eq_true
        (List.find?_some (p := fun item : Group => decide (item.digits = digits)) selected)
      simpa only [countOf, selected, Option.map_some, Option.getD_some, key] using summary.count group member

/-- Equal histograms have equal decimal widths, including the special representation of zero. -/
theorem signature_length {left right : Nat} (same : signature left = signature right) :
    (decimal left).length = (decimal right).length :=
  ((DigitHistogram.counts_eq_iff _ _ (decimal_bounded left) (decimal_bounded right)).mp same).length_eq

theorem occurrences_width (query : Nat) (inputs : List Nat) :
    occurrences (signature query) (inputs.filter (fun cube => decide ((decimal cube).length = (decimal query).length))) =
      occurrences (signature query) inputs := by
  unfold occurrences
  rw [List.filter_filter]
  congr 1
  apply List.filter_congr
  intro cube member
  by_cases same : signature cube = signature query
  · simp [same, signature_length same]
  · simp [same]

theorem occurrences_separated (query : Nat) (inputs later : List Nat)
    (different : ∀ cube ∈ later, (decimal cube).length ≠ (decimal query).length) :
    occurrences (signature query) (inputs ++ later) = occurrences (signature query) inputs := by
  have empty : later.filter (fun cube => decide (signature cube = signature query)) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro cube member
    simpa only [decide_eq_true_eq] using fun same => different cube member (signature_length same)
  simp [occurrences, List.filter_append, empty]

def firstCube (accept : Nat → Bool) : Nat → Nat → Int
  | _, 0 => 0
  | first, count + 1 => if accept first then (first ^ 3 : Nat) else firstCube accept (first + 1) count

theorem firstCube_head (accept : Nat → Bool) (first count : Nat) :
    firstCube accept first count =
      (((List.range' first count).filter accept).map (fun n : Nat => ((n ^ 3 : Nat) : Int))).head?.getD 0 := by
  induction count generalizing first with
  | zero => rfl
  | succ count ih =>
      cases accepted : accept first <;> simp [firstCube, List.range'_succ, accepted, ih]

theorem min_eq_head (values : List Int) (ordered : values.Pairwise (· ≤ ·)) : values.min? = values.head? := by
  cases values with
  | nil => rfl
  | cons first rest =>
      apply List.min?_eq_some_iff.mpr
      refine ⟨by simp, ?_⟩
      intro other member
      rcases List.mem_cons.mp member with same | later
      · subst other
        exact le_refl _
      · exact (List.pairwise_cons.mp ordered).1 other later

/-- Ascending cube-base traversal is the same minimum selection as exhaustive filtering. -/
theorem firstCube_min (accept : Nat → Bool) (first count : Nat) :
    firstCube accept first count =
      (((List.range' first count).filter accept).map (fun n : Nat => ((n ^ 3 : Nat) : Int))).min?.getD 0 := by
  rw [firstCube_head]
  congr 1
  symm
  apply min_eq_head
  rw [List.pairwise_map]
  have ordered := (List.sortedLT_range' first count (s := 1) (by decide)).pairwise.sublist
    (List.filter_sublist (p := accept))
  exact ordered.imp (fun smaller => by exact_mod_cast Nat.pow_le_pow_left (Nat.le_of_lt smaller) 3)

end ProjectEulerSolutions.CubicGroupCounts
