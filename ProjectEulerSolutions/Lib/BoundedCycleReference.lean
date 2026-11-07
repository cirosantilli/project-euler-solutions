import ProjectEulerSolutions.Lib.BoundedCycleWalk
import Mathlib.Order.Interval.Finset.Nat

/-! The reference's interval-derived iteration bound covers every first-repeat walk. -/
namespace ProjectEulerSolutions.BoundedCycleReference
open ProjectEulerStatements.BoundedOrbit BoundedCycles BoundedCycleWalk
set_option Elab.async false

theorem positive_length_le (limit : Nat) (seen : List Nat) (distinct : seen.Nodup)
    (inside : Within limit seen) (positive : ∀ n ∈ seen, 0 < n) : seen.length ≤ limit := by
  have subset : seen.toFinset ⊆ Finset.Icc 1 limit := by
    intro n member
    have member := List.mem_toFinset.mp member
    exact Finset.mem_Icc.mpr ⟨positive n member, inside n member⟩
  have bound := Finset.card_le_card subset
  simpa only [List.toFinset_card_of_nodup distinct, Nat.card_Icc, Nat.add_sub_cancel] using bound

def chainFuel (f : Nat → Nat) (limit start : Nat) : Nat → List Nat → Nat → List Nat
  | _, _, 0 => []
  | current, seen, fuel + 1 =>
      if current = 0 ∨ limit < current then []
      else if current ∈ seen then (if current = start then seen else [])
      else chainFuel f limit start (f current) (seen ++ [current]) fuel

def output (limit start last : Nat) (seen : List Nat) : List Nat :=
  if 0 < last ∧ last ≤ limit ∧ last = start then seen else []

theorem trace_reference {f : Nat → Nat} {limit current last : Nat} {seen result : List Nat}
    (trace : BoundedCycleWalk.Trace f limit ∅ current seen last result)
    (start fuel : Nat) (enough : limit + 1 ≤ seen.length + fuel)
    (distinct : seen.Nodup) (inside : Within limit seen) (positive : ∀ n ∈ seen, 0 < n) :
    chainFuel f limit start current seen fuel = output limit start last result := by
  induction trace generalizing fuel with
  | @exit current seen outside =>
      have bound := positive_length_le limit seen distinct inside positive
      cases fuel with
      | zero => omega
      | succ fuel =>
          have invalid : ¬(0 < current ∧ current ≤ limit ∧ current = start) := by omega
          simp only [chainFuel, if_pos outside, output, if_neg invalid]
  | cached _ _ member => simp at member
  | @repeated current seen currentPositive bounded _ repeated =>
      have bound := positive_length_le limit seen distinct inside positive
      cases fuel with
      | zero => omega
      | succ fuel =>
          have notExit : ¬(current = 0 ∨ limit < current) := by omega
          simp [chainFuel, notExit, repeated, output, currentPositive, bounded]
  | @next current seen last result currentPositive bounded _ fresh rest ih =>
      have bound := positive_length_le limit seen distinct inside positive
      cases fuel with
      | zero => omega
      | succ fuel =>
          have notExit : ¬(current = 0 ∨ limit < current) := by omega
          rw [chainFuel, if_neg notExit, if_neg fresh]
          exact ih fuel (by simp only [List.length_append, List.length_singleton]; omega)
            (append_distinct _ _ distinct fresh) (append_within _ _ _ inside bounded)
            (by intro n member; rcases List.mem_append.mp member with before | here
                · exact positive n before
                · simpa only [List.mem_singleton.mp here] using currentPositive)

def reference (f : Nat → Nat) (limit start : Nat) : List Nat :=
  chainFuel f limit start start [] (limit + 1)

theorem reference_run (f : Nat → Nat) (limit start : Nat) :
    reference f limit start =
      output limit start (fromStart f limit ∅ start).current (fromStart f limit ∅ start).path :=
  trace_reference (fromStart f limit ∅ start).trace start (limit + 1) (by simp)
    (by simp) (by simp [Within]) (by simp)

theorem prefix_subset {f : Nat → Nat} {limit start current : Nat} {seen : List Nat}
    (visited : Prefix f limit start current seen) : seen.toFinset ⊆ orbit f limit start := by
  intro n member
  rw [List.mem_toFinset, visited.visited, mem_segment] at member
  obtain ⟨steps, _, reached⟩ := member
  exact (mem_orbit f limit start n).mpr ⟨steps, reached⟩

theorem repeated_at_start {f : Nat → Nat} {limit start current index : Nat} {seen : List Nat}
    (visited : Prefix f limit start current seen) (cycle : Cyclic f limit start)
    (positive : 0 < current) (inside : current ≤ limit) (distinct : seen.Nodup)
    (selected : seen[index]? = some current) : current = start := by
  have classified := repeated_cycle visited positive inside distinct selected
  have member := List.mem_toFinset.mpr (List.mem_of_getElem? selected)
  have subset := prefix_subset visited
  have same := cyclic_orbit_eq cycle (subset member)
  have cardinality := Finset.card_le_card subset
  rw [← same, ← classified.2.1, List.toFinset_card_of_nodup distinct,
    List.toFinset_card_of_nodup distinct.drop, List.length_drop] at cardinality
  have insideIndex := (List.getElem?_eq_some_iff.mp selected).1
  have first : index = 0 := by omega
  have atStart : seen[0]? = some start := by
    rw [visited.visited]
    simp [segment, show 0 < seen.length by omega]
  rw [first, atStart] at selected
  exact (Option.some.inj selected).symm

/-- A positive periodic start cannot reach the absorbing exit. -/
theorem cyclic_no_exit {f : Nat → Nat} {limit start current : Nat} {seen : List Nat}
    (visited : Prefix f limit start current seen) (cycle : Cyclic f limit start)
    (outside : current = 0 ∨ limit < current) : False := by
  have zero : clamp limit current = 0 := by rcases outside with rfl | outside <;> simp [clamp]; omega
  have member : 0 ∈ orbit f limit start :=
    (mem_orbit f limit start 0).mpr ⟨seen.length, visited.frontier.trans zero⟩
  have positive := cyclic_positive cycle member
  omega

/-- Successful reference chains are precisely the complete positive bounded cycles. -/
theorem reference_correct (f : Nat → Nat) (limit start : Nat) (bounded : start ≤ limit) :
    (Cyclic f limit start → (reference f limit start).Nodup ∧
      (reference f limit start).toFinset = orbit f limit start) ∧
    (¬Cyclic f limit start → reference f limit start = []) := by
  let finished := fromStart f limit ∅ start
  have visited := finished.trace.visited (Prefix.initial f limit start bounded)
  have distinct := finished.trace.distinct (by simp)
  have result : reference f limit start = output limit start finished.current finished.path := reference_run f limit start
  cases stopping : finished.stop with
  | exit =>
      have outside : finished.current = 0 ∨ limit < finished.current := by
        simpa only [stopping] using finished.selected
      constructor
      · intro cycle
        exact False.elim (cyclic_no_exit visited cycle outside)
      · intro _
        rw [result, output, if_neg]
        omega
  | cached =>
      have cached : 0 < finished.current ∧ finished.current ≤ limit ∧ finished.current ∈ (∅ : Finset Nat) := by
        simpa only [stopping] using finished.selected
      simpa using cached.2.2
  | cycle index =>
      have selected : 0 < finished.current ∧ finished.current ≤ limit ∧ finished.current ∉ (∅ : Finset Nat) ∧
          finished.path[index]? = some finished.current := by
        simpa only [stopping] using finished.selected
      have classified := repeated_cycle visited selected.1 selected.2.1 distinct selected.2.2.2
      constructor
      · intro cycle
        have same := repeated_at_start visited cycle selected.1 selected.2.1 distinct selected.2.2.2
        have outputPath : reference f limit start = finished.path := by
          rw [result, output, if_pos ⟨selected.1, selected.2.1, same⟩]
        rw [outputPath]
        refine ⟨distinct, Finset.Subset.antisymm (prefix_subset visited) ?_⟩
        have complete := classified.2.1.trans (congrArg (orbit f limit) same)
        rw [← complete]
        intro n member
        exact List.mem_toFinset.mpr (List.mem_of_mem_drop (List.mem_toFinset.mp member))
      · intro noncyclic
        have different : finished.current ≠ start := by
          intro same
          exact noncyclic (same ▸ classified.1)
        simp [result, output, different]

def candidate (f : Nat → Nat) (limit start : Nat) : Nat × Nat :=
  if Cyclic f limit start then score f limit start else (0, 0)

def least (values : List Nat) : Nat :=
  if nonempty : values.toFinset.Nonempty then values.toFinset.min' nonempty else 0

def listScore (values : List Nat) : Nat × Nat := (values.length, least values)

theorem least_fold (first : Nat) (rest : List Nat) : least (first :: rest) = rest.foldl min first := by
  have nonempty : (first :: rest).toFinset.Nonempty := ⟨first, by simp⟩
  rw [least, dif_pos nonempty]
  apply (Finset.min'_eq_iff _ _ _).mpr
  have minimum := List.min?_eq_some_iff.mp (List.min?_cons' (x := first) (xs := rest))
  exact ⟨List.mem_toFinset.mpr minimum.1, fun n member => minimum.2 n (List.mem_toFinset.mp member)⟩

theorem reference_score (f : Nat → Nat) (limit start : Nat) (bounded : start ≤ limit) :
    listScore (reference f limit start) = candidate f limit start := by
  have checked := reference_correct f limit start bounded
  by_cases cycle : Cyclic f limit start
  · have result := checked.1 cycle
    have nonempty : (reference f limit start).toFinset.Nonempty := by
      rw [result.2]
      exact orbit_nonempty f limit start
    have length : (reference f limit start).length = (orbit f limit start).card := by
      rw [← result.2, List.toFinset_card_of_nodup result.1]
    simp only [candidate, if_pos cycle, listScore, least]
    rw [dif_pos nonempty]
    simp only [score, length, result.2]
  · simp [listScore, least, checked.2 cycle, candidate, cycle]

theorem score_min_le (f : Nat → Nat) (limit start : Nat) : (score f limit start).2 ≤ start :=
  Finset.min'_le _ _ (start_mem f limit start)

def Before (f : Nat → Nat) (limit first : Nat) (best : Nat × Nat) : Prop :=
  (best = (0, 0) ∨ ∃ n < first, Cyclic f limit n ∧ best = score f limit n) ∧
  ∀ n < first, Cyclic f limit n → Dominates best (score f limit n)

theorem Before.attained {f : Nat → Nat} {limit first : Nat} {best : Nat × Nat}
    (before : Before f limit first best) : Attained f limit best := by
  rcases before.1 with empty | ⟨n, _, cycle, same⟩
  · exact Or.inl empty
  · exact Or.inr ⟨n, cycle, same⟩

/-- When equal lengths retain the first start, the smaller cycle minimum has already been considered. -/
theorem tie_covered {f : Nat → Nat} {limit first : Nat} {best : Nat × Nat}
    (before : Before f limit first best) (cycle : Cyclic f limit first)
    (notLonger : (score f limit first).1 ≤ best.1) : Dominates best (score f limit first) := by
  by_cases shorter : (score f limit first).1 < best.1
  · exact Or.inl shorter
  have sameLength : (score f limit first).1 = best.1 := by omega
  let smallest := (score f limit first).2
  have member : smallest ∈ orbit f limit first := Finset.min'_mem _ (orbit_nonempty f limit first)
  have bounded : smallest ≤ first := score_min_le f limit first
  have sameScore := score_eq_of_mem cycle member
  by_cases earlier : smallest < first
  · have covered := before.2 smallest earlier (cyclic_of_mem cycle member)
    simpa only [sameScore] using covered
  · have same : smallest = first := by omega
    refine Or.inr ⟨sameLength, ?_⟩
    rcases before.1 with empty | ⟨n, earlier, _, value⟩
    · have positive := (score_positive cycle).1
      simp only [empty] at sameLength
      omega
    · have atMost : best.2 ≤ n := by rw [value]; exact score_min_le f limit n
      change best.2 ≤ smallest
      omega

def select (best next : Nat × Nat) : Nat × Nat := if best.1 < next.1 then next else best

theorem select_correct (f : Nat → Nat) (limit first : Nat) (best : Nat × Nat)
    (before : Before f limit first best) :
    Before f limit (first + 1) (select best (candidate f limit first)) := by
  by_cases cycle : Cyclic f limit first
  · simp only [candidate, if_pos cycle, select]
    by_cases longer : best.1 < (score f limit first).1
    · rw [if_pos longer]
      refine ⟨Or.inr ⟨first, by omega, cycle, rfl⟩, ?_⟩
      intro n lower accepted
      by_cases same : n = first
      · subst n
        exact Dominates.refl _
      · exact (show Dominates (score f limit first) best from Or.inl longer).trans
          (before.2 n (by omega) accepted)
    · rw [if_neg longer]
      refine ⟨?_, ?_⟩
      · rcases before.1 with empty | ⟨n, lower, accepted, value⟩
        · exact Or.inl empty
        · exact Or.inr ⟨n, by omega, accepted, value⟩
      · intro n lower accepted
        by_cases same : n = first
        · subst n
          exact tie_covered before cycle (by omega)
        · exact before.2 n (by omega) accepted
  · simp only [candidate, if_neg cycle, select, Nat.not_lt_zero, if_false]
    refine ⟨?_, ?_⟩
    · rcases before.1 with empty | ⟨n, lower, accepted, value⟩
      · exact Or.inl empty
      · exact Or.inr ⟨n, by omega, accepted, value⟩
    · intro n lower accepted
      have different : n ≠ first := by rintro rfl; exact cycle accepted
      exact before.2 n (by omega) accepted

def scan (f : Nat → Nat) (limit : Nat) : Nat → Nat → (Nat × Nat) → (Nat × Nat)
  | _, 0, best => best
  | first, count + 1, best => scan f limit (first + 1) count (select best (candidate f limit first))

theorem scan_before (f : Nat → Nat) (limit first count : Nat) (best : Nat × Nat)
    (before : Before f limit first best) : Before f limit (first + count) (scan f limit first count best) := by
  induction count generalizing first best with
  | zero => exact before
  | succ count ih =>
      simpa only [scan, Nat.add_assoc, Nat.add_comm 1 count] using
        ih (first + 1) _ (select_correct f limit first best before)

theorem scan_optimal (f : Nat → Nat) (limit : Nat) :
    Optimal f limit (scan f limit 0 (limit + 1) (0, 0)) := by
  have before := scan_before f limit 0 (limit + 1) (0, 0)
    ⟨Or.inl rfl, by intro n lower; omega⟩
  refine ⟨before.attained, ?_⟩
  intro n cycle
  exact before.2 n (by have := cyclic_bounded cycle; omega) cycle

theorem scan_eq_optimized (f : Nat → Nat) (limit : Nat) (one : f 1 = 0) :
    scan f limit 0 (limit + 1) (0, 0) = solve f limit :=
  (scan_optimal f limit).unique (solve_optimal f limit one)

end ProjectEulerSolutions.BoundedCycleReference
