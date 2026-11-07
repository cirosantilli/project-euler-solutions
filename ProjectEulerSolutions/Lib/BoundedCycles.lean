import ProjectEulerStatements.Lib.BoundedOrbit
import Mathlib.Dynamics.PeriodicPts.Defs
import Mathlib.Data.Finset.Max

/-! Finite functional graphs with an absorbing exit and nonzero cycles. -/
namespace ProjectEulerSolutions.BoundedCycles
open ProjectEulerStatements.BoundedOrbit
set_option Elab.async false

/-- Leaving the permitted positive interval enters the absorbing exit. -/
def step (f : Nat → Nat) (limit n : Nat) : Nat :=
  if 0 < n ∧ n ≤ limit ∧ f n ≤ limit then f n else 0

theorem step_le (f : Nat → Nat) (limit n : Nat) : step f limit n ≤ limit := by
  unfold step
  split <;> omega

@[simp] theorem step_zero (f : Nat → Nat) (limit : Nat) : step f limit 0 = 0 := by
  simp [step]

theorem step_exit (f : Nat → Nat) (limit n : Nat) (outside : limit < n) :
    step f limit n = 0 := by simp [step, show ¬n ≤ limit by omega]

def path (f : Nat → Nat) (limit start : Nat) : List Nat :=
  walk (step f limit) (max limit start)
    (fun n _ => (step_le f limit n).trans (le_max_left _ _)) start []
    (le_max_right _ _) (by simp) (by simp [Within])

theorem path_trace (f : Nat → Nat) (limit start : Nat) :
    Trace (step f limit) start [] (path f limit start) := walk_trace ..

theorem path_distinct (f : Nat → Nat) (limit start : Nat) :
    (path f limit start).Nodup := (path_trace f limit start).distinct (by simp)

def orbit (f : Nat → Nat) (limit start : Nat) : Finset Nat := (path f limit start).toFinset

theorem mem_orbit (f : Nat → Nat) (limit start n : Nat) :
    n ∈ orbit f limit start ↔ ∃ steps, (step f limit)^[steps] start = n := by
  simpa only [orbit, List.mem_toFinset] using (path_trace f limit start).mem_iff

theorem start_mem (f : Nat → Nat) (limit start : Nat) : start ∈ orbit f limit start :=
  (mem_orbit f limit start start).mpr ⟨0, rfl⟩

theorem orbit_nonempty (f : Nat → Nat) (limit start : Nat) : (orbit f limit start).Nonempty :=
  ⟨start, start_mem f limit start⟩

@[simp] theorem orbit_card (f : Nat → Nat) (limit start : Nat) :
    (orbit f limit start).card = (path f limit start).length :=
  List.toFinset_card_of_nodup (path_distinct f limit start)

theorem orbit_subset {f : Nat → Nat} {limit start n : Nat} (reached : n ∈ orbit f limit start) :
    orbit f limit n ⊆ orbit f limit start := by
  obtain ⟨before, rfl⟩ := (mem_orbit f limit start n).mp reached
  intro m member
  obtain ⟨after, rfl⟩ := (mem_orbit f limit _ m).mp member
  exact (mem_orbit f limit _ _).mpr ⟨after + before, Function.iterate_add_apply ..⟩

theorem orbit_eq_of_mutual {f : Nat → Nat} {limit start n : Nat}
    (forward : n ∈ orbit f limit start) (backward : start ∈ orbit f limit n) :
    orbit f limit start = orbit f limit n :=
  Finset.Subset.antisymm (orbit_subset backward) (orbit_subset forward)

theorem orbit_next (f : Nat → Nat) (limit start : Nat) :
    orbit f limit start = insert start (orbit f limit (step f limit start)) := by
  ext n
  rw [mem_orbit, Finset.mem_insert, mem_orbit]
  constructor
  · rintro ⟨steps, reached⟩
    cases steps with
    | zero => exact Or.inl reached.symm
    | succ steps => exact Or.inr ⟨steps, reached⟩
  · rintro (rfl | ⟨steps, reached⟩)
    · exact ⟨0, rfl⟩
    · exact ⟨steps + 1, reached⟩

theorem orbit_zero (f : Nat → Nat) (limit : Nat) : orbit f limit 0 = {0} := by
  ext n
  rw [mem_orbit, Finset.mem_singleton]
  constructor
  · rintro ⟨steps, rfl⟩
    induction steps with
    | zero => rfl
    | succ steps ih => simp only [Function.iterate_succ_apply', ih, step_zero]
  · rintro rfl
    exact ⟨0, rfl⟩

theorem orbit_periodic (f : Nat → Nat) (limit start count : Nat) (positive : 0 < count)
    (periodic : (step f limit)^[count] start = start) :
    orbit f limit start = (segment (step f limit) start count).toFinset := by
  have period : Function.IsPeriodicPt (step f limit) count start := periodic
  ext n
  rw [mem_orbit, List.mem_toFinset, mem_segment]
  constructor
  · rintro ⟨steps, reached⟩
    exact ⟨steps % count, Nat.mod_lt _ positive, (period.iterate_mod_apply steps).trans reached⟩
  · rintro ⟨steps, _, reached⟩
    exact ⟨steps, reached⟩

theorem periodic_backward (f : Nat → Nat) (limit start count : Nat) (positive : 0 < count)
    (periodic : (step f limit)^[count] start = start) (n : Nat) (member : n ∈ orbit f limit start) :
    start ∈ orbit f limit n := by
  rw [orbit_periodic f limit start count positive periodic, List.mem_toFinset, mem_segment] at member
  obtain ⟨steps, before, rfl⟩ := member
  apply (mem_orbit f limit _ start).mpr
  refine ⟨count - steps, ?_⟩
  rw [← Function.iterate_add_apply, Nat.sub_add_cancel before.le]
  exact periodic

def Cyclic (f : Nat → Nat) (limit start : Nat) : Prop :=
  0 < start ∧ ∃ count, 0 < count ∧ (step f limit)^[count] start = start

theorem cyclic_bounded {f : Nat → Nat} {limit start : Nat} (cycle : Cyclic f limit start) :
    start ≤ limit := by
  obtain ⟨_, count, positive, periodic⟩ := cycle
  cases count with
  | zero => omega
  | succ count =>
      rw [Function.iterate_succ_apply'] at periodic
      rw [← periodic]
      exact step_le f limit _

theorem cyclic_positive {f : Nat → Nat} {limit start n : Nat}
    (cycle : Cyclic f limit start) (member : n ∈ orbit f limit start) : 0 < n := by
  obtain ⟨positive, count, countPositive, periodic⟩ := cycle
  have backward := periodic_backward f limit start count countPositive periodic n member
  by_contra zero
  have same : n = 0 := by omega
  rw [same, orbit_zero, Finset.mem_singleton] at backward
  omega

theorem cyclic_of_mem {f : Nat → Nat} {limit start n : Nat}
    (cycle : Cyclic f limit start) (member : n ∈ orbit f limit start) : Cyclic f limit n := by
  refine ⟨cyclic_positive cycle member, ?_⟩
  obtain ⟨_, count, positive, periodic⟩ := cycle
  obtain ⟨steps, rfl⟩ := (mem_orbit f limit start n).mp member
  refine ⟨count, positive, ?_⟩
  rw [← Function.iterate_add_apply, Nat.add_comm count steps, Function.iterate_add_apply, periodic]

theorem cyclic_orbit_eq {f : Nat → Nat} {limit start n : Nat}
    (cycle : Cyclic f limit start) (member : n ∈ orbit f limit start) :
    orbit f limit n = orbit f limit start := by
  obtain ⟨_, count, positive, periodic⟩ := cycle
  exact (orbit_eq_of_mutual member (periodic_backward f limit start count positive periodic n member)).symm

theorem cyclic_iff_return (f : Nat → Nat) (limit start : Nat) :
    Cyclic f limit start ↔ 0 < start ∧ start ∈ orbit f limit (step f limit start) := by
  constructor
  · rintro ⟨positive, count, countPositive, periodic⟩
    refine ⟨positive, ?_⟩
    cases count with
    | zero => omega
    | succ count => exact (mem_orbit f limit _ _).mpr ⟨count, periodic⟩
  · rintro ⟨positive, member⟩
    obtain ⟨count, periodic⟩ := (mem_orbit f limit _ _).mp member
    exact ⟨positive, count + 1, by omega, periodic⟩

instance (f : Nat → Nat) (limit start : Nat) : Decidable (Cyclic f limit start) :=
  decidable_of_iff' _ (cyclic_iff_return f limit start)

def score (f : Nat → Nat) (limit start : Nat) : Nat × Nat :=
  ((orbit f limit start).card, (orbit f limit start).min' (orbit_nonempty f limit start))

/-- Longer cycles win, with the least member breaking a length tie. -/
def Dominates (best candidate : Nat × Nat) : Prop :=
  candidate.1 < best.1 ∨ candidate.1 = best.1 ∧ best.2 ≤ candidate.2

theorem Dominates.refl (score : Nat × Nat) : Dominates score score := Or.inr ⟨rfl, le_rfl⟩

theorem Dominates.trans {a b c : Nat × Nat} (ab : Dominates a b) (bc : Dominates b c) :
    Dominates a c := by unfold Dominates at *; omega

theorem Dominates.antisymm {a b : Nat × Nat} (ab : Dominates a b) (ba : Dominates b a) : a = b := by
  apply Prod.ext <;> unfold Dominates at * <;> omega

theorem dominates_zero (best : Nat × Nat) (valid : best.1 = 0 → best.2 = 0) : Dominates best (0, 0) := by
  unfold Dominates
  omega

theorem score_positive {f : Nat → Nat} {limit start : Nat} (cycle : Cyclic f limit start) :
    0 < (score f limit start).1 ∧ 0 < (score f limit start).2 := by
  exact ⟨Finset.card_pos.mpr (orbit_nonempty f limit start),
    cyclic_positive cycle (Finset.min'_mem _ (orbit_nonempty f limit start))⟩

theorem score_eq_of_mem {f : Nat → Nat} {limit start n : Nat}
    (cycle : Cyclic f limit start) (member : n ∈ orbit f limit start) :
    score f limit n = score f limit start := by
  have same := cyclic_orbit_eq cycle member
  simp only [score, same]

/-- Every cycle reachable from a processed node is already covered by the winner. -/
def Safe (f : Nat → Nat) (limit : Nat) (best : Nat × Nat) (start : Nat) : Prop :=
  ∀ n ∈ orbit f limit start, Cyclic f limit n → Dominates best (score f limit n)

theorem Safe.mono {f : Nat → Nat} {limit : Nat} {best next : Nat × Nat} {start : Nat}
    (safe : Safe f limit best start) (better : Dominates next best) : Safe f limit next start := by
  intro n member cycle
  exact better.trans (safe n member cycle)

theorem safe_zero (f : Nat → Nat) (limit : Nat) (best : Nat × Nat) : Safe f limit best 0 := by
  intro n member cycle
  rw [orbit_zero, Finset.mem_singleton] at member
  have positive := cycle.1
  omega

theorem safe_exit (f : Nat → Nat) (limit n : Nat) (best : Nat × Nat) (outside : limit < n) :
    Safe f limit best n := by
  intro m member cycle
  rw [orbit_next, step_exit f limit n outside, orbit_zero, Finset.mem_insert, Finset.mem_singleton] at member
  have bound := cyclic_bounded cycle
  have positive := cycle.1
  omega

theorem safe_previous {f : Nat → Nat} {limit n : Nat} {best : Nat × Nat}
    (following : Safe f limit best (step f limit n)) : Safe f limit best n := by
  intro m member cycle
  rw [orbit_next, Finset.mem_insert] at member
  rcases member with same | later
  · subst m
    exact following n ((cyclic_iff_return f limit n).mp cycle).2 cycle
  · exact following m later cycle

theorem safe_predecessor {f : Nat → Nat} {limit n : Nat} {best : Nat × Nat}
    (following : Safe f limit best (f n)) : Safe f limit best n := by
  apply safe_previous
  unfold step
  split
  · exact following
  · exact safe_zero f limit best

theorem safe_cycle {f : Nat → Nat} {limit n : Nat} {best : Nat × Nat}
    (cycle : Cyclic f limit n) (covered : Dominates best (score f limit n)) : Safe f limit best n := by
  intro m member _
  rw [score_eq_of_mem cycle member]
  exact covered

def clamp (limit n : Nat) : Nat := if n ≤ limit then n else 0

@[simp] theorem clamp_inside (limit n : Nat) (inside : n ≤ limit) : clamp limit n = n := by
  simp [clamp, inside]

theorem step_eq_clamp (f : Nat → Nat) (limit n : Nat) (positive : 0 < n) (inside : n ≤ limit) :
    step f limit n = clamp limit (f n) := by simp [step, clamp, positive, inside]

/-- The source keeps the raw next value so that an out-of-range value takes its exit branch. -/
structure Prefix (f : Nat → Nat) (limit start current : Nat) (seen : List Nat) : Prop where
  visited : seen = segment (step f limit) start seen.length
  frontier : (step f limit)^[seen.length] start = clamp limit current

theorem Prefix.initial (f : Nat → Nat) (limit start : Nat) (inside : start ≤ limit) :
    Prefix f limit start start [] := ⟨rfl, (clamp_inside limit start inside).symm⟩

theorem Prefix.advance {f : Nat → Nat} {limit start current : Nat} {seen : List Nat}
    (visited : Prefix f limit start current seen) (positive : 0 < current) (inside : current ≤ limit) :
    Prefix f limit start (f current) (seen ++ [current]) := by
  have frontier : (step f limit)^[seen.length] start = current := by
    simpa only [clamp_inside limit current inside] using visited.frontier
  constructor
  · simp only [List.length_append, List.length_singleton]
    calc
      seen ++ [current] = segment (step f limit) start seen.length ++ [current] :=
        congrArg (· ++ [current]) visited.visited
      _ = segment (step f limit) start seen.length ++
          segment (step f limit) ((step f limit)^[seen.length] start) 1 := by simp [segment, frontier]
      _ = _ := (segment_append (step f limit) start seen.length 1).symm
  · simp only [List.length_append, List.length_singleton, Function.iterate_succ_apply', frontier]
    exact step_eq_clamp f limit current positive inside

/-- An index in the current path identifies the entire newly discovered cycle. -/
theorem repeated_cycle {f : Nat → Nat} {limit start current index : Nat} {seen : List Nat}
    (visited : Prefix f limit start current seen) (positive : 0 < current) (inside : current ≤ limit)
    (distinct : seen.Nodup) (selected : seen[index]? = some current) :
    Cyclic f limit current ∧ (seen.drop index).toFinset = orbit f limit current ∧
      (seen.drop index).length = (score f limit current).1 := by
  obtain ⟨before, picked⟩ := List.getElem?_eq_some_iff.mp selected
  have reached : (step f limit)^[index] start = current := by
    have value : seen[index]? = some ((step f limit)^[index] start) := by
      rw [visited.visited]
      simp [segment, before]
    exact Option.some.inj (value.symm.trans selected)
  have countPositive : 0 < seen.length - index := by omega
  have periodic : (step f limit)^[seen.length - index] current = current := by
    rw [← reached, ← Function.iterate_add_apply, Nat.sub_add_cancel before.le]
    exact visited.frontier.trans ((clamp_inside limit current inside).trans reached.symm)
  have suffix : seen.drop index = segment (step f limit) current (seen.length - index) := by
    calc
      seen.drop index = (segment (step f limit) start seen.length).drop index := by rw [← visited.visited]
      _ = segment (step f limit) ((step f limit)^[index] start) (seen.length - index) :=
        segment_drop _ _ _ _ before.le
      _ = _ := by rw [reached]
  have same : (seen.drop index).toFinset = orbit f limit current := by
    rw [suffix, orbit_periodic f limit current (seen.length - index) countPositive periodic]
  refine ⟨⟨positive, seen.length - index, countPositive, periodic⟩, same, ?_⟩
  change (seen.drop index).length = (orbit f limit current).card
  rw [← same, List.toFinset_card_of_nodup distinct.drop]

end ProjectEulerSolutions.BoundedCycles
