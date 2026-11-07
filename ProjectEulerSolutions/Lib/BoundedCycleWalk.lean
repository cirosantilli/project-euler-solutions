import ProjectEulerSolutions.Lib.BoundedCycles

/-! First-repeat walks and the processed-node invariant for longest bounded cycles. -/
namespace ProjectEulerSolutions.BoundedCycleWalk
open ProjectEulerStatements.BoundedOrbit BoundedCycles
set_option Elab.async false

inductive Trace (f : Nat → Nat) (limit : Nat) (processed : Finset Nat) :
    Nat → List Nat → Nat → List Nat → Prop
  | exit {current seen} : current = 0 ∨ limit < current → Trace f limit processed current seen current seen
  | cached {current seen} : 0 < current → current ≤ limit → current ∈ processed →
      Trace f limit processed current seen current seen
  | repeated {current seen} : 0 < current → current ≤ limit → current ∉ processed → current ∈ seen →
      Trace f limit processed current seen current seen
  | next {current seen last result} : 0 < current → current ≤ limit → current ∉ processed → current ∉ seen →
      Trace f limit processed (f current) (seen ++ [current]) last result →
      Trace f limit processed current seen last result

theorem Trace.distinct {f : Nat → Nat} {limit : Nat} {processed : Finset Nat}
    {current last : Nat} {seen result : List Nat} (trace : Trace f limit processed current seen last result)
    (distinct : seen.Nodup) : result.Nodup := by
  induction trace with
  | exit => exact distinct
  | cached => exact distinct
  | repeated => exact distinct
  | next _ _ _ fresh _ ih => exact ih (append_distinct _ _ distinct fresh)

theorem Trace.within {f : Nat → Nat} {limit : Nat} {processed : Finset Nat}
    {current last : Nat} {seen result : List Nat} (trace : Trace f limit processed current seen last result)
    (inside : Within limit seen) : Within limit result := by
  induction trace with
  | exit => exact inside
  | cached => exact inside
  | repeated => exact inside
  | next _ bounded _ _ _ ih => exact ih (append_within _ _ _ inside bounded)

theorem Trace.visited {f : Nat → Nat} {limit start : Nat} {processed : Finset Nat}
    {current last : Nat} {seen result : List Nat} (trace : Trace f limit processed current seen last result)
    (visited : Prefix f limit start current seen) : Prefix f limit start last result := by
  induction trace with
  | exit => exact visited
  | cached => exact visited
  | repeated => exact visited
  | next positive bounded _ _ _ ih => exact ih (visited.advance positive bounded)

theorem Trace.safe {f : Nat → Nat} {limit : Nat} {processed : Finset Nat}
    {current last : Nat} {seen result : List Nat} (trace : Trace f limit processed current seen last result)
    (best : Nat × Nat) (terminal : Safe f limit best last) :
    Safe f limit best current ∧ ∀ n ∈ result, n ∉ seen → Safe f limit best n := by
  induction trace with
  | exit => exact ⟨terminal, fun _ member absent => False.elim (absent member)⟩
  | cached => exact ⟨terminal, fun _ member absent => False.elim (absent member)⟩
  | repeated => exact ⟨terminal, fun _ member absent => False.elim (absent member)⟩
  | @next current seen last result positive bounded uncached fresh rest ih =>
      have ih := ih terminal
      have previous := safe_predecessor ih.1
      refine ⟨previous, ?_⟩
      intro n member absent
      by_cases same : n = current
      · simpa only [same] using previous
      · exact ih.2 n member (by simpa only [List.mem_append, List.mem_singleton, not_or] using And.intro absent same)

inductive Stop where
  | exit
  | cached
  | cycle (index : Nat)

structure Finished (f : Nat → Nat) (limit : Nat) (processed : Finset Nat) (initial : Nat) (seen : List Nat) where
  current : Nat
  path : List Nat
  stop : Stop
  trace : Trace f limit processed initial seen current path
  selected : match stop with
    | .exit => current = 0 ∨ limit < current
    | .cached => 0 < current ∧ current ≤ limit ∧ current ∈ processed
    | .cycle index => 0 < current ∧ current ≤ limit ∧ current ∉ processed ∧ path[index]? = some current

def run (f : Nat → Nat) (limit : Nat) (processed : Finset Nat) (current : Nat) (seen : List Nat)
    (distinct : seen.Nodup) (inside : Within limit seen) : Finished f limit processed current seen :=
  if outside : current = 0 ∨ limit < current then ⟨current, seen, .exit, .exit outside, outside⟩
  else if cached : current ∈ processed then
    ⟨current, seen, .cached, .cached (by omega) (by omega) cached, by omega, by omega, cached⟩
  else if repeated : current ∈ seen then
    ⟨current, seen, .cycle (seen.idxOf current), .repeated (by omega) (by omega) cached repeated,
      by omega, by omega, cached, List.getElem?_idxOf repeated⟩
  else
    let following := run f limit processed (f current) (seen ++ [current])
      (append_distinct _ _ distinct repeated) (append_within _ _ _ inside (by omega))
    ⟨following.current, following.path, following.stop,
      .next (by omega) (by omega) cached repeated following.trace, following.selected⟩
termination_by limit + 1 - seen.length
decreasing_by
  have enough := room limit current seen distinct inside (by omega) repeated
  simp only [List.length_append, List.length_singleton]
  omega

def fromStart (f : Nat → Nat) (limit : Nat) (processed : Finset Nat) (start : Nat) :
    Finished f limit processed start [] := run f limit processed start [] (by simp) (by simp [Within])

def Attained (f : Nat → Nat) (limit : Nat) (best : Nat × Nat) : Prop :=
  best = (0, 0) ∨ ∃ n, Cyclic f limit n ∧ best = score f limit n

theorem Attained.zero_min {f : Nat → Nat} {limit : Nat} {best : Nat × Nat}
    (attained : Attained f limit best) : best.1 = 0 → best.2 = 0 := by
  rcases attained with rfl | ⟨n, cycle, rfl⟩
  · simp
  · have positive := score_positive cycle
    omega

def choose (best candidate : Nat × Nat) : Nat × Nat :=
  if best.1 < candidate.1 ∨ candidate.1 = best.1 ∧ (best.2 = 0 ∨ candidate.2 < best.2)
  then candidate else best

theorem choose_dominates {f : Nat → Nat} {limit : Nat} {best candidate : Nat × Nat}
    (bestValid : Attained f limit best) (candidateValid : Attained f limit candidate) :
    Dominates (choose best candidate) best ∧ Dominates (choose best candidate) candidate := by
  have bestZero := bestValid.zero_min
  have candidateZero := candidateValid.zero_min
  have bestPositive : best.2 = 0 → best.1 = 0 := by
    rcases bestValid with rfl | ⟨n, cycle, rfl⟩
    · simp
    · have positive := score_positive cycle; omega
  unfold choose Dominates
  split <;> omega

theorem choose_attained {f : Nat → Nat} {limit : Nat} {best candidate : Nat × Nat}
    (bestValid : Attained f limit best) (candidateValid : Attained f limit candidate) :
    Attained f limit (choose best candidate) := by
  unfold choose
  split
  · exact candidateValid
  · exact bestValid

def CacheValid (f : Nat → Nat) (limit : Nat) (best : Nat × Nat) (processed : Finset Nat) : Prop :=
  ∀ n ∈ processed, Safe f limit best n

def winner {f : Nat → Nat} {limit start : Nat} {processed : Finset Nat} {seen : List Nat}
    (best : Nat × Nat) (finished : Finished f limit processed start seen) : Nat × Nat :=
  match finished.stop with
  | .cycle _ => choose best (score f limit finished.current)
  | _ => best

/-- Both exit branches and cycle discovery justify marking every traversed node processed. -/
theorem finished_correct (f : Nat → Nat) (limit : Nat) (processed : Finset Nat)
    (start : Nat) (bounded : start ≤ limit) (best : Nat × Nat)
    (attained : Attained f limit best) (valid : CacheValid f limit best processed)
    (finished : Finished f limit processed start []) :
    Attained f limit (winner best finished) ∧ Dominates (winner best finished) best ∧
    Safe f limit (winner best finished) start ∧
    CacheValid f limit (winner best finished) (processed ∪ finished.path.toFinset) := by
  have visited := finished.trace.visited (Prefix.initial f limit start bounded)
  have distinct := finished.trace.distinct (by simp)
  have terminal : Attained f limit (winner best finished) ∧ Dominates (winner best finished) best ∧
      Safe f limit (winner best finished) finished.current := by
    cases stopping : finished.stop with
    | exit =>
        have outside : finished.current = 0 ∨ limit < finished.current := by
          simpa only [stopping] using finished.selected
        refine ⟨by simpa [winner, stopping] using attained, ?_, ?_⟩
        · simp only [winner, stopping]; exact Dominates.refl best
        · simp only [winner, stopping]
          rcases outside with zero | outside
          · simpa only [zero] using safe_zero f limit best
          · exact safe_exit f limit finished.current best outside
    | cached =>
        have cached : 0 < finished.current ∧ finished.current ≤ limit ∧ finished.current ∈ processed := by
          simpa only [stopping] using finished.selected
        exact ⟨by simpa [winner, stopping] using attained,
          by simp only [winner, stopping]; exact Dominates.refl best,
          by simpa [winner, stopping] using valid finished.current cached.2.2⟩
    | cycle index =>
        have selected : 0 < finished.current ∧ finished.current ≤ limit ∧ finished.current ∉ processed ∧
            finished.path[index]? = some finished.current := by
          simpa only [stopping] using finished.selected
        have cycle := (repeated_cycle visited selected.1 selected.2.1 distinct selected.2.2.2).1
        have candidate : Attained f limit (score f limit finished.current) := Or.inr ⟨_, cycle, rfl⟩
        have dominates := choose_dominates attained candidate
        simp only [winner, stopping]
        exact ⟨choose_attained attained candidate, dominates.1, safe_cycle cycle dominates.2⟩
  have safe := finished.trace.safe (winner best finished) terminal.2.2
  refine ⟨terminal.1, terminal.2.1, safe.1, ?_⟩
  intro n member
  rcases Finset.mem_union.mp member with cached | traversed
  · exact (valid n cached).mono terminal.2.1
  · exact safe.2 n (List.mem_toFinset.mp traversed) (by simp)

theorem fromStart_correct (f : Nat → Nat) (limit : Nat) (processed : Finset Nat)
    (start : Nat) (bounded : start ≤ limit) (best : Nat × Nat)
    (attained : Attained f limit best) (valid : CacheValid f limit best processed) :
    let finished := fromStart f limit processed start
    Attained f limit (winner best finished) ∧ Dominates (winner best finished) best ∧
    Safe f limit (winner best finished) start ∧
    CacheValid f limit (winner best finished) (processed ∪ finished.path.toFinset) :=
  finished_correct f limit processed start bounded best attained valid (fromStart f limit processed start)

structure State where
  processed : Finset Nat
  best : Nat × Nat

def State.Valid (f : Nat → Nat) (limit : Nat) (state : State) : Prop :=
  Attained f limit state.best ∧ CacheValid f limit state.best state.processed

def advance (f : Nat → Nat) (limit start : Nat) (state : State) : State :=
  if start ∈ state.processed then state
  else
    let finished := fromStart f limit state.processed start
    ⟨state.processed ∪ finished.path.toFinset, winner state.best finished⟩

theorem advance_correct (f : Nat → Nat) (limit start : Nat) (state : State)
    (bounded : start ≤ limit) (valid : state.Valid f limit) :
    (advance f limit start state).Valid f limit ∧
    Dominates (advance f limit start state).best state.best ∧
    Safe f limit (advance f limit start state).best start := by
  unfold advance
  split
  · exact ⟨valid, Dominates.refl _, valid.2 start ‹_›⟩
  · have result := fromStart_correct f limit state.processed start bounded state.best valid.1 valid.2
    exact ⟨⟨result.1, result.2.2.2⟩, result.2.1, result.2.2.1⟩

def search (f : Nat → Nat) (limit : Nat) : Nat → Nat → State → State
  | _, 0, state => state
  | start, count + 1, state => search f limit (start + 1) count (advance f limit start state)

theorem search_correct (f : Nat → Nat) (limit start count : Nat) (state : State)
    (bounded : start + count ≤ limit + 1) (valid : state.Valid f limit) :
    (search f limit start count state).Valid f limit ∧
    Dominates (search f limit start count state).best state.best ∧
    ∀ n, start ≤ n → n < start + count → Safe f limit (search f limit start count state).best n := by
  induction count generalizing start state with
  | zero => exact ⟨valid, Dominates.refl _, by intro n lower upper; omega⟩
  | succ count ih =>
      have first := advance_correct f limit start state (by omega) valid
      have rest := ih (start + 1) (advance f limit start state) (by omega) first.1
      refine ⟨rest.1, rest.2.1.trans first.2.1, ?_⟩
      intro n lower upper
      by_cases same : n = start
      · subst n
        exact first.2.2.mono rest.2.1
      · exact rest.2.2 n (by omega) (by omega)

def Optimal (f : Nat → Nat) (limit : Nat) (best : Nat × Nat) : Prop :=
  Attained f limit best ∧ ∀ n, Cyclic f limit n → Dominates best (score f limit n)

theorem Optimal.unique {f : Nat → Nat} {limit : Nat} {left right : Nat × Nat}
    (first : Optimal f limit left) (second : Optimal f limit right) : left = right := by
  apply Dominates.antisymm
  · rcases second.1 with rfl | ⟨n, cycle, rfl⟩
    · exact dominates_zero left first.1.zero_min
    · exact first.2 n cycle
  · rcases first.1 with rfl | ⟨n, cycle, rfl⟩
    · exact dominates_zero right second.1.zero_min
    · exact second.2 n cycle

theorem one_not_cyclic (f : Nat → Nat) (limit : Nat) (one : f 1 = 0) : ¬Cyclic f limit 1 := by
  intro cycle
  have returns := ((cyclic_iff_return f limit 1).mp cycle).2
  have following : step f limit 1 = 0 := by simp [step, one]
  simp [following, orbit_zero] at returns

def solve (f : Nat → Nat) (limit : Nat) : Nat × Nat :=
  (search f limit 2 (limit - 1) ⟨∅, (0, 0)⟩).best

theorem solve_optimal (f : Nat → Nat) (limit : Nat) (one : f 1 = 0) : Optimal f limit (solve f limit) := by
  by_cases small : limit < 2
  · have empty : limit - 1 = 0 := by omega
    refine ⟨by simp [solve, empty, search, Attained], ?_⟩
    intro n cycle
    have bounded := cyclic_bounded cycle
    have positive := cycle.1
    have same : n = 1 := by omega
    subst n
    exact False.elim (one_not_cyclic f limit one cycle)
  · have checked := search_correct f limit 2 (limit - 1) ⟨∅, (0, 0)⟩ (by omega)
      ⟨Or.inl rfl, by intro n member; simp at member⟩
    refine ⟨checked.1.1, ?_⟩
    intro n cycle
    have bounded := cyclic_bounded cycle
    have positive := cycle.1
    have notOne : n ≠ 1 := by rintro rfl; exact one_not_cyclic f limit one cycle
    exact checked.2.2 n (by omega) (by omega) n (start_mem f limit n) cycle

end ProjectEulerSolutions.BoundedCycleWalk
