import ProjectEulerSolutions.Lib.DigitFactorialCache

/-! The positions dictionary records exactly the current path's zero-based indices. -/
namespace ProjectEulerSolutions.DigitFactorialWalk.Positions
open DigitFactorialCache
set_option Elab.async false

structure Exact (entries : Cache) (path : List Nat) : Prop where
  sound : ∀ n index, lookup entries n = some index → path[index]? = some n
  complete : ∀ index n, path[index]? = some n → lookup entries n = some index

theorem empty : Exact ∅ [] where
  sound := by intro n index impossible; simp [lookup] at impossible
  complete := by intro index n impossible; cases impossible

theorem missing_iff (entries : Cache) (path : List Nat) (recorded : Exact entries path)
    (n : Nat) : lookup entries n = none ↔ n ∉ path := by
  constructor
  · intro missing member
    obtain ⟨index, inside, selected⟩ := List.mem_iff_getElem.mp member
    have picked : path[index]? = some n := List.getElem?_eq_some_iff.mpr ⟨inside, selected⟩
    have found := recorded.complete index n picked
    simp [missing] at found
  · intro absent
    cases found : lookup entries n with
    | none => rfl
    | some index => exact False.elim (absent (List.mem_of_getElem? (recorded.sound n index found)))

theorem insert (entries : Cache) (path : List Nat) (recorded : Exact entries path)
    (current : Nat) (fresh : current ∉ path) :
    Exact (DigitFactorialCache.insert entries current path.length) (path ++ [current]) := by
  constructor
  · intro n index found
    rw [lookup_insert] at found
    by_cases same : n = current
    · rw [if_pos same] at found
      have atEnd : index = path.length := (Option.some.inj found).symm
      subst index
      simpa only [same] using (show (path ++ [current])[path.length]? = some current by simp)
    · rw [if_neg same] at found
      have selected := recorded.sound n index found
      rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp selected).1]
      exact selected
  · intro index n selected
    by_cases before : index < path.length
    · have picked : path[index]? = some n := by
        simpa only [List.getElem?_append_left before] using selected
      have different : n ≠ current := by
        intro same
        exact fresh (same ▸ List.mem_of_getElem? picked)
      rw [lookup_insert, if_neg different]
      exact recorded.complete index n picked
    · have inside := (List.getElem?_eq_some_iff.mp selected).1
      have atEnd : index = path.length := by
        simp only [List.length_append, List.length_singleton] at inside
        omega
      subst index
      have same : n = current := by simpa using selected.symm
      rw [lookup_insert, if_pos same]

theorem selected (entries : Cache) (path : List Nat) (recorded : Exact entries path)
    (n index : Nat) (found : lookup entries n = some index) :
    index < path.length ∧ path[index]'((List.getElem?_eq_some_iff.mp
      (recorded.sound n index found)).1) = n := by
  obtain ⟨inside, picked⟩ := List.getElem?_eq_some_iff.mp (recorded.sound n index found)
  exact ⟨inside, picked⟩

end ProjectEulerSolutions.DigitFactorialWalk.Positions

namespace ProjectEulerSolutions.DigitFactorialWalk
open ProjectEulerStatements.BoundedOrbit ProjectEulerStatements.DigitFactorialChains
open DigitFactorialCache
set_option Elab.async false

inductive WalkTrace (entries : Cache) : Nat → List Nat → Nat → List Nat → Prop
  | cached {current seen remaining} : lookup entries current = some remaining →
      WalkTrace entries current seen current seen
  | repeated {current seen} : current ∈ seen → WalkTrace entries current seen current seen
  | step {current seen last result} : lookup entries current = none → current ∉ seen →
      WalkTrace entries (nextTerm current) (seen ++ [current]) last result →
        WalkTrace entries current seen last result

theorem WalkTrace.visited {entries : Cache} {start current last : Nat} {seen result : List Nat}
    (trace : WalkTrace entries current seen last result) (visited : Prefix start current seen) :
    Prefix start last result := by
  induction trace with
  | cached => exact visited
  | repeated => exact visited
  | step _ _ _ ih => exact ih visited.advance

theorem WalkTrace.distinct {entries : Cache} {current last : Nat} {seen result : List Nat}
    (trace : WalkTrace entries current seen last result) (distinct : seen.Nodup) :
    result.Nodup := by
  induction trace with
  | cached => exact distinct
  | repeated => exact distinct
  | step _ fresh _ ih => exact ih (append_distinct _ _ distinct fresh)

theorem WalkTrace.uncached {entries : Cache} {current last : Nat} {seen result : List Nat}
    (trace : WalkTrace entries current seen last result) (uncached : Uncached entries seen) :
    Uncached entries result := by
  induction trace with
  | cached => exact uncached
  | repeated => exact uncached
  | step missing _ _ ih => exact ih (uncached.advance missing)

/-- Without a cache hit, the memoized walk is the complete first-repeat walk. -/
theorem WalkTrace.complete {entries : Cache} {current last : Nat} {seen result : List Nat}
    (trace : WalkTrace entries current seen last result) (missing : lookup entries last = none) :
    Trace nextTerm current seen result := by
  induction trace with
  | cached found => simp [missing] at found
  | repeated present => exact .done present
  | step _ fresh _ ih => exact .step fresh (ih missing)

inductive Stop where
  | cached (remaining : Nat)
  | cycle (index : Nat)

structure Finished (entries : Cache) (initial : Nat) (seen : List Nat) where
  current : Nat
  path : List Nat
  positions : Cache
  stop : Stop
  trace : WalkTrace entries initial seen current path
  recorded : Positions.Exact positions path
  selected : match stop with
    | .cached remaining => lookup entries current = some remaining
    | .cycle index => lookup entries current = none ∧ lookup positions current = some index

/-- The proved finite invariant is only a termination measure; there is no exhausted branch. -/
def run (start : Nat) (entries : Cache) (current : Nat) (seen : List Nat) (positions : Cache)
    (bounded : current ≤ bound start) (distinct : seen.Nodup) (inside : Within (bound start) seen)
    (recorded : Positions.Exact positions seen) : Finished entries current seen :=
  match found : lookup entries current with
  | some remaining => ⟨current, seen, positions, .cached remaining, .cached found, recorded, found⟩
  | none =>
      match indexed : lookup positions current with
      | some index =>
          ⟨current, seen, positions, .cycle index,
            .repeated (List.mem_of_getElem? (recorded.sound current index indexed)),
            recorded, found, indexed⟩
      | none =>
          have fresh := (Positions.missing_iff positions seen recorded current).mp indexed
          let following := run start entries (nextTerm current) (seen ++ [current])
            (insert positions current seen.length) (nextTerm_bounded start current bounded)
            (append_distinct seen current distinct fresh)
            (append_within (bound start) seen current inside bounded)
            (Positions.insert positions seen recorded current fresh)
          ⟨following.current, following.path, following.positions, following.stop,
            .step found fresh following.trace, following.recorded, following.selected⟩
termination_by bound start + 1 - seen.length
decreasing_by
  have enough := room (bound start) current seen distinct inside bounded fresh
  simp only [List.length_append, List.length_singleton]
  omega

def fromStart (start : Nat) (entries : Cache) : Finished entries start [] :=
  run start entries start [] ∅ (le_max_left _ _) (by simp) (by simp [Within]) Positions.empty

def finalCache (entries : Cache) (initial : Nat) (seen : List Nat)
    (finished : Finished entries initial seen) : Cache :=
  match finished.stop with
  | .cached remaining => tailFill entries remaining finished.path.reverse
  | .cycle index => finishedEntries entries finished.path index

/-- The exact cache invariant holds after either batch, and the requested start is stored. -/
theorem fromStart_correct (start : Nat) (entries : Cache) (valid : Valid entries) :
    Valid (finalCache entries start [] (fromStart start entries)) ∧
      lookup (finalCache entries start [] (fromStart start entries)) start =
        some (chainLen start) := by
  let finished := fromStart start entries
  change Valid (finalCache entries start [] finished) ∧
    lookup (finalCache entries start [] finished) start = some (chainLen start)
  have visited := finished.trace.visited (Prefix.initial start)
  have distinct := finished.trace.distinct (by simp)
  have uncached := finished.trace.uncached (by simp [Uncached])
  cases stopping : finished.stop with
  | cached remaining =>
      have selected : lookup entries finished.current = some remaining := by
        simpa only [stopping] using finished.selected
      simpa only [finalCache, stopping] using
        cached_correct start finished.current remaining finished.path entries valid visited distinct
          uncached selected
  | cycle index =>
      have selected : lookup entries finished.current = none ∧
          lookup finished.positions finished.current = some index := by
        simpa only [stopping] using finished.selected
      have complete : finished.path = path start :=
        (finished.trace.complete selected.1).unique (path_trace start)
      simpa only [finalCache, stopping] using
        cycle_correct start finished.current index finished.path entries valid visited complete
          uncached (finished.recorded.sound _ _ selected.2)

end ProjectEulerSolutions.DigitFactorialWalk
