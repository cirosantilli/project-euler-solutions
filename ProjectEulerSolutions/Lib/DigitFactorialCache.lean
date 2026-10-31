import ProjectEulerStatements.Lib.DigitFactorialMemo
import Std.Data.HashMap.Lemmas

/-! Physical cache batches store exact orbit lengths and retain successor-closed domains. -/
namespace ProjectEulerSolutions.DigitFactorialCache
open ProjectEulerStatements.BoundedOrbit ProjectEulerStatements.DigitFactorialChains
set_option Elab.async false

abbrev Cache := Std.HashMap Nat Nat

def lookup (entries : Cache) (key : Nat) : Option Nat := entries[key]?
def insert (entries : Cache) (key value : Nat) : Cache := entries.insert key value

theorem lookup_insert (entries : Cache) (key value query : Nat) :
    lookup (insert entries key value) query =
      if query = key then some value else lookup entries query := by
  simp only [lookup, insert, Std.HashMap.getElem?_insert, beq_iff_eq]
  by_cases same : query = key
  · subst query
    simp
  · simp [same, Ne.symm same]

theorem lookup_insert_function (entries : Cache) (key value : Nat) :
    lookup (insert entries key value) = Memo.write (lookup entries) key value := by
  funext query
  exact lookup_insert entries key value query

def tailFill (entries : Cache) (remaining : Nat) : List Nat → Cache
  | [] => entries
  | value :: rest => tailFill (insert entries value (remaining + 1)) (remaining + 1) rest

def cycleFill (entries : Cache) (length : Nat) : List Nat → Cache
  | [] => entries
  | value :: rest => cycleFill (insert entries value length) length rest

abbrev cycleEntries (entries : Cache) (path : List Nat) (index : Nat) : Cache :=
  cycleFill entries (path.length - index) (path.drop index)

abbrev finishedEntries (entries : Cache) (path : List Nat) (index : Nat) : Cache :=
  tailFill (cycleEntries entries path index) (path.length - index) (path.take index).reverse

structure Prefix (start current : Nat) (path : List Nat) : Prop where
  visited : path = segment nextTerm start path.length
  frontier : nextTerm^[path.length] start = current

def Uncached (entries : Cache) (path : List Nat) : Prop :=
  ∀ n ∈ path, lookup entries n = none

theorem Prefix.initial (start : Nat) : Prefix start start [] := ⟨rfl, rfl⟩

theorem Prefix.advance {start current : Nat} {path : List Nat}
    (visited : Prefix start current path) :
    Prefix start (nextTerm current) (path ++ [current]) := by
  constructor
  · simp only [List.length_append, List.length_singleton]
    calc
      path ++ [current] = segment nextTerm start path.length ++ [current] :=
        congrArg (fun values => values ++ [current]) visited.visited
      _ = segment nextTerm start path.length ++
          segment nextTerm (nextTerm^[path.length] start) 1 := by simp [segment, visited.frontier]
      _ = segment nextTerm start (path.length + 1) :=
        (segment_append nextTerm start path.length 1).symm
  · simp only [List.length_append, List.length_singleton, Function.iterate_succ_apply',
      visited.frontier]

theorem Uncached.advance {entries : Cache} {path : List Nat} {current : Nat}
    (uncached : Uncached entries path) (missing : lookup entries current = none) :
    Uncached entries (path ++ [current]) := by
  intro n member
  rcases List.mem_append.mp member with earlier | last
  · exact uncached n earlier
  · simpa only [List.mem_singleton.mp last] using missing

theorem cycleFill_lookup (entries : Cache) (length : Nat) (values : List Nat) :
    lookup (cycleFill entries length values) = Memo.addCycle (lookup entries) values length := by
  induction values generalizing entries with
  | nil => funext key; simp [cycleFill, Memo.addCycle]
  | cons value rest ih =>
      funext key
      simp only [cycleFill, ih, Memo.addCycle, lookup_insert, List.mem_cons]
      by_cases here : key = value <;> by_cases later : key ∈ rest <;> simp [here, later]

theorem cycleFill_valid (entries : Cache) (length : Nat) (values : List Nat)
    (valid : Memo.Valid (lookup entries)) (correct : ∀ n ∈ values, chainLen n = length)
    (closed : ∀ n ∈ values, nextTerm n ∈ values) :
    Memo.Valid (lookup (cycleFill entries length values)) := by
  rw [cycleFill_lookup]
  exact valid.with_cycle values length correct closed

abbrev Valid (entries : Cache) := Memo.Valid (lookup entries)
abbrev Known (entries : Cache) (n : Nat) := Memo.Has (lookup entries) n

theorem empty_valid : Valid (∅ : Cache) := by
  have empty : lookup (∅ : Cache) = fun _ => none := by
    funext key
    simp [lookup]
  change Memo.Valid (lookup ∅)
  rw [empty]
  exact Memo.empty_valid

theorem tailFill_known (entries : Cache) (remaining : Nat) (values : List Nat)
    (key : Nat) : Known (tailFill entries remaining values) key ↔
      key ∈ values ∨ Known entries key := by
  induction values generalizing entries remaining with
  | nil => simp only [tailFill, List.not_mem_nil, false_or]
  | cons value rest ih =>
      rw [tailFill, ih]
      simp only [Known, lookup_insert_function, Memo.has_write, List.mem_cons]
      tauto

theorem before_last {start current value : Nat} {front : List Nat}
    (visited : Prefix start current (front ++ [value])) :
    Prefix start value front ∧ nextTerm value = current := by
  have expanded : front ++ [value] = segment nextTerm start front.length ++
      [nextTerm^[front.length] start] := by
    calc
      front ++ [value] = segment nextTerm start (front.length + 1) := by
        simpa only [List.length_append, List.length_singleton] using visited.visited
      _ = _ := by rw [segment_append]; rfl
  obtain ⟨head, last⟩ := List.append_inj expanded (by simp)
  have reached : nextTerm^[front.length] start = value := by simpa using last.symm
  refine ⟨⟨head, reached⟩, ?_⟩
  simpa only [List.length_append, List.length_singleton, Function.iterate_succ_apply', reached]
    using visited.frontier

/-- Reverse tail writes add one fresh predecessor to a complete cache at each step. -/
theorem tailFill_valid (start current remaining : Nat) (path : List Nat)
    (entries : Cache) (valid : Valid entries)
    (visited : Prefix start current path) (distinct : path.Nodup)
    (uncached : Uncached entries path)
    (following : lookup entries current = some remaining) :
    Valid (tailFill entries remaining path.reverse) := by
  induction path using List.reverseRecOn generalizing entries current remaining with
  | nil => exact valid
  | append_singleton front value ih =>
      obtain ⟨before, next⟩ := before_last visited
      have fresh : lookup entries value = none := uncached value (by simp)
      have updated : Valid (insert entries value (remaining + 1)) := by
        rw [Valid, lookup_insert_function]
        exact valid.write_tail value remaining fresh (by simpa only [next] using following)
      have earlier : Uncached (insert entries value (remaining + 1)) front := by
        intro n member
        have different : n ≠ value := by
          intro same
          subst n
          exact (List.nodup_append.mp distinct).2.2 value member value (by simp) rfl
        rw [lookup_insert, if_neg different]
        exact uncached n (List.mem_append_left _ member)
      have continued := ih value (remaining + 1) (insert entries value (remaining + 1))
        updated before (List.nodup_append.mp distinct).1 earlier
        (by simp [lookup_insert])
      simpa only [List.reverse_append, List.reverse_cons, List.reverse_nil, List.nil_append,
        List.singleton_append, tailFill] using continued

theorem start_known (start current remaining : Nat) (path : List Nat)
    (entries : Cache) (visited : Prefix start current path)
    (following : lookup entries current = some remaining) :
    Known (tailFill entries remaining path.reverse) start := by
  apply (tailFill_known entries remaining path.reverse start).mpr
  by_cases empty : path = []
  · have same : start = current := by simpa only [empty, List.length_nil] using visited.frontier
    exact Or.inr ⟨remaining, same ▸ following⟩
  · apply Or.inl
    rw [List.mem_reverse, visited.visited]
    exact (mem_segment _ _ _ _).mpr ⟨0, List.length_pos_iff.mpr empty, rfl⟩

theorem cached_correct (start current remaining : Nat) (path : List Nat)
    (entries : Cache) (valid : Valid entries)
    (visited : Prefix start current path) (distinct : path.Nodup)
    (uncached : Uncached entries path)
    (following : lookup entries current = some remaining) :
    Valid (tailFill entries remaining path.reverse) ∧
      lookup (tailFill entries remaining path.reverse) start =
        some (chainLen start) := by
  have completed := tailFill_valid start current remaining path entries valid visited distinct
    uncached following
  obtain ⟨length, selected⟩ := start_known start current remaining path entries visited following
  exact ⟨completed, by simpa only [completed.correct start length selected] using selected⟩

theorem prefix_taken (start current : Nat) (path : List Nat)
    (visited : Prefix start current path) (index : Nat) (inside : index ≤ path.length) :
    Prefix start (nextTerm^[index] start) (path.take index) := by
  have sized : (path.take index).length = index := List.length_take_of_le inside
  constructor
  · rw [sized, visited.visited]
    simp only [segment, ← List.map_take, List.take_range, Nat.min_eq_left inside]
  · rw [sized]

/-- The repeated path position identifies a complete cycle followed by a fresh tail prefix. -/
theorem cycle_correct (start current index : Nat) (path : List Nat)
    (entries : Cache) (valid : Valid entries)
    (visited : Prefix start current path) (canonical : path =
      ProjectEulerStatements.DigitFactorialChains.path start)
    (uncached : Uncached entries path) (selected : path[index]? = some current) :
    Valid (finishedEntries entries path index) ∧
      lookup (finishedEntries entries path index) start = some (chainLen start) := by
  have inside : index < path.length := (List.getElem?_eq_some_iff.mp selected).1
  have reached : nextTerm^[index] start = current := by
    rw [visited.visited] at selected
    simpa [segment, inside] using selected
  have repeated : nextTerm^[chainLen start] start = nextTerm^[index] start := by
    simpa only [chainLen, ← canonical] using visited.frontier.trans reached.symm
  have correctCycle : ∀ n ∈ path.drop index, chainLen n = path.length - index := by
    intro n member
    simpa only [chainLen, ← canonical] using chainLen_on_cycle start index n
      (by simpa only [chainLen, ← canonical] using inside) repeated
      (by simpa only [canonical] using member)
  have cycleShape : path.drop index =
      segment nextTerm (nextTerm^[index] start) (path.length - index) := by
    rw [visited.visited]
    simpa only [segment_length] using segment_drop nextTerm start path.length index inside.le
  have closedCycle : ∀ n ∈ path.drop index, nextTerm n ∈ path.drop index := by
    have periodic : nextTerm^[path.length - index] (nextTerm^[index] start) =
        nextTerm^[index] start := by
      rw [← Function.iterate_add_apply, Nat.sub_add_cancel inside.le]
      simpa only [chainLen, ← canonical] using repeated
    intro n member
    rw [cycleShape] at member ⊢
    exact cycle_closed _ _ (by omega) periodic n member
  have cycleValid : Valid (cycleEntries entries path index) :=
    cycleFill_valid entries (path.length - index) (path.drop index) valid
      correctCycle closedCycle
  have frontFresh : Uncached (cycleEntries entries path index) (path.take index) := by
    intro n member
    have outside : n ∉ path.drop index := by
      intro later
      have distinct := path_distinct start
      rw [← canonical, ← List.take_append_drop index path, List.nodup_append] at distinct
      exact distinct.2.2 n member n later rfl
    simp only [cycleEntries, cycleFill_lookup, Memo.addCycle, if_neg outside]
    exact uncached n (List.mem_of_mem_take member)
  have frontierKnown : lookup (cycleEntries entries path index) current =
      some (path.length - index) := by
    have member : current ∈ path.drop index := by
      have atStart : (path.drop index)[0]? = some current := by simpa using selected
      exact List.mem_of_getElem? atStart
    simp only [cycleEntries, cycleFill_lookup, Memo.addCycle, if_pos member]
  apply cached_correct start current (path.length - index) (path.take index)
    (cycleEntries entries path index) cycleValid
    (by simpa only [reached] using prefix_taken start current path visited index inside.le)
    ((path_distinct start).sublist (by simpa only [canonical] using List.take_sublist index path))
    frontFresh frontierKnown

end ProjectEulerSolutions.DigitFactorialCache
