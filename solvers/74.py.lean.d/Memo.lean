import solvers.«74.py.lean.d».Branches

/-! Physical cache batches store exact orbit lengths and retain successor-closed domains. -/
namespace solvers.«74».memo
open ProjectEulerStatements.BoundedOrbit ProjectEulerStatements.DigitFactorialChains
set_option Elab.async false

abbrev Valid (entries : store.Entries) := Memo.Valid (store.lookup entries)
abbrev Known (entries : store.Entries) (n : Nat) := Memo.Has (store.lookup entries) n

theorem tailFill_known (entries : store.Entries) (remaining : Nat) (values : List Nat)
    (key : Nat) : Known (write.tailFill entries remaining values) key ↔
      key ∈ values ∨ Known entries key := by
  induction values generalizing entries remaining with
  | nil => simp only [write.tailFill, List.not_mem_nil, false_or]
  | cons value rest ih =>
      rw [write.tailFill, ih]
      simp only [Known, store.lookup_insert_function, Memo.has_write, List.mem_cons]
      tauto

theorem before_last {start current value : Nat} {front : List Nat}
    (visited : walk.Prefix start current (front ++ [value])) :
    walk.Prefix start value front ∧ nextTerm value = current := by
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
    (entries : store.Entries) (valid : Valid entries)
    (visited : walk.Prefix start current path) (distinct : path.Nodup)
    (uncached : walk.Uncached entries path)
    (following : store.lookup entries current = some remaining) :
    Valid (write.tailFill entries remaining path.reverse) := by
  induction path using List.reverseRecOn generalizing entries current remaining with
  | nil => exact valid
  | append_singleton front value ih =>
      obtain ⟨before, next⟩ := before_last visited
      have fresh : store.lookup entries value = none := uncached value (by simp)
      have updated : Valid (store.insert entries value (remaining + 1)) := by
        rw [Valid, store.lookup_insert_function]
        exact valid.write_tail value remaining fresh (by simpa only [next] using following)
      have earlier : walk.Uncached (store.insert entries value (remaining + 1)) front := by
        intro n member
        have different : n ≠ value := by
          intro same
          subst n
          exact (List.nodup_append.mp distinct).2.2 value member value (by simp) rfl
        rw [store.lookup_insert, if_neg different]
        exact uncached n (List.mem_append_left _ member)
      have continued := ih value (remaining + 1) (store.insert entries value (remaining + 1))
        updated before (List.nodup_append.mp distinct).1 earlier
        (by simp [store.lookup_insert])
      simpa only [List.reverse_append, List.reverse_cons, List.reverse_nil, List.nil_append,
        List.singleton_append, write.tailFill] using continued

theorem start_known (start current remaining : Nat) (path : List Nat)
    (entries : store.Entries) (visited : walk.Prefix start current path)
    (following : store.lookup entries current = some remaining) :
    Known (write.tailFill entries remaining path.reverse) start := by
  apply (tailFill_known entries remaining path.reverse start).mpr
  by_cases empty : path = []
  · have same : start = current := by simpa only [empty, List.length_nil] using visited.frontier
    exact Or.inr ⟨remaining, same ▸ following⟩
  · apply Or.inl
    rw [List.mem_reverse, visited.visited]
    exact (mem_segment _ _ _ _).mpr ⟨0, List.length_pos_iff.mpr empty, rfl⟩

theorem cached_correct (start current remaining : Nat) (path : List Nat)
    (entries : store.Entries) (valid : Valid entries)
    (visited : walk.Prefix start current path) (distinct : path.Nodup)
    (uncached : walk.Uncached entries path)
    (following : store.lookup entries current = some remaining) :
    Valid (write.tailFill entries remaining path.reverse) ∧
      store.lookup (write.tailFill entries remaining path.reverse) start =
        some (chainLen start) := by
  have completed := tailFill_valid start current remaining path entries valid visited distinct
    uncached following
  obtain ⟨length, selected⟩ := start_known start current remaining path entries visited following
  exact ⟨completed, by simpa only [completed.correct start length selected] using selected⟩

theorem prefix_taken (start current : Nat) (path : List Nat)
    (visited : walk.Prefix start current path) (index : Nat) (inside : index ≤ path.length) :
    walk.Prefix start (nextTerm^[index] start) (path.take index) := by
  have sized : (path.take index).length = index := List.length_take_of_le inside
  constructor
  · rw [sized, visited.visited]
    simp only [segment, ← List.map_take, List.take_range, Nat.min_eq_left inside]
  · rw [sized]

/-- The repeated path position identifies a complete cycle followed by a fresh tail prefix. -/
theorem cycle_correct (start current index : Nat) (path : List Nat)
    (entries : store.Entries) (valid : Valid entries)
    (visited : walk.Prefix start current path) (canonical : path =
      ProjectEulerStatements.DigitFactorialChains.path start)
    (uncached : walk.Uncached entries path) (selected : path[index]? = some current) :
    Valid (branches.finishedEntries entries path index) ∧
      store.lookup (branches.finishedEntries entries path index) start = some (chainLen start) := by
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
  have cycleValid : Valid (branches.cycleEntries entries path index) :=
    write.cycleFill_valid entries (path.length - index) (path.drop index) valid
      correctCycle closedCycle
  have frontFresh : walk.Uncached (branches.cycleEntries entries path index) (path.take index) := by
    intro n member
    have outside : n ∉ path.drop index := by
      intro later
      have distinct := path_distinct start
      rw [← canonical, ← List.take_append_drop index path, List.nodup_append] at distinct
      exact distinct.2.2 n member n later rfl
    simp only [branches.cycleEntries, write.cycleFill_lookup, Memo.addCycle, if_neg outside]
    exact uncached n (List.mem_of_mem_take member)
  have frontierKnown : store.lookup (branches.cycleEntries entries path index) current =
      some (path.length - index) := by
    have member : current ∈ path.drop index := by
      have atStart : (path.drop index)[0]? = some current := by simpa using selected
      exact List.mem_of_getElem? atStart
    simp only [branches.cycleEntries, write.cycleFill_lookup, Memo.addCycle, if_pos member]
  apply cached_correct start current (path.length - index) (path.take index)
    (branches.cycleEntries entries path index) cycleValid
    (by simpa only [reached] using prefix_taken start current path visited index inside.le)
    ((path_distinct start).sublist (by simpa only [canonical] using List.take_sublist index path))
    frontFresh frontierKnown

end solvers.«74».memo
