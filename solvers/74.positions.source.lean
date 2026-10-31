import solvers.«74.store.source»

/-! The positions dictionary records exactly the current path's zero-based indices. -/
namespace solvers.«74».positions
open store
set_option Elab.async false

structure Exact (entries : Entries) (path : List Nat) : Prop where
  sound : ∀ n index, lookup entries n = some index → path[index]? = some n
  complete : ∀ index n, path[index]? = some n → lookup entries n = some index

theorem empty : Exact [] [] where
  sound := by intro n index impossible; cases impossible
  complete := by intro index n impossible; cases impossible

theorem missing_iff (entries : Entries) (path : List Nat) (recorded : Exact entries path)
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

theorem insert (entries : Entries) (path : List Nat) (recorded : Exact entries path)
    (current : Nat) (fresh : current ∉ path) :
    Exact (store.insert entries current path.length) (path ++ [current]) := by
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

theorem selected (entries : Entries) (path : List Nat) (recorded : Exact entries path)
    (n index : Nat) (found : lookup entries n = some index) :
    index < path.length ∧ path[index]'((List.getElem?_eq_some_iff.mp
      (recorded.sound n index found)).1) = n := by
  obtain ⟨inside, picked⟩ := List.getElem?_eq_some_iff.mp (recorded.sound n index found)
  exact ⟨inside, picked⟩

end solvers.«74».positions
