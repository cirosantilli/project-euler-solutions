import solvers.«84.py.lean.d».LandingTableSource

namespace solvers.«84».landings
open FormalSW.PyAst ProjectEulerSolutions resolve_landing
set_option Elab.async false

theorem Post.append {base : Mutable.Heap} {squares : List Nat} {value : Mutable.Value} {heap : Mutable.Heap}
    (post : Post base squares value heap) (extra : Mutable.Heap) : Post base squares value (heap ++ extra) := by
  obtain ⟨dictionaries, garbage, returned, rfl, views⟩ := post
  refine ⟨dictionaries, garbage ++ extra, returned, HeapCollection.append _ _ _ _, views.append extra⟩

theorem Post.prefix {base : Mutable.Heap} {squares : List Nat} {value : Mutable.Value} {heap : Mutable.Heap}
    (post : Post base squares value heap) : ∃ extra, heap = base ++ extra := by
  obtain ⟨dictionaries, garbage, _, same, _⟩ := post
  exact ⟨.list (references dictionaries) :: garbage, same⟩

theorem Post.at {base : Mutable.Heap} {value : Mutable.Value} {heap : Mutable.Heap}
    (post : Post base (List.range' 0 40) value heap) (square : Nat) (bound : square < 40) :
    ∃ (items : List Mutable.Value) (dictionary : Dictionary), value = .ref base.length ∧ heap[base.length]? = some (.list items) ∧
      items.length = 40 ∧ items[square]? = some (.ref dictionary.address) ∧ dictionary.View heap square := by
  obtain ⟨dictionaries, garbage, returned, rfl, views⟩ := post
  have length : dictionaries.length = 40 := by simpa using views.length_eq.symm
  have inside : square < dictionaries.length := by omega
  have viewed := views.get (i := square) (by simpa) inside
  refine ⟨references dictionaries, dictionaries[square], returned, HeapCollection.found _ _ _, ?_, ?_, ?_⟩
  · simp [references, length]
  · simp [references, List.getElem?_eq_getElem inside]
  · simpa using viewed

theorem Dictionary.View.sum {dictionary : Dictionary} {heap : Mutable.Heap} {square : Nat}
    (view : dictionary.View heap square) (landing : Int × Bool) :
    (dictionary.entries.map fun entry => if landing = entry.landing then entry.weight else 0).sum =
      weights.total (modelData square) landing := by
  rw [← weights.lookup_sum dictionary.entries landing view.2.2.1]
  exact view.2.2.2.2 landing

theorem Dictionary.View.bounded {dictionary : Dictionary} {heap : Mutable.Heap} {square : Nat}
    (view : dictionary.View heap square) (bound : square < 40) (entry : weights.Entry) (member : entry ∈ dictionary.entries) :
    0 ≤ entry.landing.1 ∧ entry.landing.1 < 40 ∧ 0 < entry.weight := by
  have positive := view.2.2.2.1 entry member
  have count := view.2.2.2.2 entry.landing
  rw [weights.lookup_at dictionary.entries entry member view.2.2.1] at count
  obtain ⟨item, allowed, key⟩ := weights.total_support (modelData square) entry.landing
    (by simpa only [Option.getD_some, ← count] using positive)
  have bounds := model_bound square bound item allowed
  rw [key] at bounds
  exact ⟨bounds.1, bounds.2, positive⟩

end solvers.«84».landings
