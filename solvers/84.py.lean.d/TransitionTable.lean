import solvers.«84.py.lean.d».TransitionItemsLoop
import solvers.«84.py.lean.d».LandingTableAccess

namespace solvers.«84».transition_row
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary resolve_landing
set_option Elab.async false

def Table (heap : Mutable.Heap) (root : Nat) : Prop :=
  ∃ values, heap[root]? = some (.list values) ∧ values.length = 40 ∧
    ∀ square : Nat, square < 40 → ∃ dictionary : landings.Dictionary,
      values[square]? = some (.ref dictionary.address) ∧ dictionary.View heap square

theorem Table.of_post {base : Mutable.Heap} {heap : Mutable.Heap}
    (post : landings.Post base (List.range' 0 40) (.ref base.length) heap) : Table heap base.length := by
  obtain ⟨values, _, _, found, length, _⟩ := post.at 0 (by decide)
  refine ⟨values, found, length, ?_⟩
  intro square bound
  obtain ⟨other, dictionary, _, present, _, selected, view⟩ := post.at square bound
  have same : other = values := by simpa only [found, Option.some.injEq, Mutable.Object.list.injEq] using present.symm
  subst other
  exact ⟨dictionary, selected, view⟩

theorem Table.frame {before after : Mutable.Heap} {tableRoot root : Nat}
    (table : Table before tableRoot) (entries : Entries) (found : Found before root entries)
    (frame : items.HeapFrame before after root) : Table after tableRoot := by
  have other : ∀ (address : Nat) (object : Mutable.Object), object ≠ .dict (encode entries) →
      before[address]? = some object → after[address]? = some object := by
    intro address object distinct present
    apply frame address object _ present
    intro same
    subst address
    exact distinct (Option.some.inj (present.symm.trans found))
  obtain ⟨values, present, length, contents⟩ := table
  refine ⟨values, other tableRoot _ (by intro equal; cases equal) present, length, ?_⟩
  intro square bound
  obtain ⟨dictionary, selected, view⟩ := contents square bound
  refine ⟨dictionary, selected, view.frame (frame.tuples entries found) ?_⟩
  intro address fields present
  exact other address _ (by intro equal; cases equal) present

/-- The source uses the bounded modular index and retrieves the corresponding live landing dictionary. -/
theorem receiver_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tableRoot pos move : Nat) (table : Table heap tableRoot)
    (getTable : env "landings" = some (.ref tableRoot))
    (getPos : env "pos" = some (number pos)) (getMove : env "move" = some (number move)) :
    ∃ dictionary : landings.Dictionary,
      Evaluates program env items.receiver heap (.ref dictionary.address) heap globals ∧
      dictionary.View heap ((pos + move) % 40) := by
  obtain ⟨values, found, _, contents⟩ := table
  obtain ⟨dictionary, selected, view⟩ := contents ((pos + move) % 40) (Nat.mod_lt _ (by decide))
  refine ⟨dictionary, ?_, view⟩
  have added : Evaluates program env (.binOp (.name "pos") .add (.name "move")) heap (number (pos + move)) heap globals := by
    apply Evaluates.binary (Evaluates.lookup getPos) (Evaluates.lookup getMove)
    simp [Mutable.binary, scalarBinary, number]
  have reduced : Evaluates program env (.binOp (.binOp (.name "pos") .add (.name "move")) .mod (.constant (.int 40)))
      heap (number ((pos + move) % 40)) heap globals := by
    apply Evaluates.binary added Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getTable) reduced
  generalize ((pos + move) % 40) = index at selected ⊢
  have bound := (List.getElem?_eq_some_iff.mp selected).1
  simp [Mutable.getItem, Mutable.index, found, selected, number, Except.map,
    not_lt.mpr (Int.natCast_nonneg index), not_le.mpr bound]

end solvers.«84».transition_row
