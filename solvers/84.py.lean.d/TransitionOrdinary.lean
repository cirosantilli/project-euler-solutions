import solvers.«84.py.lean.d».TransitionTable

namespace solvers.«84».transition_row
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary resolve_landing
set_option Elab.async false

/-- The complete ordinary-roll branch performs its lookup and all live landing-item updates. -/
theorem ordinary_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tableRoot root pos move doubles count : Nat) (isDouble : Bool) (entries : Entries)
    (ordinary : ¬(isDouble = true ∧ doubles = 2)) (table : Table heap tableRoot)
    (found : Found heap root entries) (getTable : env "landings" = some (.ref tableRoot))
    (getRow : env "row" = some (.ref root)) (getPos : env "pos" = some (number pos))
    (getMove : env "move" = some (number move)) (getDoubles : env "doubles" = some (number doubles))
    (getCount : env "count" = some (number count)) (getDouble : env "is_double" = some (.scalar (.bool isDouble))) :
    ∃ (dictionary : landings.Dictionary) (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env),
      dictionary.View heap ((pos + move) % 40) ∧
      BlockContinues program env code heap finalEnv finalHeap globals ∧
      Found finalHeap root (dictionary.entries.foldl (items.applyEntry doubles count isDouble) entries) ∧
      items.HeapFrame heap finalHeap root ∧ items.LocalsFrame env finalEnv := by
  obtain ⟨dictionary, evaluated, view⟩ := receiver_result env heap globals tableRoot pos move table getTable getPos getMove
  obtain ⟨finalHeap, finalEnv, ran, finalFound, frame, localFrame⟩ :=
    items.result heap env globals dictionary.address root doubles count isDouble dictionary.entries entries
      getRow getDoubles getCount getDouble evaluated view.1 view.2.1 found
      (fun entry member => by
        have bounds := view.bounded (Nat.mod_lt _ (by decide)) entry member
        exact ⟨bounds.1, Int.le_of_lt bounds.2.2⟩)
  exact ⟨dictionary, finalHeap, finalEnv, view,
    ordinary_branch env finalEnv heap finalHeap globals isDouble doubles ordinary getDouble getDoubles ran,
    finalFound, frame, localFrame⟩

end solvers.«84».transition_row
