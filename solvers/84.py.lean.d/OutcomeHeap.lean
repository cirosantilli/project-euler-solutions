import solvers.«84.py.lean.d».Transport
import ProjectEulerSolutions.Lib.HeapCollection

namespace solvers.«84».outcomes
open FormalSW.PyAst HeapCalls
set_option Elab.async false

structure Outcome where
  move : Int
  isDouble : Bool
  count : Int
deriving DecidableEq, Repr

def Outcome.items (outcome : Outcome) : List Mutable.Value :=
  [.scalar (.int outcome.move), .scalar (.bool outcome.isDouble), .scalar (.int outcome.count)]

def references (base : Mutable.Heap) (values : List Outcome) : List Mutable.Value :=
  (List.range values.length).map fun index => .ref (base.length + 1 + index)

def heap (base : Mutable.Heap) (values : List Outcome) : Mutable.Heap :=
  base ++ (.list (references base values) :: values.map fun outcome => .tuple outcome.items)

@[simp] theorem found (base : Mutable.Heap) (values : List Outcome) :
    (heap base values)[base.length]? = some (.list (references base values)) := by simp [heap]

theorem tuple_found (base : Mutable.Heap) (values : List Outcome) (index : Nat) (outcome : Outcome)
    (item : values[index]? = some outcome) :
    (heap base values)[base.length + 1 + index]? = some (.tuple outcome.items) := by
  rw [heap, List.getElem?_append_right (by omega)]
  rw [show base.length + 1 + index - base.length = index + 1 by omega]
  simp [item]

theorem references_append (base : Mutable.Heap) (values : List Outcome) (outcome : Outcome) :
    references base (values ++ [outcome]) = references base values ++ [.ref (heap base values).length] := by
  simp [references, heap, List.range_succ, List.map_append, Nat.add_assoc]
  omega

theorem append_store (base : Mutable.Heap) (values : List Outcome) (outcome : Outcome) :
    ((heap base values) ++ [Mutable.Object.tuple outcome.items]).set base.length
      (.list (references base values ++ [Mutable.Value.ref (heap base values).length])) = heap base (values ++ [outcome]) := by
  rw [← references_append base values outcome]
  simp [heap, List.append_assoc, List.map_append, List.set_append_right]

/-- Each accepted dice category allocates one tuple and appends its actual reference. -/
theorem emit (base : Mutable.Heap) (values : List Outcome) (outcome : Outcome) (env : HeapCalls.Env)
    (expressions : List Expr) (globals : Mutable.Globals)
    (getOutcomes : env "outcomes" = some (.ref base.length))
    (arguments : Evaluates.Arguments program env expressions (heap base values) outcome.items (heap base values) globals) :
    BlockContinues program env [.expr (.call (.attribute (.name "outcomes") "append") [.tuple expressions])]
      (heap base values) env (heap base (values ++ [outcome])) globals := by
  have tuple := Evaluates.tuple arguments (by simp [Outcome.items])
  have appended := Evaluates.append (Evaluates.lookup getOutcomes) (found base values) tuple
    (ProjectEulerSolutions.HeapCollection.prefix_found _ [.tuple outcome.items] _ _ (found base values))
  simpa only [append_store] using BlockContinues.discard appended

end solvers.«84».outcomes
