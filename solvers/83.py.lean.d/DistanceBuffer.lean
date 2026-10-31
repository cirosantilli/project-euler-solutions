import solvers.«83.py.lean.d».NeighborSyntax

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number encode)
set_option Elab.async false

def distanceValues (values : List Int) (infinity : Int) : Nat → Int :=
  fun index => values.getD index infinity

theorem distanceValues_read (values : List Int) (infinity : Int) (index : Nat)
    (inside : index < values.length) : distanceValues values infinity index = values[index] := by
  simp [distanceValues, List.getD, List.getElem?_eq_getElem inside]

theorem distanceValues_set (values : List Int) (infinity candidate : Int) (index : Nat)
    (inside : index < values.length) :
    distanceValues (values.set index candidate) infinity = Function.update (distanceValues values infinity) index candidate := by
  funext other
  by_cases same : other = index
  · subst other
    simp [distanceValues, List.getD, inside]
  · simp [distanceValues, List.getD, same, Ne.symm same]

theorem distance_read (heap : Mutable.Heap) (env : HeapCalls.Env) (address index : Nat) (values : List Int)
    (found : heap[address]? = some (.list (encode values))) (inside : index < values.length)
    (localDistance : env "dist" = some (.ref address)) (localIndex : env "v" = some (number index)) :
    Evaluates program env distanceAt heap (number values[index]) heap := by
  apply Evaluates.subscript (by intros a b c impossible; cases impossible)
    (Evaluates.lookup localDistance) (Evaluates.lookup localIndex)
  simpa only [encode, List.getElem_map] using
    Mutable.getItem_list heap address index (encode values) found (by simpa [encode] using inside)

theorem distance_test (heap : Mutable.Heap) (env : HeapCalls.Env) (address index : Nat)
    (values : List Int) (candidate : Int)
    (found : heap[address]? = some (.list (encode values))) (inside : index < values.length)
    (localDistance : env "dist" = some (.ref address)) (localIndex : env "v" = some (number index))
    (localCandidate : env "nd" = some (number candidate)) :
    Evaluates program env relaxationTest heap (.scalar (.bool (decide (candidate < values[index])))) heap := by
  apply Evaluates.compare (Evaluates.lookup localCandidate)
    (distance_read heap env address index values found inside localDistance localIndex)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem distance_write (heap : Mutable.Heap) (env : HeapCalls.Env) (address index : Nat)
    (values : List Int) (candidate : Int)
    (found : heap[address]? = some (.list (encode values))) (inside : index < values.length)
    (localDistance : env "dist" = some (.ref address)) (localIndex : env "v" = some (number index))
    (localCandidate : env "nd" = some (number candidate)) :
    BlockContinues program env [.assign [distanceAt] (.name "nd")] heap env
      (heap.set address (.list (encode (values.set index candidate)))) := by
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible) (Evaluates.lookup localCandidate)
    (.cons (Evaluates.lookup localDistance) (.cons (Evaluates.lookup localIndex) .nil))
  have nonnegative : ¬ (index : Int) < 0 := by omega
  have bounded : ¬ values.length ≤ index := by omega
  simp [Mutable.setItem, Mutable.index, found, encode, nonnegative, bounded, Except.map, List.map_set]

end solvers.«83»
