import solvers.«83.py.lean.d».Bound
import solvers.«83.py.lean.d».OrderedQueue
import ProjectEulerSolutions.Lib.TriangleBufferCalls

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls (number encode working)
set_option Elab.async false

abbrev firstCell : Expr := .subscript (.subscript (.name "matrix") (.constant (.int 0))) (.constant (.int 0))
abbrev distanceExpression : Expr := .binOp (.list [.name "infinity"]) .mult (.name "total")
abbrev queueExpression : Expr := .list [.tuple [firstCell, .name "start"]]
abbrev visitedExpression : Expr := .binOp (.list [.constant (.bool false)]) .mult (.name "total")

def distanceHeap (heap : Mutable.Heap) (bound : Int) (values : List Int) : Mutable.Heap :=
  heap ++ [.list [number bound], .list (encode values)]

def queueHeap (heap : Mutable.Heap) (distance : Int) : Mutable.Heap :=
  heap ++ [.tuple [number distance, number 0], .list [.ref heap.length]]

def visitedHeap (heap : Mutable.Heap) (size : Nat) : Mutable.Heap :=
  heap ++ [.list [.scalar (.bool false)], .list (List.replicate size (.scalar (.bool false)))]

theorem first_cell (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (first : Nat × List Int) (rest : MatrixRows)
    (found : heap[address]? = some (.list ((first :: rest).map (fun pair => .ref pair.1))))
    (readFirst : heap[first.1]? = some (.list (encode first.2))) (positive : 0 < first.2.length)
    (localMatrix : env "matrix" = some (.ref address)) :
    Evaluates program env firstCell heap (number first.2[0]) heap := by
  have row : Evaluates program env (.subscript (.name "matrix") (.constant (.int 0)))
      heap (.ref first.1) heap := by
    apply Evaluates.subscript (by intros a b c impossible; cases impossible)
      (Evaluates.lookup localMatrix) Evaluates.constant
    simpa using Mutable.getItem_list heap address 0
      ((first :: rest).map (fun pair => .ref pair.1)) found (by simp)
  apply Evaluates.subscript (by intros a b c impossible; cases impossible) row Evaluates.constant
  simpa only [encode, List.getElem_map] using
    Mutable.getItem_list heap first.1 0 (encode first.2) readFirst (by simpa [encode] using positive)

theorem allocate_distances (heap : Mutable.Heap) (env : HeapCalls.Env) (size : Nat) (bound : Int)
    (localSize : env "total" = some (number size)) (localBound : env "infinity" = some (number bound)) :
    Evaluates program env distanceExpression heap (.ref (heap.length + 1))
      (distanceHeap heap bound (List.replicate size bound)) := by
  let scratch := heap ++ [Mutable.Object.list [number bound]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number size)
    (middle := scratch) (afterRight := scratch)
  · exact Evaluates.list (.cons (Evaluates.lookup localBound) .nil)
  · exact Evaluates.lookup localSize
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, scratch, distanceHeap, encode, Except.map]

theorem set_initial_distance (heap : Mutable.Heap) (env : HeapCalls.Env) (size : Nat)
    (bound distance : Int) (positive : 0 < size)
    (localDistance : env "dist" = some (.ref (heap.length + 1)))
    (localStart : env "start" = some (number 0))
    (readFirst : Evaluates program env firstCell (distanceHeap heap bound (List.replicate size bound))
      (number distance) (distanceHeap heap bound (List.replicate size bound))) :
    BlockContinues program env [.assign [.subscript (.name "dist") (.name "start")] firstCell]
      (distanceHeap heap bound (List.replicate size bound)) env
      (distanceHeap heap bound ((List.replicate size bound).set 0 distance)) := by
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible) readFirst
    (.cons (Evaluates.lookup localDistance) (.cons (Evaluates.lookup localStart) .nil))
  have write := ProjectEulerSolutions.TriangleBufferCalls.set_item (heap ++ [.list [number bound]])
    (List.replicate size bound) 0 distance (by simpa using positive)
  simp only [ProjectEulerSolutions.TriangleBufferCalls.working, List.length_append,
    List.length_singleton, List.append_assoc, List.singleton_append, Int.natCast_zero] at write
  exact write

theorem allocate_queue (heap : Mutable.Heap) (env : HeapCalls.Env) (distance : Int)
    (localStart : env "start" = some (number 0))
    (readFirst : Evaluates program env firstCell heap (number distance) heap) :
    Evaluates program env queueExpression heap (.ref (heap.length + 1)) (queueHeap heap distance) := by
  simpa only [queueHeap, List.append_assoc, List.singleton_append, List.length_append,
    List.length_singleton] using Evaluates.list (.cons
      (Evaluates.tuple (.cons readFirst (.cons (Evaluates.lookup localStart) .nil)) (by simp)) .nil)

theorem allocate_visited (heap : Mutable.Heap) (env : HeapCalls.Env) (size : Nat)
    (localSize : env "total" = some (number size)) :
    Evaluates program env visitedExpression heap (.ref (heap.length + 1)) (visitedHeap heap size) := by
  let scratch := heap ++ [Mutable.Object.list [.scalar (.bool false)]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number size)
    (middle := scratch) (afterRight := scratch)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.lookup localSize
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, scratch, visitedHeap, Except.map]

end solvers.«83»
