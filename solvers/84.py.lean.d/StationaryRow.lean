import solvers.«84.py.lean.d».StationaryColumns

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open TriangleBufferCalls (encode working)
set_option Elab.async false

def rowValues (sides : Nat) (target : Fin 120) : List Int := columnValues sides target 0 120 ++ [0]

def rowObjects (sides : Nat) (target : Fin 120) : Mutable.Heap :=
  [.list (encode (columnValues sides target 0 120)), .list [.scalar (.int 0)], .list (encode (rowValues sides target))]

def rowHeap (heap : Mutable.Heap) (sides : Nat) (target : Fin 120) : Mutable.Heap := heap ++ rowObjects sides target

theorem row_result (sides : Nat) (target : Fin 120) (ordinary : target ≠ 119)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : transitions.View sides heap value)
    (getTransitions : env "transitions" = some value) (getTarget : env "target" = some (.scalar (.int target.val)))
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides))))
    (getSize : env "size" = some (.scalar (.int 120))) :
    Evaluates program env rowExpression heap (.ref (heap.length + 2)) (rowHeap heap sides target) globals := by
  apply Evaluates.binary (columns_result sides target ordinary heap env globals value view getTransitions getTarget getDenominator getSize)
    (Evaluates.list (.cons Evaluates.constant .nil))
  simp [Mutable.binary, Mutable.array, Mutable.allocate, working, rowHeap, rowObjects, rowValues, encode, List.append_assoc, Except.map,
    show (BinOp.add == BinOp.add) = true from rfl]

theorem row_found (heap : Mutable.Heap) (sides : Nat) (target : Fin 120) :
    (rowHeap heap sides target)[heap.length + 2]? = some (.list (encode (rowValues sides target))) := by
  simp [rowHeap, rowObjects]

theorem row_length (sides : Nat) (target : Fin 120) : (rowValues sides target).length = 121 := by
  simp [rowValues, columnValues]

theorem row_coefficient (sides : Nat) (target column : Fin 120) :
    (rowValues sides target)[column.val]? = some (Monopoly.coefficients sides target column) := by
  rw [rowValues, List.getElem?_append_left (by simp [columnValues])]
  simp [columnValues, column.isLt]

theorem row_constant (sides : Nat) (target : Fin 120) : (rowValues sides target)[120]? = some 0 := by
  simp [rowValues, columnValues]

end solvers.«84».stationary_weights
