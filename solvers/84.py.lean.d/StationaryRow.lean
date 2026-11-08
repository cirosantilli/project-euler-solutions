import solvers.«84.py.lean.d».StationaryColumns
import ProjectEulerSolutions.Lib.MonopolyMatrixLayout

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open TriangleBufferCalls (encode working)
set_option Elab.async false

export ProjectEulerSolutions.MonopolyMatrixLayout (rowValues rowObjects rowHeap row_found row_length row_coefficient row_constant)

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

end solvers.«84».stationary_weights
