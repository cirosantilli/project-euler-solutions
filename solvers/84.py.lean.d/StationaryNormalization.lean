import solvers.«84.py.lean.d».StationaryMatrixLoop
import ProjectEulerSolutions.Lib.MonopolyMatrixLayout

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open TriangleBufferCalls (encode)
set_option Elab.async false

export ProjectEulerSolutions.MonopolyMatrixLayout (normalizationExpression normalizationObjects preparedHeap prepared_prefix prepared_found prepared_row_found prepared_normalization_found)

theorem normalization_source : (executableBody definition.2)[(2 : Nat)]? =
    some (.expr (.call (.attribute (.name "matrix") "append") [normalizationExpression])) := rfl

theorem normalization_result (size : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSize : env "size" = some (.scalar (.int size))) :
    Evaluates program env normalizationExpression heap (.ref (heap.length + 3)) (heap ++ normalizationObjects size) globals := by
  have repeated : Evaluates program env (.binOp (.list [.constant (.int 1)]) .mult (.name "size")) heap
      (.ref (heap.length + 1)) (heap ++ [.list [.scalar (.int 1)], .list (encode (List.replicate size 1))]) globals := by
    apply Evaluates.binary (Evaluates.list (.cons Evaluates.constant .nil)) (Evaluates.lookup getSize)
    simp [Mutable.binary, Mutable.array, Mutable.allocate, encode, Except.map, List.append_assoc,
      show (BinOp.mult == BinOp.mult) = true from rfl]
  apply Evaluates.binary repeated (Evaluates.list (.cons Evaluates.constant .nil))
  simp [Mutable.binary, Mutable.array, Mutable.allocate, encode, normalizationObjects, Except.map, List.append_assoc,
    List.replicate_succ', show (BinOp.add == BinOp.add) = true from rfl]

theorem append_normalization (sides : Nat) (base : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref base.length)) (getSize : env "size" = some (.scalar (.int 120))) :
    BlockContinues program env [.expr (.call (.attribute (.name "matrix") "append") [normalizationExpression])]
      (matrixHeap base sides 119) env (preparedHeap base sides) globals := by
  apply BlockContinues.discard
  exact Evaluates.append (Evaluates.lookup getMatrix) (matrix_found base sides 119)
    (normalization_result 120 _ env globals getSize)
    (HeapCollection.prefix_found _ _ _ _ (matrix_found base sides 119))

end solvers.«84».stationary_weights
