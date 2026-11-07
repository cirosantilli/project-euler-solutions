import solvers.«84.py.lean.d».StationaryMatrixLoop

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open TriangleBufferCalls (encode)
set_option Elab.async false

def normalizationExpression : Expr :=
  .binOp (.binOp (.list [.constant (.int 1)]) .mult (.name "size")) .add (.list [.constant (.int 1)])

def normalizationObjects (size : Nat) : Mutable.Heap :=
  [.list [.scalar (.int 1)], .list (encode (List.replicate size 1)),
    .list [.scalar (.int 1)], .list (encode (List.replicate (size + 1) 1))]

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

def preparedHeap (base : Mutable.Heap) (sides : Nat) : Mutable.Heap :=
  (matrixHeap base sides 119 ++ normalizationObjects 120).set base.length
    (.list (rowReferences base 119 ++ [.ref ((matrixHeap base sides 119).length + 3)]))

theorem append_normalization (sides : Nat) (base : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMatrix : env "matrix" = some (.ref base.length)) (getSize : env "size" = some (.scalar (.int 120))) :
    BlockContinues program env [.expr (.call (.attribute (.name "matrix") "append") [normalizationExpression])]
      (matrixHeap base sides 119) env (preparedHeap base sides) globals := by
  apply BlockContinues.discard
  exact Evaluates.append (Evaluates.lookup getMatrix) (matrix_found base sides 119)
    (normalization_result 120 _ env globals getSize)
    (HeapCollection.prefix_found _ _ _ _ (matrix_found base sides 119))

theorem prepared_prefix (base : Mutable.Heap) (sides : Nat) : ∃ extra, preparedHeap base sides = base ++ extra := by
  unfold preparedHeap matrixHeap
  rw [HeapCollection.append, HeapCollection.store]
  exact ⟨_, rfl⟩

theorem prepared_found (base : Mutable.Heap) (sides : Nat) :
    (preparedHeap base sides)[base.length]? =
      some (.list (rowReferences base 119 ++ [.ref ((matrixHeap base sides 119).length + 3)])) := by
  apply List.getElem?_set_self
  simp [matrix_length, normalizationObjects]
  omega

theorem prepared_row_found (base : Mutable.Heap) (sides row : Nat) (inside : row < 119) :
    (preparedHeap base sides)[base.length + 3 * row + 3]? =
      some (.list (encode (rowValues sides (Fin.ofNat 120 row)))) := by
  unfold preparedHeap
  rw [List.getElem?_set_ne (by omega)]
  exact HeapCollection.prefix_found _ _ _ _ (matrix_row_found base sides 119 row inside)

theorem prepared_normalization_found (base : Mutable.Heap) (sides : Nat) :
    (preparedHeap base sides)[(matrixHeap base sides 119).length + 3]? =
      some (.list (encode (List.replicate 121 1))) := by
  unfold preparedHeap
  rw [List.getElem?_set_ne (by rw [matrix_length]; omega)]
  simp [normalizationObjects]

end solvers.«84».stationary_weights
