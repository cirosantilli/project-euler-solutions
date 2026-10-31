import solvers.«83.py.lean.d».Relaxation
import ProjectEulerSolutions.Lib.RectangularMatrixCalls

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls (number encode)
open ProjectEulerSolutions.BinaryHeapOrder (Ordered)
set_option Elab.async false

theorem neighbor_value (heap : Mutable.Heap) (env : HeapCalls.Env) (address row column : Nat)
    (rows : MatrixRows) (distance : Int) (rowExpr columnExpr : Expr)
    (matrixFound : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (rowInside : row < rows.length) (columnInside : column < rows[row].2.length)
    (localMatrix : env "matrix" = some (.ref address)) (localDistance : env "d" = some (number distance))
    (evaluatedRow : Evaluates program env rowExpr heap (number row) heap)
    (evaluatedColumn : Evaluates program env columnExpr heap (number column) heap)
    (ordinaryRow : ∀ a b c, rowExpr ≠ .slice a b c) (ordinaryColumn : ∀ a b c, columnExpr ≠ .slice a b c) :
    Evaluates program env (.binOp (.name "d") .add (.subscript (.subscript (.name "matrix") rowExpr) columnExpr))
      heap (number (distance + rows[row].2[column])) heap := by
  have rowRead : Evaluates program env (.subscript (.name "matrix") rowExpr) heap (.ref rows[row].1) heap := by
    apply Evaluates.subscript ordinaryRow (Evaluates.lookup localMatrix) evaluatedRow
    have got := Mutable.getItem_list heap address row (rows.map (fun pair => .ref pair.1)) matrixFound (by simpa using rowInside)
    simpa only [List.getElem_map] using got
  have cellRead : Evaluates program env (.subscript (.subscript (.name "matrix") rowExpr) columnExpr)
      heap (number rows[row].2[column]) heap := by
    apply Evaluates.subscript ordinaryColumn rowRead evaluatedColumn
    simpa only [encode, List.getElem_map] using Mutable.getItem_list heap rows[row].1 column (encode rows[row].2)
      (reads _ (List.getElem_mem rowInside)) (by simpa [encode] using columnInside)
  exact Evaluates.binary (Evaluates.lookup localDistance) cellRead rfl

theorem neighbor_disabled (heap : Mutable.Heap) (env : HeapCalls.Env) (guard index row column : Expr)
    (tested : Evaluates program env guard heap (.scalar (.bool false)) heap) :
    BlockContinues program env [neighborBranch guard index row column] heap env heap :=
  BlockContinues.branch tested rfl BlockContinues.nil

/-- A valid neighbor branch computes its actual matrix cost and then executes the shared relaxation. -/
theorem neighbor_enabled (heap : Mutable.Heap) (env : HeapCalls.Env) (matrixAddress distanceAddress queueAddress index row column : Nat)
    (rows : MatrixRows) (distances : List Int) (queue : List Mutable.Value) (infinity distance : Int)
    (visited : Nat → Bool) (guard indexExpr rowExpr columnExpr : Expr)
    (matrixFound : heap[matrixAddress]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (rowInside : row < rows.length) (columnInside : column < rows[row].2.length)
    (distanceFound : heap[distanceAddress]? = some (.list (encode distances)))
    (queueFound : heap[queueAddress]? = some (.list queue)) (separate : distanceAddress ≠ queueAddress)
    (inside : index < distances.length) (bounded : distances[index] ≤ infinity)
    (typed : ∀ value ∈ queue, QueueEntry heap value) (ordered : Ordered (priority heap) queue)
    (covered : CoveredQueue heap distances.length infinity (distanceValues distances infinity) visited queue)
    (localMatrix : env "matrix" = some (.ref matrixAddress)) (localDistance : env "d" = some (number distance))
    (localBuffer : env "dist" = some (.ref distanceAddress)) (localQueue : env "heap" = some (.ref queueAddress))
    (tested : Evaluates program env guard heap (.scalar (.bool true)) heap)
    (evaluatedIndex : Evaluates program env indexExpr heap (number index) heap)
    (evaluatedRow : Evaluates program (setLocal env "v" (number index)) rowExpr heap (number row) heap)
    (evaluatedColumn : Evaluates program (setLocal env "v" (number index)) columnExpr heap (number column) heap)
    (ordinaryRow : ∀ a b c, rowExpr ≠ .slice a b c) (ordinaryColumn : ∀ a b c, columnExpr ≠ .slice a b c) :
    let candidate := distance + rows[row].2[column]
    let updated := distances.set index (min distances[index] candidate)
    let ready := setLocal (setLocal env "v" (number index)) "nd" (number candidate)
    ∃ after result, BlockContinues program env [neighborBranch guard indexExpr rowExpr columnExpr] heap ready after ∧
      after[distanceAddress]? = some (.list (encode updated)) ∧
      after[queueAddress]? = some (.list result) ∧ Ordered (priority after) result ∧
      (∀ value ∈ result, QueueEntry after value) ∧
      CoveredQueue after distances.length infinity (distanceValues updated infinity) visited result ∧
      result.length ≤ queue.length + 1 ∧ heap.length ≤ after.length ∧ after.length ≤ heap.length + 1 ∧
      ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ queueAddress → after[other]? = heap[other]? := by
  dsimp only
  let indexed := setLocal env "v" (number index)
  let candidate := distance + rows[row].2[column]
  let ready := setLocal indexed "nd" (number candidate)
  have evaluated := neighbor_value heap indexed matrixAddress row column rows distance rowExpr columnExpr
    matrixFound reads rowInside columnInside (by simpa [indexed, setLocal] using localMatrix)
    (by simpa [indexed, setLocal] using localDistance) evaluatedRow evaluatedColumn ordinaryRow ordinaryColumn
  obtain ⟨after, result, ran, finalDistance, finalQueue, finalOrder, finalTyped, finalCoverage, queueBound,
      heapLower, heapUpper, frame⟩ := relaxation_result heap ready distanceAddress queueAddress index distances queue infinity
        candidate visited distanceFound queueFound separate inside bounded typed ordered covered
        (by simpa [ready, indexed, setLocal] using localBuffer) (by simpa [ready, indexed, setLocal] using localQueue)
        (by simp [ready, indexed, setLocal]) rfl
  refine ⟨after, result, ?_, finalDistance, finalQueue, finalOrder, finalTyped, finalCoverage, queueBound,
    heapLower, heapUpper, frame⟩
  apply BlockContinues.branch (flag := true) tested rfl
  exact (BlockContinues.assign evaluatedIndex).append ((BlockContinues.assign evaluated).append ran)

end solvers.«83»
