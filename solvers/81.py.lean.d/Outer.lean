import solvers.«81.py.lean.d».Inputs
import ProjectEulerSolutions.Lib.GridPathBufferCalls

namespace solvers.«81»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P81
open ProjectEulerSolutions.RectangularMatrixCalls
open ProjectEulerSolutions.TriangleBufferCalls (working number found prefix_found)
open ProjectEulerSolutions.GridPathBufferCalls
set_option Elab.async false

abbrev columnFor : Stmt :=
  .forIn (.name "j") (.call (.name "range") [.constant (.int 1), .name "m"]) rowBody []
abbrev matrixBody : List Stmt :=
  [.assign [.name "row"] (.subscript (.name "matrix") (.name "i"))] ++ firstColumn ++ [columnFor]
abbrev lastExpression : Expr := .subscript (.name "dp") (.unaryOp .uSub (.constant (.int 1)))
abbrev finalReturn : Stmt := .return lastExpression

def remainingBuffer (rows : MatrixRows) (cursor : Nat) (buffer : List Int) : List Int :=
  ((rows.drop cursor).map Prod.snd).foldl rowStep buffer

theorem last_result (base : Mutable.Heap) (env : HeapCalls.Env) (buffer : List Int)
    (positive : 0 < buffer.length) (getBuffer : env "dp" = some (.ref base.length)) :
    Evaluates program env lastExpression (working base buffer)
      (number (buffer.getLast?.getD 0)) (working base buffer) := by
  have nonempty : buffer ≠ [] := by intro empty; simp [empty] at positive
  have tail : buffer[buffer.length - 1]? = some (buffer.getLast?.getD 0) := by
    rw [List.getLast?_eq_getElem?]
    simp [List.getElem?_eq_getElem (by omega : buffer.length - 1 < buffer.length)]
  apply Evaluates.subscript (middle := working base buffer) (afterIndex := working base buffer)
    (by intros a b c impossible; cases impossible) (Evaluates.lookup getBuffer)
    (Evaluates.unary Evaluates.constant rfl)
  have offset : ((-1 : Int) + buffer.length).toNat = buffer.length - 1 := by omega
  have indexed : Mutable.index (encode buffer) (number (-1)) (working base buffer) =
      some (.ok (buffer.length - 1), working base buffer) := by
    simp [Mutable.index, encode, nonempty, offset]
  have item : (encode buffer)[buffer.length - 1]? = some (number (buffer.getLast?.getD 0)) := by
    simp only [encode, List.getElem?_map, tail, Option.map_some]
  change Mutable.index (ProjectEulerSolutions.TriangleBufferCalls.encode buffer) (number (-1)) (working base buffer) = _ at indexed
  change (ProjectEulerSolutions.TriangleBufferCalls.encode buffer)[buffer.length - 1]? = _ at item
  simp [Mutable.getItem, found base buffer, indexed, Except.map, item]

theorem matrix_loop (cursor width : Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (rows : MatrixRows) (buffer : List Int)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (shape : ∀ pair ∈ rows, pair.2.length = width)
    (bounded : cursor ≤ rows.length) (positive : 0 < width) (size : buffer.length = width)
    (localMatrix : env "matrix" = some (.ref address))
    (localBuffer : env "dp" = some (.ref base.length))
    (localWidth : env "m" = some (number width)) :
    BlockReturns program env
      [.rangeStepLoop "i" cursor rows.length 1 matrixBody [], finalReturn] (working base buffer)
      (number ((remainingBuffer rows cursor buffer).getLast?.getD 0))
      (working base (remainingBuffer rows cursor buffer)) := by
  by_cases active : cursor < rows.length
  · let pair := rows[cursor]
    have rowSize : pair.2.length = width := shape pair (List.getElem_mem active)
    let indexed := setLocal env "i" (number cursor)
    let entered := setLocal indexed "row" (.ref pair.1)
    have readRow : Evaluates program indexed (.subscript (.name "matrix") (.name "i"))
        (working base buffer) (.ref pair.1) (working base buffer) := by
      apply Evaluates.subscript (by intros a b c impossible; cases impossible)
        (Evaluates.lookup (by simpa [indexed, setLocal] using localMatrix)) (Evaluates.lookup rfl)
      have read := Mutable.getItem_list (working base buffer) address cursor
        (rows.map (fun pair => .ref pair.1)) (prefix_found base buffer address _ getMatrix) (by simpa using active)
      rw [List.getElem_map] at read
      exact read
    have first := first_column_result program base entered pair.1 buffer pair.2
      (getRows pair (List.getElem_mem active)) (by omega) (by omega) rfl
      (by simpa [entered, indexed, setLocal] using localBuffer)
    obtain ⟨after, columns, kept⟩ := row_loop program base entered pair.1 1 buffer pair.2
      (getRows pair (List.getElem_mem active)) (by omega) (by omega) (by omega) rfl
      (by simpa [entered, indexed, setLocal] using localBuffer)
    have columnBlock : BlockContinues program entered [columnFor]
        (working base (partialBuffer (rowStep buffer pair.2) buffer 1)) after
        (working base (rowStep buffer pair.2)) := by
      apply BlockContinues.for_range
        (.cons Evaluates.constant (.cons (Evaluates.lookup (by simpa [entered, indexed, setLocal] using localWidth)) .nil)) rfl
      simpa only [rowSize] using columns
    have updatedSize : (rowStep buffer pair.2).length = width := by simp [size, rowSize]
    have continued := matrix_loop (cursor + 1) width base after address rows (rowStep buffer pair.2)
      getMatrix getRows shape (by omega) positive updatedSize
      (by rw [kept "matrix" (by decide)]; simpa [entered, indexed, setLocal] using localMatrix)
      (by rw [kept "dp" (by decide)]; simpa [entered, indexed, setLocal] using localBuffer)
      (by rw [kept "m" (by decide)]; simpa [entered, indexed, setLocal] using localWidth)
    have same : remainingBuffer rows cursor buffer = remainingBuffer rows (cursor + 1) (rowStep buffer pair.2) := by
      simp only [remainingBuffer, List.drop_eq_getElem_cons active, List.map_cons, List.foldl_cons]
      rfl
    rw [same]
    apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockReturns.assign readRow
    simpa [matrixBody, columnFor, firstColumn, rowBody, Calls.loopContinuation, Calls.loopControl,
      Calls.loopControlStmt, List.append_assoc] using (first.append columnBlock).returns continued
  · have done : cursor = rows.length := by omega
    subst cursor
    simp only [remainingBuffer, List.drop_length, List.map_nil, List.foldl_nil]
    apply BlockReturns.range_done (by simp [Calls.rangeContinues])
    exact BlockReturns.return_value (last_result base env buffer (by omega) localBuffer)
termination_by rows.length - cursor
decreasing_by omega

end solvers.«81»
