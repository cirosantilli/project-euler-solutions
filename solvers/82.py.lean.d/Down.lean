import solvers.«82.py.lean.d».Inputs
import ProjectEulerSolutions.Lib.GridPathColumnCalls

namespace solvers.«82»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls
open ProjectEulerSolutions.GridPathColumnCalls
set_option Elab.async false

abbrev column (rows : MatrixRows) (index : Nat) : List Int := rows.map (fun pair => pair.2.getD index 0)

theorem column_getD (rows : MatrixRows) (rowIndex columnIndex : Nat) (inside : rowIndex < rows.length) :
    (column rows columnIndex).getD rowIndex 0 = rows[rowIndex].2.getD columnIndex 0 := by
  simp [column, List.getD, List.getElem?_eq_getElem inside]

theorem down_step (base : Mutable.Heap) (env : HeapCalls.Env) (address index col : Nat)
    (rows : MatrixRows) (buffer : List Int)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (bounds : ∀ pair ∈ rows, col < pair.2.length) (size : buffer.length = rows.length)
    (positive : 0 < index) (inside : index < rows.length)
    (localMatrix : env "matrix" = some (.ref address)) (localBuffer : env "temp" = some (.ref base.length))
    (localIndex : env "i" = some (number index)) (localColumn : env "j" = some (number col)) :
    BlockContinues program env (relaxation downCandidate) (working base buffer)
      (setLocal env "cand" (number (buffer.getD (index - 1) 0 + (column rows col).getD index 0)))
      (working base (ProjectEulerSolutions.P82.downUpdate (column rows col) buffer index)) := by
  apply relaxation_result program base env buffer index downCandidate _ (by omega) localBuffer localIndex
  have previousIndex : Evaluates program env (.binOp (.name "i") .sub (.constant (.int 1)))
      (working base buffer) (number (index - 1 : Nat)) (working base buffer) :=
    Evaluates.binary (Evaluates.lookup localIndex) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number, Int.ofNat_sub (by omega : 1 ≤ index)])
  have previous := index_evaluates program (working base buffer) env "temp" base.length (index - 1) buffer
    (buffer.getD (index - 1) 0) _ localBuffer previousIndex (by intros a b c impossible; cases impossible)
    (found base buffer) (getD_some buffer _ (by omega))
  have cell := matrix_cell program (working base buffer) env address index col rows
    (prefix_found base buffer address _ getMatrix)
    (fun pair member => prefix_found base buffer pair.1 _ (getRows pair member)) inside
    (bounds _ (List.getElem_mem inside)) localMatrix localIndex localColumn
  rw [column_getD rows index col inside]
  exact Evaluates.binary previous cell rfl

def remainingDown (rows : MatrixRows) (col cursor : Nat) (buffer : List Int) : List Int :=
  (List.range' cursor (rows.length - cursor)).foldl (ProjectEulerSolutions.P82.downUpdate (column rows col)) buffer

theorem down_loop (base : Mutable.Heap) (env : HeapCalls.Env) (address cursor col : Nat)
    (rows : MatrixRows) (buffer : List Int)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (bounds : ∀ pair ∈ rows, col < pair.2.length) (size : buffer.length = rows.length)
    (positive : 0 < cursor) (bounded : cursor ≤ rows.length)
    (localMatrix : env "matrix" = some (.ref address)) (localBuffer : env "temp" = some (.ref base.length))
    (localColumn : env "j" = some (number col)) :
    ∃ after, BlockContinues program env
      [.rangeStepLoop "i" cursor rows.length 1 (relaxation downCandidate) []] (working base buffer) after
      (working base (remainingDown rows col cursor buffer)) ∧
      ∀ name, name ≠ "i" → name ≠ "cand" → after name = env name := by
  by_cases active : cursor < rows.length
  · let indexed := setLocal env "i" (number cursor)
    let value := buffer.getD (cursor - 1) 0 + (column rows col).getD cursor 0
    let ready := setLocal indexed "cand" (number value)
    let updated := ProjectEulerSolutions.P82.downUpdate (column rows col) buffer cursor
    have step := down_step base indexed address cursor col rows buffer getMatrix getRows bounds size positive active
      (by simpa [indexed, setLocal] using localMatrix)
      (by simpa [indexed, setLocal] using localBuffer) rfl
      (by simpa [indexed, setLocal] using localColumn)
    obtain ⟨after, continued, kept⟩ := down_loop base ready address (cursor + 1) col rows updated getMatrix getRows bounds
      (by simpa [updated, ProjectEulerSolutions.P82.downUpdate] using size) (by omega) (by omega)
      (by simpa [ready, indexed, setLocal] using localMatrix)
      (by simpa [ready, indexed, setLocal] using localBuffer)
      (by simpa [ready, indexed, setLocal] using localColumn)
    have next : remainingDown rows col cursor buffer = remainingDown rows col (cursor + 1) updated := by
      unfold remainingDown
      rw [show rows.length - cursor = (rows.length - (cursor + 1)) + 1 by omega, List.range'_succ, List.foldl_cons]
    refine ⟨after, ?_, ?_⟩
    · rw [next]
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [relaxation, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using step.append continued
    · intro name notIndex notCandidate
      rw [kept name notIndex notCandidate]
      simp [ready, indexed, setLocal, notIndex, notCandidate]
  · have done : cursor = rows.length := by omega
    subst cursor
    refine ⟨env, ?_, by intros; rfl⟩
    simp only [remainingDown, Nat.sub_self, List.range'_zero, List.foldl_nil]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
termination_by rows.length - cursor
decreasing_by omega

end solvers.«82»
