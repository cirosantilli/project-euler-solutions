import solvers.«82.py.lean.d».Down

namespace solvers.«82»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls
open ProjectEulerSolutions.GridPathColumnCalls
set_option Elab.async false

theorem up_step (base : Mutable.Heap) (env : HeapCalls.Env) (address index col : Nat)
    (rows : MatrixRows) (buffer : List Int)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (bounds : ∀ pair ∈ rows, col < pair.2.length) (size : buffer.length = rows.length)
    (inside : index + 1 < rows.length)
    (localMatrix : env "matrix" = some (.ref address)) (localBuffer : env "temp" = some (.ref base.length))
    (localIndex : env "i" = some (number index)) (localColumn : env "j" = some (number col)) :
    BlockContinues program env (relaxation upCandidate) (working base buffer)
      (setLocal env "cand" (number (buffer.getD (index + 1) 0 + (column rows col).getD index 0)))
      (working base (ProjectEulerSolutions.P82.upUpdate (column rows col) buffer index)) := by
  apply relaxation_result program base env buffer index upCandidate _ (by omega) localBuffer localIndex
  have nextIndex : Evaluates program env (.binOp (.name "i") .add (.constant (.int 1)))
      (working base buffer) (number (index + 1 : Nat)) (working base buffer) :=
    Evaluates.binary (Evaluates.lookup localIndex) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
  have next := index_evaluates program (working base buffer) env "temp" base.length (index + 1) buffer
    (buffer.getD (index + 1) 0) _ localBuffer nextIndex (by intros a b c impossible; cases impossible)
    (found base buffer) (getD_some buffer _ (by omega))
  have cell := matrix_cell program (working base buffer) env address index col rows
    (prefix_found base buffer address _ getMatrix)
    (fun pair member => prefix_found base buffer pair.1 _ (getRows pair member)) (by omega)
    (bounds _ (List.getElem_mem (by omega))) localMatrix localIndex localColumn
  rw [column_getD rows index col (by omega)]
  exact Evaluates.binary next cell rfl

def remainingUp (rows : MatrixRows) (col count : Nat) (buffer : List Int) : List Int :=
  (List.range count).reverse.foldl (ProjectEulerSolutions.P82.upUpdate (column rows col)) buffer

theorem up_loop (base : Mutable.Heap) (env : HeapCalls.Env) (address count col : Nat)
    (rows : MatrixRows) (buffer : List Int)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (bounds : ∀ pair ∈ rows, col < pair.2.length) (size : buffer.length = rows.length)
    (bounded : count < rows.length)
    (localMatrix : env "matrix" = some (.ref address)) (localBuffer : env "temp" = some (.ref base.length))
    (localColumn : env "j" = some (number col)) :
    ∃ after, BlockContinues program env
      [.rangeStepLoop "i" ((count : Int) - 1) (-1) (-1) (relaxation upCandidate) []] (working base buffer) after
      (working base (remainingUp rows col count buffer)) ∧
      ∀ name, name ≠ "i" → name ≠ "cand" → after name = env name := by
  cases count with
  | zero =>
      refine ⟨env, ?_, by intros; rfl⟩
      simp only [remainingUp, List.range_zero, List.reverse_nil, List.foldl_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
  | succ count =>
      let indexed := setLocal env "i" (number count)
      let value := buffer.getD (count + 1) 0 + (column rows col).getD count 0
      let ready := setLocal indexed "cand" (number value)
      let updated := ProjectEulerSolutions.P82.upUpdate (column rows col) buffer count
      have step := up_step base indexed address count col rows buffer getMatrix getRows bounds size bounded
        (by simpa [indexed, setLocal] using localMatrix)
        (by simpa [indexed, setLocal] using localBuffer) rfl
        (by simpa [indexed, setLocal] using localColumn)
      obtain ⟨after, continued, kept⟩ := up_loop base ready address count col rows updated getMatrix getRows bounds
        (by simpa [updated, ProjectEulerSolutions.P82.upUpdate] using size) (by omega)
        (by simpa [ready, indexed, setLocal] using localMatrix)
        (by simpa [ready, indexed, setLocal] using localBuffer)
        (by simpa [ready, indexed, setLocal] using localColumn)
      have next : remainingUp rows col (count + 1) buffer = remainingUp rows col count updated := by
        simp [remainingUp, List.range_succ, updated]
      refine ⟨after, ?_, ?_⟩
      · rw [next]
        simp only [Nat.cast_add, Nat.cast_one, Int.add_sub_cancel]
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [relaxation, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Int.sub_eq_add_neg] using step.append continued
      · intro name notIndex notCandidate
        rw [kept name notIndex notCandidate]
        simp [ready, indexed, setLocal, notIndex, notCandidate]
termination_by count

end solvers.«82»
