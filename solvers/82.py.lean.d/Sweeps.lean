import solvers.«82.py.lean.d».Up

namespace solvers.«82»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P82
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls
open ProjectEulerSolutions.GridPathColumnCalls
set_option Elab.async false

abbrev downFor : Stmt := .forIn (.name "i")
  (.call (.name "range") [.constant (.int 1), .name "n"]) (relaxation downCandidate) []
abbrev upFor : Stmt := .forIn (.name "i")
  (.call (.name "range") [.binOp (.name "n") .sub (.constant (.int 2)),
    .unaryOp .uSub (.constant (.int 1)), .unaryOp .uSub (.constant (.int 1))]) (relaxation upCandidate) []

theorem sweeps_result (base : Mutable.Heap) (env : HeapCalls.Env) (address col : Nat)
    (rows : MatrixRows) (buffer : List Int)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (bounds : ∀ pair ∈ rows, col < pair.2.length) (size : buffer.length = rows.length)
    (positive : 0 < rows.length)
    (localMatrix : env "matrix" = some (.ref address)) (localBuffer : env "temp" = some (.ref base.length))
    (localColumn : env "j" = some (number col)) (localHeight : env "n" = some (number rows.length)) :
    ∃ after, BlockContinues program env [downFor, upFor] (working base buffer) after
      (working base (relaxUp (relaxDown buffer (column rows col)) (column rows col))) ∧
      ∀ name, name ≠ "i" → name ≠ "cand" → after name = env name := by
  have same : buffer.length = (column rows col).length := by simpa [column] using size
  let down := relaxDown buffer (column rows col)
  have downSize : down.length = rows.length := (relaxDown_length buffer _ same).trans size
  have downEqual : remainingDown rows col 1 buffer = down := by
    have equal := ProjectEulerSolutions.P82.downBuffer_eq buffer (column rows col) same
    unfold ProjectEulerSolutions.P82.downBuffer at equal
    rw [if_pos same] at equal
    simpa only [remainingDown, size, down] using equal
  obtain ⟨middle, downLoop, downKept⟩ := down_loop base env address 1 col rows buffer getMatrix getRows bounds size
    (by omega) (by omega) localMatrix localBuffer localColumn
  rw [downEqual] at downLoop
  have downBlock : BlockContinues program env [downFor] (working base buffer) middle (working base down) :=
    BlockContinues.for_range (.cons Evaluates.constant (.cons (Evaluates.lookup localHeight) .nil)) rfl downLoop
  obtain ⟨after, upLoop, upKept⟩ := up_loop base middle address (rows.length - 1) col rows down getMatrix getRows bounds downSize
    (by omega)
    (by rw [downKept "matrix" (by decide) (by decide)]; exact localMatrix)
    (by rw [downKept "temp" (by decide) (by decide)]; exact localBuffer)
    (by rw [downKept "j" (by decide) (by decide)]; exact localColumn)
  have sameDown : down.length = (column rows col).length := by simpa [column] using downSize
  have upEqual : remainingUp rows col (rows.length - 1) down = relaxUp down (column rows col) := by
    have equal := (ProjectEulerSolutions.P82.upBuffer_reverse_range down (column rows col) sameDown).symm.trans
      (ProjectEulerSolutions.P82.upBuffer_eq down (column rows col) sameDown)
    simpa only [remainingUp, downSize] using equal
  rw [upEqual] at upLoop
  have start : ((rows.length - 1 : Nat) : Int) - 1 = (rows.length : Int) - 2 := by omega
  rw [start] at upLoop
  have upBlock : BlockContinues program middle [upFor] (working base down) after
      (working base (relaxUp down (column rows col))) := by
    apply BlockContinues.for_range (values := [number ((rows.length : Int) - 2), number (-1), number (-1)])
    · exact .cons (Evaluates.binary
        (Evaluates.lookup (by rw [downKept "n" (by decide) (by decide)]; exact localHeight)) Evaluates.constant rfl)
        (.cons (Evaluates.unary Evaluates.constant rfl) (.cons (Evaluates.unary Evaluates.constant rfl) .nil))
    · rfl
    · exact upLoop
  refine ⟨after, downBlock.append upBlock, ?_⟩
  intro name notIndex notCandidate
  rw [upKept name notIndex notCandidate, downKept name notIndex notCandidate]

end solvers.«82»
