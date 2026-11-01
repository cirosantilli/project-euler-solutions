import ProjectEulerSolutions.Lib.GridPathColumnBuffers
import ProjectEulerSolutions.Lib.TriangleBufferCalls
import ProjectEulerSolutions.Lib.RectangularMatrixCalls

/-! Source evaluation of matrix cells and conditional vertical buffer relaxations. -/
namespace ProjectEulerSolutions.GridPathColumnCalls
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
set_option Elab.async false

abbrev matrixCell : Expr := .subscript (.subscript (.name "matrix") (.name "i")) (.name "j")
abbrev downCandidate : Expr := .binOp
  (.subscript (.name "temp") (.binOp (.name "i") .sub (.constant (.int 1)))) .add matrixCell
abbrev upCandidate : Expr := .binOp
  (.subscript (.name "temp") (.binOp (.name "i") .add (.constant (.int 1)))) .add matrixCell
abbrev relaxation (candidate : Expr) : List Stmt :=
  [.assign [.name "cand"] candidate,
   .ifThen (.compare (.name "cand") [(.lt, .subscript (.name "temp") (.name "i"))])
     [.assign [.subscript (.name "temp") (.name "i")] (.name "cand")] []]

theorem getD_some (values : List Int) (index : Nat) (inside : index < values.length) :
    values[index]? = some (values.getD index 0) := by
  simp [List.getD, List.getElem?_eq_getElem inside]

theorem matrix_cell (program : Module) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address rowIndex columnIndex : Nat) (rows : MatrixRows)
    (getMatrix : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (inside : rowIndex < rows.length) (inRow : columnIndex < rows[rowIndex].2.length)
    (localMatrix : env "matrix" = some (.ref address))
    (localRow : env "i" = some (number rowIndex)) (localColumn : env "j" = some (number columnIndex)) :
    Evaluates program env matrixCell heap (number (rows[rowIndex].2.getD columnIndex 0)) heap := by
  have pair : Evaluates program env (.subscript (.name "matrix") (.name "i")) heap (.ref rows[rowIndex].1) heap := by
    apply Evaluates.subscript (by intros a b c impossible; cases impossible)
      (Evaluates.lookup localMatrix) (Evaluates.lookup localRow)
    have read := Mutable.getItem_list heap address rowIndex (rows.map (fun pair => .ref pair.1)) getMatrix (by simpa using inside)
    rw [List.getElem_map] at read
    exact read
  apply Evaluates.subscript (by intros a b c impossible; cases impossible) pair (Evaluates.lookup localColumn)
  have read := Mutable.getItem_list heap rows[rowIndex].1 columnIndex (encode rows[rowIndex].2)
    (getRows _ (List.getElem_mem inside)) (by simpa [encode] using inRow)
  simpa [encode, List.getD, List.getElem?_eq_getElem inRow] using read

theorem relaxation_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (buffer : List Int) (index : Nat) (candidate : Expr) (value : Int)
    (inside : index < buffer.length) (localBuffer : env "temp" = some (.ref base.length))
    (localIndex : env "i" = some (number index))
    (evaluated : Evaluates program env candidate (working base buffer) (number value) (working base buffer)) :
    BlockContinues program env (relaxation candidate) (working base buffer)
      (setLocal env "cand" (number value))
      (working base (buffer.set index (min (buffer.getD index 0) value))) := by
  let ready := setLocal env "cand" (number value)
  have getBuffer : ready "temp" = some (.ref base.length) := by simpa [ready, setLocal] using localBuffer
  have getIndex : ready "i" = some (number index) := by simpa [ready, setLocal] using localIndex
  have old := index_evaluates program (working base buffer) ready "temp" base.length index buffer
    (buffer.getD index 0) (.name "i") getBuffer (Evaluates.lookup getIndex)
    (by intros a b c impossible; cases impossible) (found base buffer) (getD_some buffer index inside)
  have tested : Evaluates program ready (.compare (.name "cand") [(.lt, .subscript (.name "temp") (.name "i"))])
      (working base buffer) (.scalar (.bool (decide (value < buffer.getD index 0)))) (working base buffer) := by
    apply Evaluates.compare (Evaluates.lookup rfl) old
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have branch : BlockContinues program ready (relaxation candidate).tail (working base buffer) ready
      (working base (buffer.set index (min (buffer.getD index 0) value))) := by
    by_cases smaller : value < buffer.getD index 0
    · rw [Int.min_eq_right (by omega : value ≤ buffer.getD index 0)]
      apply BlockContinues.branch (flag := true) (by simpa only [smaller, decide_true] using tested) rfl
      exact BlockContinues.setItem (by intros a b c impossible; cases impossible) (Evaluates.lookup rfl)
        (.cons (Evaluates.lookup getBuffer) (.cons (Evaluates.lookup getIndex) .nil))
        (set_item base buffer index value inside)
    · rw [Int.min_eq_left (by omega : buffer.getD index 0 ≤ value)]
      have unchanged : buffer.set index (buffer.getD index 0) = buffer := by
        simp [List.getD, List.getElem?_eq_getElem inside]
      rw [unchanged]
      exact BlockContinues.branch (flag := false) (by simpa only [smaller, decide_false] using tested) rfl BlockContinues.nil
  exact (BlockContinues.assign evaluated).append branch

end ProjectEulerSolutions.GridPathColumnCalls
