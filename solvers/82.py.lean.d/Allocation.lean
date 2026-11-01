import solvers.«82.py.lean.d».Sweeps
import ProjectEulerSolutions.Lib.IntegerRangeCollection

namespace solvers.«82»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls
open ProjectEulerSolutions.GridPathColumnCalls
set_option Elab.async false

abbrev firstElement : Expr := .subscript (.subscript (.name "matrix") (.name "i")) (.constant (.int 0))
abbrev firstComprehension : Expr := .listComp firstElement [("i", .call (.name "range") [.name "n"], [])]
abbrev nextElement : Expr := .binOp (.subscript (.name "dp") (.name "i")) .add matrixCell
abbrev nextComprehension : Expr := .listComp nextElement [("i", .call (.name "range") [.name "n"], [])]

theorem first_allocation (base : Mutable.Heap) (env : HeapCalls.Env) (address : Nat) (rows : MatrixRows)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (bounds : ∀ pair ∈ rows, 0 < pair.2.length)
    (localMatrix : env "matrix" = some (.ref address)) (localHeight : env "n" = some (number rows.length)) :
    Evaluates program env firstComprehension base (.ref base.length) (working base (column rows 0)) := by
  apply ProjectEulerSolutions.IntegerRangeCollection.collect (program := program) (base := base) (env := env)
    (name := "i") (expressions := [.name "n"])
    (arguments := [number rows.length])
    (evaluated := .cons (Evaluates.lookup localHeight) .nil) (bounds := by simp only [column, List.length_map]; rfl)
  intro collected index inside
  have valid : index < rows.length := by simpa [column] using inside
  let boundEnv := setLocal (comprehensionEnv env [("i", .call (.name "range") [.name "n"], [])]) "i" (number index)
  have matrix : Evaluates program boundEnv (.subscript (.name "matrix") (.name "i")) (working base collected)
      (.ref rows[index].1) (working base collected) := by
    apply Evaluates.subscript (by intros a b c impossible; cases impossible)
      (Evaluates.lookup (by simpa [boundEnv, comprehensionEnv, setLocal] using localMatrix)) (Evaluates.lookup rfl)
    have read := Mutable.getItem_list (working base collected) address index (rows.map (fun pair => .ref pair.1))
      (prefix_found base collected address _ getMatrix) (by simpa using valid)
    rw [List.getElem_map] at read
    exact read
  apply Evaluates.subscript (by intros a b c impossible; cases impossible) matrix Evaluates.constant
  have positive := bounds rows[index] (List.getElem_mem valid)
  have read := Mutable.getItem_list (working base collected) rows[index].1 0 (encode rows[index].2)
    (prefix_found base collected rows[index].1 _ (getRows _ (List.getElem_mem valid))) (by simpa [encode] using positive)
  simpa [column, List.getElem_map, encode, List.getD, List.getElem?_eq_getElem positive] using read

theorem next_allocation (base : Mutable.Heap) (env : HeapCalls.Env) (address previousAddress col : Nat)
    (rows : MatrixRows) (previous : List Int)
    (getMatrix : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (getPrevious : base[previousAddress]? = some (.list (encode previous)))
    (bounds : ∀ pair ∈ rows, col < pair.2.length) (size : previous.length = rows.length)
    (localMatrix : env "matrix" = some (.ref address)) (localHeight : env "n" = some (number rows.length))
    (localPrevious : env "dp" = some (.ref previousAddress)) (localColumn : env "j" = some (number col)) :
    Evaluates program env nextComprehension base (.ref base.length)
      (working base (ProjectEulerStatements.P82.addLists previous (column rows col))) := by
  have same : previous.length = (column rows col).length := by simpa [column] using size
  simp only [ProjectEulerStatements.P82.addLists, if_pos same]
  apply ProjectEulerSolutions.IntegerRangeCollection.collect (program := program) (base := base) (env := env)
    (name := "i") (expressions := [.name "n"])
    (arguments := [number rows.length])
    (evaluated := .cons (Evaluates.lookup localHeight) .nil) (bounds := by simp only [column, List.length_map, List.length_zipWith, size, Nat.min_self]; rfl)
  intro collected index inside
  have valid : index < rows.length := by simpa [column, size] using inside
  have inPrevious : index < previous.length := by omega
  let boundEnv := setLocal (comprehensionEnv env [("i", .call (.name "range") [.name "n"], [])]) "i" (number index)
  have old := index_evaluates program (working base collected) boundEnv "dp" previousAddress index previous previous[index]
    (.name "i") (by simpa [boundEnv, comprehensionEnv, setLocal] using localPrevious) (Evaluates.lookup rfl)
    (by intros a b c impossible; cases impossible) (prefix_found base collected previousAddress _ getPrevious)
    (List.getElem?_eq_getElem inPrevious)
  have cell := matrix_cell program (working base collected) boundEnv address index col rows
    (prefix_found base collected address _ getMatrix)
    (fun pair member => prefix_found base collected pair.1 _ (getRows pair member)) valid
    (bounds _ (List.getElem_mem valid))
    (by simpa [boundEnv, comprehensionEnv, setLocal] using localMatrix) rfl
    (by simpa [boundEnv, comprehensionEnv, setLocal] using localColumn)
  simpa only [List.getElem_zipWith, column, List.getElem_map] using Evaluates.binary old cell rfl

end solvers.«82»
