import solvers.«88.py.lean.d».Arithmetic
import ProjectEulerSolutions.Lib.ProductSumTable
import ProjectEulerSolutions.Lib.HeapTables

namespace solvers.«88»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

abbrev encode (items : List Nat) : List Mutable.Value := items.map number
abbrev tableExpression : Expr := .subscript (.name "__formalsw_cell_solve_best") (.constant (.int 0))
def acceptedBody : List Stmt := match (dfsBody[4]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

theorem captured_table (heap : Mutable.Heap) (cell address : Nat) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_best" = some (.ref cell))
    (foundCell : heap[cell]? = some (.list [.ref address])) :
    Evaluates program env tableExpression heap (.ref address) heap := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCell) Evaluates.constant
  exact Mutable.getItem_list heap cell 0 [.ref address] foundCell (by simp)

theorem best_value (best : List Nat) (size : Nat) (inside : size < best.length)
    (heap : Mutable.Heap) (cell address : Nat) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_best" = some (.ref cell))
    (getSize : env "k" = some (number size))
    (foundCell : heap[cell]? = some (.list [.ref address]))
    (foundTable : heap[address]? = some (.list (encode best))) :
    Evaluates program env (.subscript tableExpression (.name "k")) heap
      (number (best[size]?.getD 0)) heap := by
  apply Evaluates.subscript (by intros; simp)
    (captured_table heap cell address env getCell foundCell) (Evaluates.lookup getSize)
  simpa [encode, inside] using
    Mutable.getItem_list heap address size (encode best) foundTable (by simpa using inside)

/-- The actual guarded write through the captured list implements one minimum-table update. -/
theorem improve (best : List Nat) (size value : Nat) (inside : size < best.length)
    (heap : Mutable.Heap) (cell address : Nat) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_best" = some (.ref cell))
    (getSize : env "k" = some (number size)) (getValue : env "new_prod" = some (number value))
    (foundCell : heap[cell]? = some (.list [.ref address]))
    (foundTable : heap[address]? = some (.list (encode best))) :
    BlockContinues program env (acceptedBody.take 1) heap env
      (heap.set address (.list (encode (relax best (size, value))))) := by
  have tested : Evaluates program env
      (.compare (.name "new_prod") [(.lt, .subscript tableExpression (.name "k"))]) heap
      (.scalar (.bool (decide (value < best[size]?.getD 0)))) heap := by
    apply Evaluates.compare (Evaluates.lookup getValue)
      (best_value best size inside heap cell address env getCell getSize foundCell foundTable)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  rw [relax_guarded best size value inside]
  by_cases smaller : value < best[size]?.getD 0
  · rw [if_pos smaller]
    apply BlockContinues.branch (flag := true) tested
      (by simp [Mutable.truth, scalarTruth, smaller])
    apply BlockContinues.setItem (value := number value) (receiver := .ref address)
      (subscript := number size) (by intros; simp) (Evaluates.lookup getValue)
      (.cons (captured_table heap cell address env getCell foundCell)
        (.cons (Evaluates.lookup getSize) .nil))
    simpa [encode, List.map_set] using
      ProjectEulerSolutions.HeapTables.setItem_list heap address size (encode best) (number value)
        foundTable (by simpa using inside)
  · rw [if_neg smaller]
    have unchanged : heap.set address (.list (encode best)) = heap := by
      obtain ⟨bound, stored⟩ := List.getElem?_eq_some_iff.mp foundTable
      rw [← stored, List.set_getElem_self bound]
    rw [unchanged]
    exact BlockContinues.branch (flag := false) tested
      (by simp [Mutable.truth, scalarTruth, smaller]) .nil

end solvers.«88»
