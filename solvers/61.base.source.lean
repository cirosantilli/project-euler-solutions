import «.formalsw».solvers.«61.py.ast»
import ProjectEulerSolutions.Lib.PolygonalCycleSearch
import lang.py.FormalSW.SumCalls

set_option Elab.async false

namespace solvers.«61».dfs
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def numbers (values : List Int) : List Mutable.Value := values.map number
def exhausted : Expr := .unaryOp .not (.name "remaining")
def closes : Expr := .compare
  (.binOp (.subscript (.name "path") (.unaryOp .uSub (.constant (.int 1)))) .mod (.constant (.int 100)))
  [(.eq, .binOp (.subscript (.name "path") (.constant (.int 0))) .floorDiv (.constant (.int 100)))]
def sumPath : Expr := .sumSequence (.name "path") (.constant (.int 0))
def baseBody : List Stmt := [.ifThen closes [.return sumPath] [], .return (.constant (.int 0))]

theorem base_source :
    (executableBody ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2)[0]? =
      some (.ifThen exhausted baseBody []) := rfl

/-- Exhausting the type list tests the actual endpoints and returns the actual path sum. -/
theorem closed_result (values : List Int) (nonempty : values ≠ []) (heap : Mutable.Heap) (path remaining : Nat)
    (pathFound : heap[path]? = some (.list (numbers values)))
    (remainingFound : heap[remaining]? = some (.list [])) (prefixCell : Mutable.Value)
    (globals : Mutable.Globals := []) :
    ReturnsValue program "__formalsw_local_solve_dfs" [.ref path, .ref remaining, prefixCell] heap
      (number (if ProjectEulerSolutions.PolygonalCycleSearch.closed values then values.sum else 0)) heap globals := by
  have positive : 0 < values.length := List.length_pos_iff.mpr nonempty
  let first := values[0]'positive
  let last := values[values.length - 1]'(by omega)
  have head : values[0]? = some first := List.getElem?_eq_getElem positive
  have tail : values[values.length - 1]? = some last := List.getElem?_eq_getElem (by omega)
  have closed : ProjectEulerSolutions.PolygonalCycleSearch.closed values =
      ProjectEulerSolutions.PolygonalCycle.links last first := by
    simp [ProjectEulerSolutions.PolygonalCycleSearch.closed, ProjectEulerSolutions.PolygonalCycle.links, List.head?_eq_getElem?, List.getLast?_eq_getElem?, head, tail]
  let params := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
  let env := functionEnv params sourceBody [.ref path, .ref remaining, prefixCell] globals
  have getPath : env "path" = some (.ref path) := by cases globals <;> rfl
  have getRemaining : env "remaining" = some (.ref remaining) := by cases globals <;> rfl
  have done : Evaluates program env exhausted heap (flag true) heap globals := by
    apply Evaluates.unary (middle := heap) (finalHeap := heap) (Evaluates.lookup getRemaining)
    simp [HeapCalls.unary, Mutable.truth, remainingFound, flag]
  have firstRun : Evaluates program env (.subscript (.name "path") (.constant (.int 0))) heap
      (number first) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getPath) Evaluates.constant
    have got := Mutable.getItem_list heap path 0 (numbers values) pathFound (by simp [numbers]; omega)
    simpa [numbers, first] using got
  have lastRun : Evaluates program env (.subscript (.name "path") (.unaryOp .uSub (.constant (.int 1)))) heap
      (number last) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getPath)
      (Evaluates.unary (middle := heap) (finalHeap := heap) (value := number 1) Evaluates.constant rfl)
    have offset : ((-1 : Int) + values.length).toNat = values.length - 1 := by omega
    have indexed : Mutable.index (numbers values) (number (-1)) heap = some (.ok (values.length - 1), heap) := by
      simp [Mutable.index, numbers, nonempty, offset]
    have mapped : (numbers values)[values.length - 1]? = some (number last) := by simp [numbers, tail]
    simp [Mutable.getItem, pathFound, indexed, Except.map, mapped]
  have tested : Evaluates program env closes heap (flag (ProjectEulerSolutions.PolygonalCycleSearch.closed values)) heap globals := by
    rw [closed]
    apply Evaluates.compare (av := number (last % 100)) (bv := number (first / 100))
      (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        lastRun Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        firstRun Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ProjectEulerSolutions.PolygonalCycle.links, flag]

  have summed : Evaluates program env sumPath heap (number values.sum) heap globals := by
    have evaluated := Evaluates.sum_integers (module := program) (globals := globals)
      (heap := heap) (middle := heap) (initial := 0) (values := values)
      (.cons (Evaluates.lookup getPath) (.cons Evaluates.constant .nil))
      (by simp [Mutable.sequence, pathFound, numbers])
    have total : values.sum = values.foldl Int.add 0 := List.sum_eq_foldl
    rw [← total] at evaluated
    exact evaluated
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (.ifThen exhausted baseBody [] :: (executableBody sourceBody).drop 1) heap _ heap globals
  apply BlockReturns.branch (flag := true) done rfl
  cases selected : ProjectEulerSolutions.PolygonalCycleSearch.closed values with
  | false =>
      simp only [Bool.false_eq_true, if_false]
      apply BlockReturns.branch (flag := false) (by simpa only [selected] using tested) rfl
      exact BlockReturns.return_value Evaluates.constant
  | true =>
      simp only [if_true]
      apply BlockReturns.branch (flag := true) (by simpa only [selected] using tested) rfl
      exact BlockReturns.return_value summed

end solvers.«61».dfs
