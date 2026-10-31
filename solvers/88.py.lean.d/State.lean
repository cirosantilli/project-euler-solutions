import solvers.«88.py.lean.d».Table
import ProjectEulerSolutions.Lib.ProductSumSearchSteps

namespace solvers.«88»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

/-- The table can have live captured cells on either side of it. -/
structure Context where
  base : Mutable.Heap
  tail : Mutable.Heap
  bestCell : Nat
  maxCell : Nat
  limitCell : Nat

def working (context : Context) (best : List Nat) : Mutable.Heap :=
  ProjectEulerSolutions.HeapCollection.working context.base (encode best) context.tail

theorem working_found (context : Context) (best : List Nat) :
    (working context best)[context.base.length]? = some (.list (encode best)) :=
  ProjectEulerSolutions.HeapCollection.found context.base (encode best) context.tail

theorem working_store (context : Context) (before after : List Nat) :
    (working context before).set context.base.length (.list (encode after)) = working context after :=
  ProjectEulerSolutions.HeapCollection.store context.base (encode before) (encode after) context.tail

/-- Captures remain fixed as the shared best-value table changes. -/
structure Cells (context : Context) (maxSize limit : Nat) : Prop where
  best : ∀ best, (working context best)[context.bestCell]? = some (.list [.ref context.base.length])
  maximum : ∀ best, (working context best)[context.maxCell]? = some (.list [number maxSize])
  bound : ∀ best, (working context best)[context.limitCell]? = some (.list [number limit])

structure Locals (env : HeapCalls.Env) (context : Context) (before : List Nat) : Prop where
  product : env "prod" = some (number before.prod)
  summ : env "summ" = some (number before.sum)
  terms : env "terms" = some (number before.length)
  best : env "__formalsw_cell_solve_best" = some (.ref context.bestCell)
  maximum : env "__formalsw_cell_solve_k_max" = some (.ref context.maxCell)
  bound : env "__formalsw_cell_solve_limit" = some (.ref context.limitCell)

theorem Locals.measured {env : HeapCalls.Env} {context : Context} {before : List Nat}
    (ready : Locals env context before) (factor : Nat) :
    Locals (measuredEnv env before factor) context before := by
  rcases ready with ⟨hp, hs, ht, hb, hm, hl⟩
  constructor <;> simp_all [measuredEnv, setLocal]

theorem Locals.factor {env : HeapCalls.Env} {context : Context} {before : List Nat}
    (ready : Locals env context before) (factor : Nat) :
    Locals (setLocal env "f" (number factor)) context before := by
  rcases ready with ⟨hp, hs, ht, hb, hm, hl⟩
  constructor <;> simp_all [setLocal]

theorem captured_number (heap : Mutable.Heap) (cell value : Nat) (name : String) (env : HeapCalls.Env)
    (getCell : env name = some (.ref cell)) (found : heap[cell]? = some (.list [number value])) :
    Evaluates program env (.subscript (.name name) (.constant (.int 0))) heap (number value) heap := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCell) Evaluates.constant
  exact Mutable.getItem_list heap cell 0 [number value] found (by simp)

abbrev sizeTest : Expr := .compare (.name "k")
  [(.ltE, .subscript (.name "__formalsw_cell_solve_k_max") (.constant (.int 0)))]

theorem size_evaluate (size maxSize limit : Nat) (context : Context) (cells : Cells context maxSize limit)
    (best : List Nat) (env : HeapCalls.Env) (getSize : env "k" = some (number size))
    (getMax : env "__formalsw_cell_solve_k_max" = some (.ref context.maxCell)) :
    Evaluates program env sizeTest (working context best)
      (.scalar (.bool (decide (size ≤ maxSize)))) (working context best) := by
  apply Evaluates.compare (Evaluates.lookup getSize)
    (captured_number (working context best) context.maxCell maxSize "__formalsw_cell_solve_k_max"
      env getMax (cells.maximum best))
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

def SearchReturns (maxSize limit start : Nat) (before : List Nat) (context : Context) (best : List Nat) : Prop :=
  ReturnsValue program "__formalsw_local_solve_dfs"
    [number start, number before.prod, number before.sum, number before.length,
      .ref context.bestCell, .ref context.maxCell, .ref context.limitCell]
    (working context best) (.scalar .none)
    (working context (accumulate (search maxSize limit start before.prod before.sum before.length) best))

def Descendants (maxSize limit : Nat) (before : List Nat) (context : Context) : Prop :=
  ∀ factor, 2 ≤ factor → before.prod * factor ≤ limit →
    ∀ best : List Nat, best.length = maxSize + 1 →
      SearchReturns maxSize limit factor (before ++ [factor]) context best

end solvers.«88»
