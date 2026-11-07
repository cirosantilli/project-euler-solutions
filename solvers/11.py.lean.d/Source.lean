import «.formalsw».solvers.«11.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p11.py»
import Mathlib.Tactic.IntervalCases

namespace solvers.«11»
open FormalSW.PyAst ProjectEulerStatements.Grid
abbrev GridView := data.«project-euler-statements».data.py.project_euler_statements.p11.GridView

/-- The four direction pairs are Python tuples stored in a Python list. -/
structure DirectionView (heap : Mutable.Heap) (address : Nat) (choices : List Direction) where
  pairs : List Nat
  found : heap[address]? = some (.list (pairs.map Mutable.Value.ref))
  size : pairs.length = choices.length
  contents : ∀ i (within : i < pairs.length), heap[pairs[i]]? = some (.tuple
    [.scalar (.int (choices[i]'(by omega)).rowStep), .scalar (.int (choices[i]'(by omega)).columnStep)])

def pairObject (direction : Direction) : Mutable.Object :=
  .tuple [.scalar (.int direction.rowStep), .scalar (.int direction.columnStep)]

def directionObjects (heap : Mutable.Heap) : Mutable.Heap :=
  directions.map pairObject ++ [.list ([heap.length, heap.length + 1, heap.length + 2, heap.length + 3].map Mutable.Value.ref)]

def directionView (heap : Mutable.Heap) :
    DirectionView (heap ++ directionObjects heap) (heap.length + 4) directions := by
  refine ⟨[heap.length, heap.length + 1, heap.length + 2, heap.length + 3], ?_, rfl, ?_⟩
  · simp [directionObjects, directions]
  · intro i within
    have bound : i < 4 := within
    interval_cases i <;> simp [directionObjects, directions, pairObject]

namespace runtimeCost
abbrev program := Solvers.P11Ast.module

def sourceBody : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
def rowsBody : List Stmt := match sourceBody with
  | [_, _, _, _, .forIn _ _ body _, _] => body
  | _ => []
def columnsBody : List Stmt := match rowsBody with | [.forIn _ _ body _] => body | _ => []
def directionBody : List Stmt := match columnsBody with | [.forIn _ _ body _] => body | _ => []
def candidateBody : List Stmt := match directionBody with | [_, _, .ifThen _ body _] => body | _ => []
def productBody : List Stmt := match candidateBody with | [_, .forIn _ _ body _, _] => body | _ => []
def productExpr : Expr := match productBody with | [.augAssign _ _ expression] => expression | _ => .constant .none

end runtimeCost
end solvers.«11»
