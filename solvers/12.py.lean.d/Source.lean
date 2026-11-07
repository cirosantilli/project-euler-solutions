import «.formalsw».solvers.«12.py.ast»
import lang.py.FormalSW.ClassCalls
import ProjectEulerStatements.Lib.PrimeTrialDivision

namespace solvers.«12»
open FormalSW.PyAst
private abbrev program := Solvers.P12Ast.module

def origin : String := (program.classOrigins.lookup "PrimeTable").getD ""

def primeValues (ps : List Nat) : List Mutable.Value :=
  ps.map fun p : Nat => .scalar (.int (p : Int))

/-- The table and its prime list retain distinct, stable heap addresses. -/
def TableView (heap : Mutable.Heap) (address : Nat) (ps : List Nat) : Prop :=
  ∃ storage, heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]) ∧
    heap[storage]? = some (.list (primeValues ps))

def initialized (heap : Mutable.Heap) : Mutable.Heap :=
  ((heap ++ [Mutable.Object.instance origin "PrimeTable" []]) ++ [Mutable.Object.list (primeValues [2])]).set heap.length
    (.instance origin "PrimeTable" [("primes", .ref (heap.length + 1))])

theorem initialized_eq (heap : Mutable.Heap) : initialized heap = heap ++
    [.instance origin "PrimeTable" [("primes", .ref (heap.length + 1))], .list (primeValues [2])] := by
  simp [initialized, List.append_assoc, List.set_append_right]

namespace runtimeCost
abbrev program := Solvers.P12Ast.module

def initMethod : List Arg × List Stmt :=
  (findFunctionIn ((program.findClassBody "PrimeTable").getD []) "__init__").getD ([], [])
def primeMethod : List Arg × List Stmt :=
  (findFunctionIn ((program.findClassBody "PrimeTable").getD []) "_is_prime").getD ([], [])
def primeIteration : List Stmt := match primeMethod.2 with
  | [_, .forIn _ _ body _, _] => body | _ => []

def ensureMethod : List Arg × List Stmt :=
  (findFunctionIn ((program.findClassBody "PrimeTable").getD []) "ensure_up_to").getD ([], [])
def ensureIteration : List Stmt :=
  match ((executableBody ensureMethod.2)[3]? : Option Stmt) with
  | some (.whileDo _ body _) => body | _ => []
abbrev lastExpression : Expr :=
  .subscript (.attribute (.name "self") "primes") (.unaryOp .uSub (.constant (.int 1)))
abbrev ensureLoop : Stmt :=
  .whileDo (.compare lastExpression [(.lt, .name "n")]) ensureIteration []

end runtimeCost
end solvers.«12»
