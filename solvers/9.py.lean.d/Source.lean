import «.formalsw».solvers.«9.py.ast»
import lang.py.FormalSW.HeapCalls

/-! The source fragments and environment updates shared by counted and runtime
proofs. Whole-function entry lemmas check these fragments against the Python AST. -/
namespace solvers.«9».runtimeSource
open FormalSW.PyAst

def triple (a b c : Int) : List Mutable.Value :=
  [.scalar (.int a), .scalar (.int b), .scalar (.int c)]

abbrev sortedExpression : Expr :=
  .builtinCall .sorted [.tuple [.name "a", .name "b", .name "c"]]

def sortedHeap (heap : Mutable.Heap) (a b c : Int) : Mutable.Heap :=
  heap ++ [.tuple (triple a b c), .list (triple (min a b) (max a b) c)]

abbrev outerCondition : Expr := .compare
  (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult
    (.binOp (.name "m") .add (.constant (.int 1)))) [(.ltE, .name "total")]

abbrev square (name : String) : Expr := .binOp (.name name) .mult (.name name)

abbrev triangleCondition : Expr := .boolOp .and
  [.compare (.name "a") [(.gt, .constant (.int 0))],
   .compare (.binOp (square "a") .add (square "b")) [(.eq, square "c")],
   .compare (.binOp (.binOp (.name "a") .add (.name "b")) .add (.name "c")) [(.eq, .name "total")]]

abbrev productExpression : Expr := .binOp (.binOp (.name "a") .mult (.name "b")) .mult (.name "c")

abbrev updateBest : List Stmt :=
  [.ifThen triangleCondition [.assign [.name "best"] (.builtinCall .max [.name "best", productExpression])] []]

abbrev kExpression : Expr := .binOp (.name "total") .floorDiv (.name "s")

abbrev aExpression : Expr := .binOp (.name "k") .mult (.binOp (square "m") .sub (square "n"))

abbrev bExpression : Expr := .binOp (.name "k") .mult
  (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult (.name "n"))

abbrev cExpression : Expr := .binOp (.name "k") .mult (.binOp (square "m") .add (square "n"))

abbrev coefficientBody : List Stmt :=
  [.assign [.name "k"] kExpression, .assign [.name "a"] aExpression,
   .assign [.name "b"] bExpression, .assign [.name "c"] cExpression]

def coefficientEnv (env : HeapCalls.Env) (total m n s : Int) :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal
    (HeapCalls.setLocal env "k" (.scalar (.int (total / s))))
    "a" (.scalar (.int (total / s * (m * m - n * n)))))
    "b" (.scalar (.int (total / s * (2 * m * n)))))
    "c" (.scalar (.int (total / s * (m * m + n * n))))

abbrev sortedStatement : Stmt :=
  .assign [.tuple [.name "a", .name "b", .name "c"]] sortedExpression

abbrev successBody : List Stmt := coefficientBody ++ [sortedStatement] ++ updateBest

abbrev sExpression : Expr :=
  .binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult (.binOp (.name "m") .add (.name "n"))

abbrev skipCondition : Expr :=
  .compare (.binOp (.name "total") .mod (.name "s")) [(.notEq, .constant (.int 0))]

abbrev innerBody : List Stmt :=
  [.assign [.name "s"] sExpression, .ifThen skipCondition [.continueLoop] []] ++ successBody

abbrev outerBody : List Stmt :=
  [.forIn (.name "n") (.call (.name "range") [.constant (.int 1), .name "m"]) innerBody [],
   .augAssign (.name "m") .add (.constant (.int 1))]

abbrev outerLoop : Stmt := .whileDo outerCondition outerBody []

abbrev errorMessage : Expr :=
  .binOp (.binOp (.constant (.string "")) .add (.constant (.string "No Pythagorean triplet sums to ")))
    .add (.builtinCall .str [.name "total"])

abbrev raiseError : Stmt := .raise (some (.call (.name "ValueError") [errorMessage]))

abbrev functionTail : List Stmt :=
  [.ifThen (.name "best") [.return (.name "best")] [], raiseError]

end solvers.«9».runtimeSource
