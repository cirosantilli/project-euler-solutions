import solvers.«83.py.lean.d».Push
import solvers.«83.py.lean.d».LoopControl

namespace solvers.«83»
open FormalSW.PyAst

abbrev distanceAt : Expr := .subscript (.name "dist") (.name "v")
abbrev relaxationTest : Expr := .compare (.name "nd") [(.lt, distanceAt)]
abbrev relaxationCode : List Stmt := [.ifThen relaxationTest
  [.assign [distanceAt] (.name "nd"), .expr pushCandidate] []]

abbrev neighborBranch (guard index row column : Expr) : Stmt := .ifThen guard
  ([.assign [.name "v"] index,
    .assign [.name "nd"] (.binOp (.name "d") .add (.subscript (.subscript (.name "matrix") row) column))] ++
      relaxationCode) []

abbrev rowBefore : Expr := .binOp (.name "r") .sub (.constant (.int 1))
abbrev rowAfter : Expr := .binOp (.name "r") .add (.constant (.int 1))
abbrev columnBefore : Expr := .binOp (.name "c") .sub (.constant (.int 1))
abbrev columnAfter : Expr := .binOp (.name "c") .add (.constant (.int 1))
abbrev rowOffset : Expr := .binOp (.name "r") .mult (.name "m")

abbrev neighborsCode : List Stmt :=
  [neighborBranch (.compare (.name "r") [(.gt, .constant (.int 0))])
    (.binOp (.binOp rowBefore .mult (.name "m")) .add (.name "c")) rowBefore (.name "c"),
   neighborBranch (.compare rowAfter [(.lt, .name "n")])
    (.binOp (.binOp rowAfter .mult (.name "m")) .add (.name "c")) rowAfter (.name "c"),
   neighborBranch (.compare (.name "c") [(.gt, .constant (.int 0))])
    (.binOp (.binOp rowOffset .add (.name "c")) .sub (.constant (.int 1))) (.name "r") columnBefore,
   neighborBranch (.compare columnAfter [(.lt, .name "m")])
    (.binOp (.binOp rowOffset .add (.name "c")) .add (.constant (.int 1))) (.name "r") columnAfter]

theorem neighbors_source : loopBody.drop 5 = neighborsCode := rfl

end solvers.«83»
