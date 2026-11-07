import solvers.«12.py.lean.d».Source
import ProjectEulerStatements.Lib.FactorDivisors

namespace solvers.«12».runtimeCost
open FormalSW.PyAst
def factorMethod : List Arg × List Stmt :=
  (findFunctionIn ((program.findClassBody "PrimeTable").getD []) "divisor_count").getD ([], [])

def factorIteration : List Stmt :=
  match ((executableBody factorMethod.2)[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

def stripLoop : Stmt :=
  match (factorIteration[1]? : Option Stmt) with
  | some (.ifThen _ body _) => body[1]?.getD (.expr (.constant .none))
  | _ => .expr (.constant .none)


end solvers.«12».runtimeCost
