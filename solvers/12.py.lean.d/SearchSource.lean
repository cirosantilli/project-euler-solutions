import solvers.«12.py.lean.d».Source
import ProjectEulerStatements.Lib.TriangularDivisors

namespace solvers.«12».runtimeCost
open FormalSW.PyAst

def searchBody : List Stmt :=
  match ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? : Option Stmt) with
  | some (.whileDo _ body _) => body
  | _ => []

def splitEnv (env : HeapCalls.Env) (n : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal env "a" (.scalar (.int (ProjectEulerStatements.P12.splitLeft n))))
    "b" (.scalar (.int (ProjectEulerStatements.P12.splitRight n)))

abbrev searchLoop : List Stmt := [.whileDo (.constant (.bool true)) searchBody []]

end solvers.«12».runtimeCost
