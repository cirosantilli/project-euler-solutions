import solvers.«14.py.lean.d».State

/-! Select bodies from the generated Python AST for both functional and runtime proofs. -/
namespace solvers.«14»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

def walkBody : List Stmt :=
  match findFunctionIn program.body "collatz_length" with
  | some (_, sourceBody) =>
      let statements : List Stmt := executableBody sourceBody
      match statements[2]? with
      | some (Stmt.whileDo _ body _) => body
      | _ => []
  | _ => []

def advanceBody : List Stmt := walkBody.drop 2

def unwindBody : List Stmt :=
  match findFunctionIn program.body "collatz_length" with
  | some (_, sourceBody) =>
      let statements : List Stmt := executableBody sourceBody
      match statements[3]? with
      | some (Stmt.forReversed _ _ body _) => body
      | _ => []
  | _ => []

def storeBody : List Stmt := unwindBody.drop 1

def walkLoop : Stmt := .whileDo (.constant (.bool true)) walkBody []

def probeBody : List Stmt := walkBody.take 1

def lengthDefinition : List Arg × List Stmt :=
  (findFunctionIn program.body "collatz_length").getD ([], [])

def lengthReturnBody : List Stmt := (executableBody lengthDefinition.2).drop 4

def searchDefinition : List Arg × List Stmt :=
  (findFunctionIn program.body "longest_collatz_under").getD ([], [])

def searchBody : List Stmt :=
  match ((executableBody searchDefinition.2)[5]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

def searchEnv (limit : Nat) : Mutable.Env :=
  setLocal (setLocal (setLocal (setLocal
    (functionEnv searchDefinition.1 searchDefinition.2 [.scalar (.int limit)] [])
    "cache_small" (.ref 1)) "cache_big" (.ref 2)) "best_n" (.scalar (.int 1))) "best_len" (.scalar (.int 1))

end solvers.«14»
