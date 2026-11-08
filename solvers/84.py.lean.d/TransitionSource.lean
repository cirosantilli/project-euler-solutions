import solvers.«84.py.lean.d».TransitionRowUpdate
import solvers.«84.py.lean.d».OutcomeSource

namespace solvers.«84».transition_row
open FormalSW.PyAst
set_option Elab.async false

def loopBody (statements : List Stmt) (index : Nat) : List Stmt := match statements[index]? with
  | some (.forIn _ _ body _) => body
  | _ => []

def code : List Stmt := loopBody (loopBody (loopBody (executableBody outcomes.definition.2) 5) 0) 1
def jailCode : List Stmt := match code[(0 : Nat)]? with
  | some (.ifThen _ body _) => body
  | _ => []
def landingCode : List Stmt := match code[(0 : Nat)]? with
  | some (.ifThen _ _ body) => body
  | _ => []

theorem jail_source : jailCode = [.assign [.name "target"] (.call (.name "encode_state") [.constant (.int 10), .constant (.int 0)]),
    increment (.binOp (.constant (.int 256)) .mult (.name "count"))] := rfl

theorem code_source : code = [.ifThen
    (.boolOp .and [.name "is_double", .compare (.name "doubles") [(.eq, .constant (.int 2))]]) jailCode landingCode] := rfl

end solvers.«84».transition_row
