import «.formalsw».solvers.«21.py.ast»
import lang.py.FormalSW.HeapCalls

/-! Select the two stripping loops from the generated factorization helper.
The literal-two and named-candidate cases share only their execution shape. -/
namespace solvers.«21».runtimeCost.Factors
open FormalSW.PyAst

abbrev program := Solvers.P21Ast.module
def number (value : Int) : Mutable.Value := .scalar (.int value)

def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "proper_divisor_sum_factorization").getD ([], [])

def oddLoop : Stmt :=
  (executableBody definition.2)[5]?.getD (.expr (.constant .none))

def oddBody : List Stmt :=
  match oddLoop with
  | .whileDo _ body _ => body
  | _ => []

def oddCondition : Expr :=
  match oddLoop with
  | .whileDo condition _ _ => condition
  | _ => .constant .none

def passStatement (named : Bool) : Stmt :=
  if named then oddBody[0]?.getD (.expr (.constant .none))
  else (executableBody definition.2)[3]?.getD (.expr (.constant .none))

def passBody (named : Bool) : List Stmt :=
  match passStatement named with
  | .ifThen _ body _ => body
  | _ => []

def stripLoop (named : Bool) : Stmt :=
  (passBody named)[2]?.getD (.expr (.constant .none))

def stripBody (named : Bool) : List Stmt :=
  match stripLoop named with
  | .whileDo _ body _ => body
  | _ => []

def stripCondition (named : Bool) : Expr :=
  match stripLoop named with
  | .whileDo condition _ _ => condition
  | _ => .constant .none

def factorExpr (named : Bool) : Expr :=
  match ((stripBody named)[0]? : Option Stmt) with
  | some (.augAssign _ _ factor) => factor
  | _ => .constant .none

theorem factorExpr_eq (named : Bool) :
    factorExpr named = if named then .name "p" else .constant (.int 2) := by
  cases named <;> rfl

theorem stripCondition_eq (named : Bool) : stripCondition named =
    .compare (.binOp (.name "n") .mod (factorExpr named)) [(.eq, .constant (.int 0))] := by
  cases named <;> rfl

theorem stripBody_eq (named : Bool) : stripBody named =
    [.augAssign (.name "n") .floorDiv (factorExpr named),
      .augAssign (.name "p_pow") .mult (factorExpr named),
      .augAssign (.name "term") .add (.name "p_pow")] := by
  cases named <;> rfl

theorem stripLoop_eq (named : Bool) : stripLoop named =
    .whileDo (stripCondition named) (stripBody named) [] := by
  cases named <;> rfl

theorem stripBody_continuation (named : Bool) (next : Stmt) :
    Calls.loopContinuation (stripBody named) next = next := by
  cases named <;> rfl

theorem passBody_eq (named : Bool) : passBody named =
    [.assign [.name "term"] (.constant (.int 1)),
      .assign [.name "p_pow"] (.constant (.int 1)), stripLoop named,
      .augAssign (.name "sigma") .mult (.name "term")] := by
  cases named <;> rfl

theorem passStatement_eq (named : Bool) : passStatement named =
    .ifThen (stripCondition named) (passBody named) [] := by
  cases named <;> rfl

theorem oddCondition_eq : oddCondition =
    .compare (.binOp (.name "p") .mult (.name "p")) [(.ltE, .name "n")] := rfl

theorem oddBody_eq : oddBody =
    [passStatement true, .augAssign (.name "p") .add (.constant (.int 2))] := rfl

theorem oddLoop_eq : oddLoop = .whileDo oddCondition oddBody [] := rfl

/-- The constant case can only denote two; the named case reads local `p`. -/
def FactorValue (named : Bool) (p : Nat) (env : HeapCalls.Env) : Prop :=
  if named then env "p" = some (number p) else p = 2

theorem FactorValue.preserve {named : Bool} {p : Nat} {env next : HeapCalls.Env}
    (factor : FactorValue named p env) (same : next "p" = env "p") :
    FactorValue named p next := by
  cases named with
  | false => exact factor
  | true => exact same.trans factor

end solvers.«21».runtimeCost.Factors
