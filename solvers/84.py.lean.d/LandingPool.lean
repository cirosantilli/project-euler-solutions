import solvers.«84.py.lean.d».Landing
import ProjectEulerSolutions.Lib.TuplePool

namespace solvers.«84».resolve_landing.pool
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false
set_option maxRecDepth 2048

abbrev definition := (findFunctionIn program.body "resolve_landing").getD ([], [])
def branch (index : Nat) : List Stmt := match (executableBody definition.2)[index]? with
  | some (.ifThen _ body _) => body
  | _ => []
def displayKeys (body : List Stmt) : List Expr := match body[(0 : Nat)]? with
  | some (.return (.dict pairs)) | some (.assign _ (.dict pairs)) => pairs.map Prod.fst
  | _ => []
def jailKey : Expr := (displayKeys (branch 0))[0]?.getD (.constant .none)
def communityGo : Expr := (displayKeys (branch 1))[1]?.getD (.constant .none)
def communityJail : Expr := (displayKeys (branch 1))[2]?.getD (.constant .none)
def chanceJail : Expr := (displayKeys (branch 2))[1]?.getD (.constant .none)
def backThree : Expr := match (branch 2)[(1 : Nat)]? with
  | some (.forIn _ _ [.ifThen _ [.forIn _ iterable _ _] _] _) => iterable
  | _ => .constant .none
def elements (expression : Expr) : List Expr := (HeapMemoCalls.pooledTupleParts expression).getD ("", []) |>.2
def identity (expression : Expr) : String := (HeapMemoCalls.pooledIdentity expression).getD ""
def rowExpression (index : Fin 3) : Expr := (elements backThree)[index.val]?.getD (.constant .none)
def outcomeExpression (index : Fin 3) : Expr := (elements (rowExpression index))[0]?.getD (.constant .none)

def square (index : Fin 3) : Int := if index = 0 then 33 else if index = 1 then 0 else 10
def jailed (index : Fin 3) : Bool := decide (index = 2)
def weight (index : Fin 3) : Int := if index = 0 then 14 else 1
def pair (square : Int) (jailed : Bool) : List HashKey := [.scalar (.int square), .scalar (.bool jailed)]
def row (index : Fin 3) : HashKey := .tuple [.tuple (pair (square index) (jailed index)), .scalar (.int (weight index))]

def catalog : TuplePool.Catalog := fun query => List.lookup query
  ([(identity jailKey, pair 10 true), (identity communityGo, pair 0 false),
    (identity communityJail, pair 10 true), (identity chanceJail, pair 10 true),
    (identity backThree, List.ofFn row)] ++
    (List.ofFn fun index : Fin 3 =>
      [(identity (rowExpression index), [.tuple (pair (square index) (jailed index)), .scalar (.int (weight index))]),
       (identity (outcomeExpression index), pair (square index) (jailed index))]).flatten)

abbrev spec := TuplePool.spec catalog

theorem jail_source : jailKey = HeapMemoCalls.pooledTuple (identity jailKey) [.constant (.int 10), .constant (.bool true)] := rfl
theorem community_go_source : communityGo = HeapMemoCalls.pooledTuple (identity communityGo) [.constant (.int 0), .constant (.bool false)] := rfl
theorem community_jail_source : communityJail = HeapMemoCalls.pooledTuple (identity communityJail) [.constant (.int 10), .constant (.bool true)] := rfl
theorem chance_jail_source : chanceJail = HeapMemoCalls.pooledTuple (identity chanceJail) [.constant (.int 10), .constant (.bool true)] := rfl
theorem back_source : backThree = HeapMemoCalls.pooledTuple (identity backThree) (List.ofFn rowExpression) := rfl

theorem outcome_source (index : Fin 3) : outcomeExpression index = HeapMemoCalls.pooledTuple (identity (outcomeExpression index))
    [.constant (.int (square index)), .constant (.bool (jailed index))] := by
  rcases index with ⟨index, inside⟩
  have cases : index = 0 ∨ index = 1 ∨ index = 2 := by omega
  rcases cases with rfl | rfl | rfl <;> rfl

theorem row_source (index : Fin 3) : rowExpression index = HeapMemoCalls.pooledTuple (identity (rowExpression index))
    [outcomeExpression index, .constant (.int (weight index))] := by
  rcases index with ⟨index, inside⟩
  have cases : index = 0 ∨ index = 1 ∨ index = 2 := by omega
  rcases cases with rfl | rfl | rfl <;> rfl

theorem known_jail : catalog (identity jailKey) = some (pair 10 true) := rfl
theorem known_community_go : catalog (identity communityGo) = some (pair 0 false) := rfl
theorem known_community_jail : catalog (identity communityJail) = some (pair 10 true) := rfl
theorem known_chance_jail : catalog (identity chanceJail) = some (pair 10 true) := rfl
theorem known_back : catalog (identity backThree) = some (List.ofFn row) := rfl

theorem known_outcome (index : Fin 3) : catalog (identity (outcomeExpression index)) = some (pair (square index) (jailed index)) := by
  rcases index with ⟨index, inside⟩
  have cases : index = 0 ∨ index = 1 ∨ index = 2 := by omega
  rcases cases with rfl | rfl | rfl <;> rfl

theorem known_row (index : Fin 3) : catalog (identity (rowExpression index)) =
    some [.tuple (pair (square index) (jailed index)), .scalar (.int (weight index))] := by
  rcases index with ⟨index, inside⟩
  have cases : index = 0 ∨ index = 1 ∨ index = 2 := by omega
  rcases cases with rfl | rfl | rfl <;> rfl

theorem pair_result (expression : Expr) (square : Int) (jailed : Bool)
    (source : expression = HeapMemoCalls.pooledTuple (identity expression) [.constant (.int square), .constant (.bool jailed)])
    (known : catalog (identity expression) = some (pair square jailed))
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel program spec env expression heap (TuplePool.Post heap (.tuple (pair square jailed))) globals := by
  rw [source]
  exact TuplePool.pooled catalog (identity expression) (pair square jailed) known program env _ heap globals
    (TuplePool.scalars catalog program env [.int square, .bool jailed] (by simp) heap globals)

theorem row_result (index : Fin 3) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel program spec env (rowExpression index) heap (TuplePool.Post heap (row index)) globals := by
  rw [row_source]
  apply TuplePool.pooled catalog _ _ (known_row index)
  apply TuplePool.tuple catalog _ _ _ _ (by simp)
  apply TuplePool.arguments_cons catalog _ _ _ _ _ _
    _ _ (pair_result (outcomeExpression index) (square index) (jailed index) (outcome_source index) (known_outcome index) env heap globals)
  intro middle
  exact TuplePool.scalar_arguments catalog program env [.int (weight index)] middle globals

/-- The nested back-three table can be allocated or reused without changing any caller object. -/
theorem back_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel program spec env backThree heap (TuplePool.Post heap (.tuple (List.ofFn row))) globals := by
  rw [back_source]
  apply TuplePool.pooled catalog _ _ known_back
  apply TuplePool.tuple catalog _ _ _ _ (by simp)
  change HeapMemoCalls.ArgumentsRel program spec env [rowExpression 0, rowExpression 1, rowExpression 2] heap
    (TuplePool.ArgumentsPost heap [row 0, row 1, row 2]) globals
  apply TuplePool.arguments_cons catalog _ _ _ _ _ _ _ _ (row_result 0 env heap globals)
  intro first
  apply TuplePool.arguments_cons catalog _ _ _ _ _ _ _ _ (row_result 1 env first globals)
  intro second
  apply TuplePool.arguments_cons catalog _ _ _ _ _ _ _ _ (row_result 2 env second globals)
  intro third
  exact TuplePool.arguments_nil catalog program env third globals

end solvers.«84».resolve_landing.pool
