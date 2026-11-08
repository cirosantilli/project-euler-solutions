import solvers.«84.py.lean.d».TransitionSource
import solvers.«84.py.lean.d».Encoding

namespace solvers.«84».transition_row
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

def landingBody : List Stmt := match landingCode[(0 : Nat)]? with
  | some (.forDictionaryItems _ _ body _) => body
  | _ => []

def nextExpression : Expr := .ifExp (.boolOp .or [.name "sent", .unaryOp .not (.name "is_double")])
  (.constant (.int 0)) (.binOp (.name "doubles") .add (.constant (.int 1)))

def nextDoubles (sent isDouble : Bool) (doubles : Nat) : Nat := if sent then 0 else if isDouble then doubles + 1 else 0

theorem landing_body_source : landingBody = [.assign [.name "next_doubles"] nextExpression,
    .assign [.name "target"] (.call (.name "encode_state") [.name "destination", .name "next_doubles"]),
    increment (.binOp (.name "weight") .mult (.name "count"))] := rfl

theorem next_doubles_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (sent isDouble : Bool) (doubles : Nat)
    (getSent : env "sent" = some (.scalar (.bool sent))) (getDouble : env "is_double" = some (.scalar (.bool isDouble)))
    (getDoubles : env "doubles" = some (number doubles)) :
    Evaluates program env nextExpression heap (number (nextDoubles sent isDouble doubles)) heap globals := by
  have tested : Evaluates program env (.boolOp .or [.name "sent", .unaryOp .not (.name "is_double")]) heap
      (.scalar (.bool (sent || !isDouble))) heap globals := by
    cases sent with
    | true => exact Evaluates.boolean_stop (Evaluates.lookup getSent) rfl rfl
    | false =>
        apply Evaluates.boolean_next (Evaluates.lookup getSent) rfl rfl
        apply Evaluates.boolean_last (flag := !isDouble) _ rfl
        apply Evaluates.unary (Evaluates.lookup getDouble)
        cases isDouble <;> rfl
  apply Evaluates.conditional (flag := sent || !isDouble) tested rfl
  cases sent <;> cases isDouble <;> simp only [Bool.false_or, Bool.true_or, Bool.not_false, Bool.not_true, Bool.false_eq_true,
    ite_false, ite_true, nextDoubles]
  · exact Evaluates.constant
  · apply Evaluates.binary (Evaluates.lookup getDoubles) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.constant
  · exact Evaluates.constant

def landingLocals (env : HeapCalls.Env) (destination doubles : Nat) (sent isDouble : Bool) : HeapCalls.Env :=
  setLocal (setLocal env "next_doubles" (number (nextDoubles sent isDouble doubles))) "target"
    (number (Monopoly.encode destination (nextDoubles sent isDouble doubles)))

theorem landing_body_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root destination doubles weight count : Nat) (sent isDouble : Bool) (entries : Entries)
    (found : Found heap root entries) (getRow : env "row" = some (.ref root))
    (getDestination : env "destination" = some (number destination)) (getDoubles : env "doubles" = some (number doubles))
    (getWeight : env "weight" = some (number weight)) (getCount : env "count" = some (number count))
    (getSent : env "sent" = some (.scalar (.bool sent))) (getDouble : env "is_double" = some (.scalar (.bool isDouble))) :
    BlockContinues program env landingBody heap (landingLocals env destination doubles sent isDouble)
      (heap.set root (.dict (encode (add entries (Monopoly.encode destination (nextDoubles sent isDouble doubles)) (weight * count))))) globals := by
  have next := next_doubles_result env heap globals sent isDouble doubles getSent getDouble getDoubles
  let middle := setLocal env "next_doubles" (number (nextDoubles sent isDouble doubles))
  have target : Evaluates program middle (.call (.name "encode_state") [.name "destination", .name "next_doubles"]) heap
      (number (Monopoly.encode destination (nextDoubles sent isDouble doubles))) heap globals := by
    apply Evaluates.call (values := [number destination, number (nextDoubles sent isDouble doubles)])
      (.cons (Evaluates.lookup (by simpa [middle, setLocal] using getDestination))
      (.cons (Evaluates.lookup (by simp [middle, setLocal])) .nil))
    exact encode_state.natural destination (nextDoubles sent isDouble doubles) heap globals
  have amount : Evaluates program (landingLocals env destination doubles sent isDouble)
      (.binOp (.name "weight") .mult (.name "count")) heap (number (weight * count)) heap globals := by
    apply Evaluates.binary (Evaluates.lookup (by simpa [landingLocals, setLocal] using getWeight))
      (Evaluates.lookup (by simpa [landingLocals, setLocal] using getCount))
    simp [Mutable.binary, scalarBinary, number]
  have updated := increment_result (landingLocals env destination doubles sent isDouble) heap globals root
    (Monopoly.encode destination (nextDoubles sent isDouble doubles)) (weight * count) entries _ found
    (by simpa [landingLocals, setLocal] using getRow) (by simp [landingLocals, setLocal]) amount
  rw [landing_body_source]
  exact (BlockContinues.assign next).append ((BlockContinues.assign target).append updated)

theorem ordinary_branch (env finalEnv : HeapCalls.Env) (heap finalHeap : Mutable.Heap) (globals : Mutable.Globals)
    (isDouble : Bool) (doubles : Nat) (ordinary : ¬(isDouble = true ∧ doubles = 2))
    (getDouble : env "is_double" = some (.scalar (.bool isDouble))) (getDoubles : env "doubles" = some (number doubles))
    (executed : BlockContinues program env landingCode heap finalEnv finalHeap globals) :
    BlockContinues program env code heap finalEnv finalHeap globals := by
  have tested : Evaluates program env
      (.boolOp .and [.name "is_double", .compare (.name "doubles") [(.eq, .constant (.int 2))]]) heap
      (.scalar (.bool false)) heap globals := by
    cases isDouble with
    | false => exact Evaluates.boolean_stop (Evaluates.lookup getDouble) rfl rfl
    | true =>
        have different : doubles ≠ 2 := fun same => ordinary ⟨rfl, same⟩
        apply Evaluates.boolean_next (Evaluates.lookup getDouble) rfl rfl
        apply Evaluates.boolean_last (flag := false) _ rfl
        apply Evaluates.compare (Evaluates.lookup getDoubles) Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, show (doubles : Int) ≠ 2 by omega]
  rw [code_source]
  exact BlockContinues.branch (flag := false) tested rfl executed

theorem next_doubles_bound (sent isDouble : Bool) (doubles : Nat) (bound : doubles < 3)
    (ordinary : ¬(isDouble = true ∧ doubles = 2)) : nextDoubles sent isDouble doubles < 3 := by
  cases sent <;> cases isDouble <;> simp_all [nextDoubles]
  omega

end solvers.«84».transition_row
