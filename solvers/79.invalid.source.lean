import solvers.«79.valid.source»
import lang.py.FormalSW.ReprCalls

/-! Invalid attempts stop before unpacking or graph mutation, with the source's exact message. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def invalidMessage : Expr :=
  .binOp (.binOp (.constant (.string "")) .add (.constant (.string "Invalid attempt: ")))
    .add (.builtinCall .repr [.name "s"])

def invalidException (s : String) : ExceptionObject :=
  { typeName := "ValueError"
    args := [.string ("Invalid attempt: " ++ Strings.reprValue (.string s))] }

theorem invalid_source :
    invalidBody = [.raise (some (.call (.name "ValueError") [invalidMessage]))] := rfl

theorem invalid_guard (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (invalid : s.length ≠ 3 ∨ Strings.isDigit s = false) :
    Evaluates program env invalidTest heap (.scalar (.bool true)) heap := by
  have sized : Evaluates program env (.builtinCall .len [.name "s"])
      heap (number s.length) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup getS) .nil) rfl
  by_cases length : s.length = 3
  · have digits : Strings.isDigit s = false := invalid.resolve_left (by simpa using length)
    have compared : Evaluates program env
        (.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))])
        heap (.scalar (.bool false)) heap := by
      apply Evaluates.compare sized Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, length]
    have tested : Evaluates program env (.call (.attribute (.name "s") "isdigit") [])
        heap (.scalar (.bool false)) heap := by
      simpa only [digits] using Evaluates.is_digit (module := program)
        (Evaluates.lookup (heap := heap) getS)
        (by intro name equal; cases equal; simp [getS])
    have negated : Evaluates program env
        (.unaryOp .not (.call (.attribute (.name "s") "isdigit") []))
        heap (.scalar (.bool true)) heap := Evaluates.unary tested rfl
    exact Evaluates.boolean_next compared rfl rfl (Evaluates.boolean_last negated rfl)
  · have compared : Evaluates program env
        (.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))])
        heap (.scalar (.bool true)) heap := by
      apply Evaluates.compare sized Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      omega
    exact Evaluates.boolean_stop compared rfl rfl

theorem invalid_message (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) :
    Evaluates program env invalidMessage heap
      (text ("Invalid attempt: " ++ Strings.reprValue (.string s))) heap := by
  have messageStart : Evaluates program env
      (.binOp (.constant (.string "")) .add (.constant (.string "Invalid attempt: ")))
      heap (text "Invalid attempt: ") heap :=
    Evaluates.binary Evaluates.constant Evaluates.constant rfl
  exact Evaluates.binary messageStart (Evaluates.repr (Evaluates.lookup getS)) rfl

theorem invalid_body (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (following : List Stmt) (anchors : Nat)
    (stack : List HeapCalls.Frame) :
    Finishes (step program anchors) ⟨.block (invalidBody ++ following) env, stack, heap⟩
      (.ok ⟨.error (invalidException s), [], heap, anchors⟩) := by
  rw [invalid_source]
  refine Finishes.next (following :=
    argumentsState (.raiseException "ValueError") env [] [invalidMessage] stack heap) ?_ ?_
  · rfl
  apply (Evaluates.arguments_reach
    (.cons (invalid_message env heap s getS) .nil) (.raiseException "ValueError") []
    anchors stack).finishes
  exact Finishes.done (by
    simp [step, result, run, primitive, done, protectGenerator,
      Calls.generatorException, invalidException, text])

/-- Both guard failures raise before any of this attempt's graph updates. -/
theorem invalid_attempt (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (invalid : s.length ≠ 3 ∨ Strings.isDigit s = false)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    Finishes (step program anchors) ⟨.block (attemptBody ++ following) env, stack, heap⟩
      (.ok ⟨.error (invalidException s), [], heap, anchors⟩) := by
  rw [attempt_parts]
  refine Finishes.next (following := ⟨.expr invalidTest env,
    .branch invalidBody [] (unpackBody ++ updatesBody ++ following) env :: stack, heap⟩) ?_ ?_
  · rfl
  apply (invalid_guard env heap s getS invalid anchors _).finishes
  refine Finishes.next (following :=
    ⟨.block (invalidBody ++ (unpackBody ++ updatesBody ++ following)) env, stack, heap⟩) ?_ ?_
  · simp [step, run, Mutable.truth, scalarTruth]
  exact invalid_body env heap s getS _ anchors stack

end solvers.«79».graph
