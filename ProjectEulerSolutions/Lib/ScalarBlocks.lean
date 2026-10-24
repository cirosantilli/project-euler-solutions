import lang.py.FormalSW.Calls
import lang.py.FormalSW.LoopCalls

/-! Continuing scalar blocks compose without choosing the caller's return value or continuation. -/
namespace ProjectEulerSolutions.ScalarBlocks
open FormalSW.PyAst FormalSW.PyAst.Calls

set_option Elab.async false

/-- Executing a scalar block preserves every proof of its subsequent continuation. -/
def Continues (program : Module) (env : ScalarEnv) (body : List Stmt) (finalEnv : ScalarEnv) : Prop :=
  ∀ following value, BlockReturns program finalEnv following value → BlockReturns program env (body ++ following) value

namespace Continues

variable {program : Module} {env middle finalEnv : ScalarEnv}
  {body following otherwise rest yes no : List Stmt} {name : String}
  {expression condition : Expr} {value answer : Constant} {op : BinOp} {flag : Bool}
  {first stop stride : Int} {expressions : List Expr} {values : List Constant} {next : Stmt}

theorem nil : Continues program env [] env := fun _ _ rest => rest

theorem append (first : Continues program env body middle) (second : Continues program middle following finalEnv) :
    Continues program env (body ++ following) finalEnv := by
  intro rest value returned
  simpa only [List.append_assoc] using first (following ++ rest) value (second rest value returned)

theorem returns (first : Continues program env body middle) (rest : BlockReturns program middle following value) :
    BlockReturns program env (body ++ following) value := first following value rest

theorem assign (evaluated : Evaluates program env expression value) :
    Continues program env [.assign [.name name] expression] (env.set name value) :=
  fun _ _ rest => BlockReturns.assign evaluated rest

theorem augAssign (evaluated : Evaluates program env (.binOp (.name name) op expression) value) :
    Continues program env [.augAssign (.name name) op expression] (env.set name value) :=
  fun _ _ rest => BlockReturns.augAssign evaluated rest

theorem branch_next (evaluated : Evaluates program env condition value) (truth : scalarTruth value = flag)
    (executed : Continues program env ((if flag then yes else no) ++ rest) finalEnv) :
    Continues program env (.ifThen condition yes no :: rest) finalEnv := by
  intro following answer returned
  apply BlockReturns.branch evaluated
  simpa only [truth, List.append_assoc] using executed following answer returned

theorem range_step (active : rangeContinues first stop stride = true)
    (executed : Continues program (env.set name (.int first))
      (body ++ [loopContinuation body (.rangeStepLoop name (first + stride) stop stride body otherwise)]) finalEnv) :
    Continues program env [.rangeStepLoop name first stop stride body otherwise] finalEnv := by
  intro following value returned
  apply BlockReturns.range_step active
  simpa only [List.append_assoc] using executed following value returned

theorem range_done (inactive : rangeContinues first stop stride = false)
    (executed : Continues program env otherwise finalEnv) :
    Continues program env [.rangeStepLoop name first stop stride body otherwise] finalEnv :=
  fun following value returned => BlockReturns.range_done inactive (executed following value returned)

theorem for_range (arguments : Evaluates.Arguments program env expressions values)
    (bounds : rangeBounds values = .ok (first, stop, stride))
    (executed : Continues program env [.rangeStepLoop name first stop stride body otherwise] finalEnv) :
    Continues program env [.forIn (.name name) (.call (.name "range") expressions) body otherwise] finalEnv :=
  fun following value returned => BlockReturns.for_range arguments bounds (executed following value returned)

theorem loop_resume (executed : Continues program env [next] finalEnv) :
    Continues program env [.loopResume next] finalEnv :=
  fun following value returned => BlockReturns.loop_resume (executed following value returned)

theorem break_loop (target : loopTarget body = some (next, rest)) (executed : Continues program env rest finalEnv) :
    Continues program env (.breakLoop :: body) finalEnv :=
  fun following value returned => BlockReturns.break_loop (Calls.loopTarget_append target following) (executed following value returned)

end Continues
end ProjectEulerSolutions.ScalarBlocks
