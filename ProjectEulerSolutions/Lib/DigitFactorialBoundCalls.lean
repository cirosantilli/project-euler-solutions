import ProjectEulerSolutions.Lib.DigitFactorialBase
import lang.py.FormalSW.HeapCalls

/-! Symbolic execution of the common digit-width bound loop. -/
namespace ProjectEulerSolutions.DigitFactorialBoundCalls
open FormalSW.PyAst FormalSW.PyAst.HeapCalls DigitFactorialBase
set_option Elab.async false
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
def increment : List Stmt := [.augAssign (.name "digits") .add (.constant (.int 1))]

/-- Any source guard with the mathematical width criterion reaches the first failing width. -/
theorem loop (program : Module) (base : Nat) (valid : 2 ≤ base) (condition : Expr) (context : HeapCalls.Env → Prop)
    (guard : ∀ env heap digits, 1 ≤ digits → context env → env "digits" = some (number digits) →
      Evaluates program env condition heap (.scalar (.bool (decide (base ^ (digits - 1) ≤ digits * maxFact base)))) heap)
    (keep : ∀ env digits, context env → context (setLocal env "digits" (number digits)))
    (digits : Nat) (positive : 1 ≤ digits) (bounded : digits ≤ stoppingWidth base valid)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (current : context env) (getDigits : env "digits" = some (number digits)) :
    ∃ finalEnv, BlockContinues program env [.whileDo condition increment []] heap finalEnv heap ∧
      finalEnv "digits" = some (number (stoppingWidth base valid)) ∧ context finalEnv := by
  have evaluated := guard env heap digits positive current getDigits
  by_cases before : digits < stoppingWidth base valid
  · have active := before_stopping base valid digits positive before
    rw [decide_eq_true active] at evaluated
    let advanced := setLocal env "digits" (number (digits + 1))
    obtain ⟨finalEnv, rest, finalDigits, finalContext⟩ := loop program base valid condition context guard keep (digits + 1)
      (by omega) (by omega) advanced heap (keep env (digits + 1) current) (by simp [advanced, setLocal])
    have incremented : BlockContinues program env increment heap advanced heap :=
      BlockContinues.augAssign (left := number digits) (right := number 1) (value := number (digits + 1))
        getDigits Evaluates.constant (by simp [inplace, Mutable.binary, scalarBinary, number])
    refine ⟨finalEnv, ?_, finalDigits, finalContext⟩
    apply BlockContinues.while_loop evaluated rfl
    simpa only [Calls.loopContinuation, show Calls.loopControl increment = false from rfl, Bool.false_eq_true, if_false] using
      incremented.append rest
  · have same : digits = stoppingWidth base valid := by omega
    have stopped : ¬ base ^ (digits - 1) ≤ digits * maxFact base := by
      rw [same]
      exact Nat.not_le_of_lt (stopping_spec base valid).2
    rw [decide_eq_false stopped] at evaluated
    exact ⟨env, BlockContinues.while_loop evaluated rfl .nil, by simpa only [same] using getDigits, current⟩
termination_by stoppingWidth base valid - digits
decreasing_by omega

end ProjectEulerSolutions.DigitFactorialBoundCalls
