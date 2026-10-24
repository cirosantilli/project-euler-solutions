import «.formalsw».solvers.«34.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.DigitFactorialBase
import ProjectEulerSolutions.Lib.HeapCollection

/-! An incrementally constructed factorial table for an arbitrary supported base. -/
namespace solvers.«34»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.DigitFactorialBase
set_option Elab.async false
private abbrev program := Solvers.P34Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
def factorialTable (base : Nat) : List Mutable.Value := (facts base).map number
private def tableHeap (caller : Mutable.Heap) (next : Nat) : Mutable.Heap := caller ++ [.list (factorialTable next)]
private def fillBody : List Stmt :=
  [.augAssign (.name "f") .mult (.name "digit"), .expr (.call (.attribute (.name "facts") "append") [.name "f"])]

private theorem table_append (next : Nat) :
    factorialTable (next + 1) = factorialTable next ++ [number (Nat.factorial next)] := by
  simp only [factorialTable, facts_append, List.map_append, List.map_cons, List.map_nil]

private theorem fill_step (next : Nat) (positive : 1 ≤ next) (caller : Mutable.Heap) (env : HeapCalls.Env)
    (getFacts : env "facts" = some (.ref caller.length)) (getDigit : env "digit" = some (number next))
    (getFactorial : env "f" = some (number (Nat.factorial (next - 1)))) :
    BlockContinues program env fillBody (tableHeap caller next)
      (setLocal env "f" (number (Nat.factorial next))) (tableHeap caller (next + 1)) := by
  have product : Nat.factorial (next - 1) * next = Nat.factorial next := by
    cases next with
    | zero => omega
    | succ next => simp [Nat.factorial_succ, Nat.mul_comm]
  let ready := setLocal env "f" (number (Nat.factorial next))
  have incremented : BlockContinues program env [fillBody[0]'(by decide)] (tableHeap caller next) ready (tableHeap caller next) :=
    BlockContinues.augAssign (left := number (Nat.factorial (next - 1))) (right := number next)
      getFactorial (Evaluates.lookup getDigit) (by simp [inplace, Mutable.binary, scalarBinary, number, ← product])
  have appended : BlockContinues program ready [fillBody[1]'(by decide)] (tableHeap caller next) ready (tableHeap caller (next + 1)) := by
    apply BlockContinues.discard
    have executed := Evaluates.append (globals := []) (module := program) (env := ready) (old := factorialTable next)
      (current := factorialTable next) (value := number (Nat.factorial next))
      (base := .name "facts") (expression := .name "f")
      (Evaluates.lookup (heap := tableHeap caller next) (by simpa [ready, setLocal] using getFacts))
      (by simp [tableHeap]) (Evaluates.lookup (by simp [ready, setLocal])) (by simp [tableHeap])
    simpa [tableHeap, table_append] using executed
  exact incremented.append appended

private theorem fill_loop (base next count : Nat) (positive : 1 ≤ next) (length : next + count = base)
    (caller : Mutable.Heap) (env : HeapCalls.Env) (getFacts : env "facts" = some (.ref caller.length))
    (getFactorial : env "f" = some (number (Nat.factorial (next - 1)))) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "digit" next base 1 fillBody []]
      (tableHeap caller next) finalEnv (tableHeap caller base) ∧ finalEnv "facts" = some (.ref caller.length) := by
  induction count generalizing next env with
  | zero =>
      have same : next = base := by omega
      subst next
      exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, getFacts⟩
  | succ count ih =>
      let bound := setLocal env "digit" (number next)
      let ready := setLocal bound "f" (number (Nat.factorial next))
      have filled := fill_step next positive caller bound (by simpa [bound, setLocal] using getFacts)
        (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getFactorial)
      obtain ⟨finalEnv, rest, finalFacts⟩ := ih (next + 1) (by omega) (by omega) ready
        (by simpa [ready, bound, setLocal] using getFacts) (by simp [ready, setLocal])
      refine ⟨finalEnv, ?_, finalFacts⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, show Calls.loopControl fillBody = false from rfl, Bool.false_eq_true, if_false,
        Int.natCast_add, Int.natCast_one] using filled.append rest

/-- Repeated append retains the freshly allocated table's identity and fills all factorial entries. -/
theorem precompute_factorials.result (base : Nat) (valid : 2 ≤ base) (caller : Mutable.Heap) :
    ReturnsValue program "precompute_factorials" [number base] caller (.ref caller.length) (tableHeap caller base) := by
  let initial := functionEnv ((findFunctionIn program.body "precompute_factorials").getD ([], [])).1
    ((findFunctionIn program.body "precompute_factorials").getD ([], [])).2 [number base] []
  let ready := setLocal (setLocal initial "facts" (.ref caller.length)) "f" (number 1)
  obtain ⟨finalEnv, filled, finalFacts⟩ := fill_loop base 1 (base - 1) (by decide) (by omega) caller ready
    (by simp [ready, setLocal]) (by simp [ready, setLocal])
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref caller.length) (middle := tableHeap caller 1)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · apply BlockReturns.assign Evaluates.constant
    have scanned : BlockContinues program ready
        [.forIn (.name "digit") (.call (.name "range") [.constant (.int 1), .name "base"]) fillBody []]
        (tableHeap caller 1) finalEnv (tableHeap caller base) :=
      BlockContinues.for_range (.cons Evaluates.constant
        (.cons (Evaluates.lookup (by simp [ready, setLocal, initial]; rfl)) .nil)) rfl filled
    exact scanned.returns (BlockReturns.return_value (Evaluates.lookup finalFacts))

end solvers.«34»
