import solvers.«34.digits.source»
import ProjectEulerSolutions.Lib.DigitFactorialBoundCalls
import data.«project-euler-statements».data.py.project_euler_statements.«p34.py»

/-! The optimized search executes every candidate in the complete base-dependent interval. -/
namespace solvers.«34»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.DigitFactorialBase
set_option Elab.async false
private abbrev program := Solvers.P34Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private theorem found_append {heap : Mutable.Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Mutable.Heap) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private abbrev scanBody : List Stmt :=
  [.ifThen (.compare (.name "n")
    [(.eq, .call (.name "digit_factorial_sum") [.name "n", .name "facts", .name "base"])])
    [.augAssign (.name "total") .add (.name "n")] []]

/-- Symbolic induction over an arbitrary finite candidate interval. -/
private theorem scan (base : Nat) (valid : 2 ≤ base) (first count stop total address : Nat) (length : first + count = stop)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (factorialTable base)))
    (getFacts : env "facts" = some (.ref address))
    (getTotal : env "total" = some (number total)) (getBase : env "base" = some (number base)) :
    ∃ extra finalEnv,
      BlockContinues program env [.rangeStepLoop "n" first stop 1 scanBody []]
        heap finalEnv (heap ++ extra) ∧
      finalEnv "facts" = some (.ref address) ∧
      finalEnv "total" = some (number (foldValues base (List.range' first count) total)) := by
  induction count generalizing first total heap env with
  | zero =>
      refine ⟨[], env, ?_, getFacts, ?_⟩
      · simpa only [List.append_nil] using
          (BlockContinues.range_done (module := program) (heap := heap)
            (env := env) (by simp [Calls.rangeContinues]; omega) BlockContinues.nil)
      · simpa [foldValues] using getTotal
  | succ count ih =>
      let bound := setLocal env "n" (number first)
      have boundFacts : bound "facts" = some (.ref address) := by simpa [bound, setLocal] using getFacts
      have boundTotal : bound "total" = some (number total) := by simpa [bound, setLocal] using getTotal
      have boundBase : bound "base" = some (number base) := by simpa [bound, setLocal] using getBase
      have boundX : bound "n" = some (number first) := by simp [bound, setLocal]
      obtain ⟨extra, helper⟩ := digit_factorial_sum.result base first address valid heap found
      have preserved := found_append found extra
      have evaluated : Evaluates program bound
          (.compare (.name "n") [(.eq, .call (.name "digit_factorial_sum") [.name "n", .name "facts", .name "base"])])
          heap (.scalar (.bool (decide (digitSum base first = first)))) (heap ++ extra) := by
        apply Evaluates.compare (av := number first) (bv := number (digitSum base first))
          (middle := heap) (afterRight := heap ++ extra) (Evaluates.lookup boundX)
          (Evaluates.call (.cons (Evaluates.lookup boundX) (.cons (Evaluates.lookup boundFacts) (.cons (Evaluates.lookup boundBase) .nil))) helper)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, eq_comm]
      by_cases accepted : digitSum base first = first
      · let updated := setLocal bound "total" (number (total + first))
        obtain ⟨later, finalEnv, rest, finalFacts, finalTotal⟩ := ih (first + 1) (total + first)
          (by omega) (heap ++ extra) updated preserved
          (by simpa [updated, setLocal] using boundFacts) (by simp [updated, setLocal])
          (by simpa [updated, setLocal] using boundBase)
        refine ⟨extra ++ later, finalEnv, ?_, finalFacts, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          have increment : BlockContinues program bound
              [.augAssign (.name "total") .add (.name "n")] (heap ++ extra) updated (heap ++ extra) :=
            BlockContinues.augAssign (left := number total) (right := number first)
              (value := number (total + first)) (middle := heap ++ extra) (finalHeap := heap ++ extra)
              boundTotal (Evaluates.lookup boundX)
              (by simp [inplace, Mutable.binary, scalarBinary, number])
          have iteration : BlockContinues program bound scanBody heap updated (heap ++ extra) :=
            BlockContinues.branch (flag := true) (by simpa [accepted] using evaluated) rfl increment
          simpa only [Calls.loopContinuation, show Calls.loopControl scanBody = false from rfl,
            ↓reduceIte, Int.natCast_add, Int.natCast_one, List.append_assoc] using iteration.append rest
        · simpa [foldValues, List.range'_succ, accepted] using finalTotal
      · obtain ⟨later, finalEnv, rest, finalFacts, finalTotal⟩ := ih (first + 1) total (by omega)
          (heap ++ extra) bound preserved boundFacts boundTotal boundBase
        refine ⟨extra ++ later, finalEnv, ?_, finalFacts, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          have iteration : BlockContinues program bound scanBody heap bound (heap ++ extra) :=
            BlockContinues.branch (by simpa [accepted] using evaluated) rfl .nil
          simpa only [Calls.loopContinuation, show Calls.loopControl scanBody = false from rfl,
            ↓reduceIte, Int.natCast_add, Int.natCast_one, List.append_assoc] using iteration.append rest
        · simpa [foldValues, List.range'_succ, accepted] using finalTotal

private def boundCondition : Expr := .compare (.binOp (.name "digits") .mult (.name "max_fact"))
  [(.gtE, .binOp (.name "base") .pow (.binOp (.name "digits") .sub (.constant (.int 1))))]
private def boundWhile : Stmt := .whileDo boundCondition ProjectEulerSolutions.DigitFactorialBoundCalls.increment []

private theorem bound_guard (base digits : Nat) (positive : 1 ≤ digits)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getBase : env "base" = some (number base))
    (getDigits : env "digits" = some (number digits)) (getMax : env "max_fact" = some (number (maxFact base))) :
    Evaluates program env boundCondition heap
      (.scalar (.bool (decide (base ^ (digits - 1) ≤ digits * maxFact base)))) heap := by
  apply Evaluates.compare (av := number (digits * maxFact base)) (bv := number (base ^ (digits - 1)))
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (Evaluates.lookup getDigits) (Evaluates.lookup getMax) (by simp [Mutable.binary, scalarBinary, number])
  · apply Evaluates.binary (av := number base) (bv := number (digits - 1))
      (middle := heap) (afterRight := heap) (Evaluates.lookup getBase)
    · apply Evaluates.binary (Evaluates.lookup getDigits) Evaluates.constant
      have cast : ((digits - 1 : Nat) : Int) = (digits : Int) - 1 := by omega
      simp [Mutable.binary, scalarBinary, number, cast]
    · simp [Mutable.binary, scalarBinary, number, show ¬ ((digits - 1 : Nat) : Int) < 0 by omega]
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    exact Int.ofNat_le

/-- The cached table, width loop, and arbitrary finite scan implement the full mathematical solution set. -/
theorem solve.result (base : Nat) (valid : 2 ≤ base) (caller : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number base] caller (number (answer base valid)) finalHeap := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number base] []
  let heap := caller ++ [.list (factorialTable base)]
  let factsEnv := setLocal initial "facts" (.ref caller.length)
  let maxEnv := setLocal factsEnv "max_fact" (number (maxFact base))
  let start := setLocal maxEnv "digits" (number 1)
  let context : HeapCalls.Env → Prop := fun env => env "base" = some (number base) ∧
    env "max_fact" = some (number (maxFact base)) ∧ env "facts" = some (.ref caller.length)
  obtain ⟨boundedEnv, bounded, width, finalContext⟩ := ProjectEulerSolutions.DigitFactorialBoundCalls.loop program base valid
    boundCondition context
    (by intro env heap digits positive current getDigits; exact bound_guard base digits positive env heap current.1 getDigits current.2.1)
    (by intro env digits current; simpa [context, setLocal] using current)
    1 (by decide) (stopping_spec base valid).1 start heap
    (by simp [context, start, maxEnv, factsEnv, setLocal, initial]; rfl) (by simp [start, setLocal])
  let upperEnv := setLocal boundedEnv "upper" (number (upper base valid))
  let ready := setLocal upperEnv "total" (number 0)
  have large := upper_ge_base base valid
  obtain ⟨extra, finalEnv, scanned, finalFacts, finalTotal⟩ := scan base valid base (upper base valid + 1 - base)
    (upper base valid + 1) 0 caller.length (by omega) heap ready (by simp [heap])
    (by simpa [ready, upperEnv, setLocal] using finalContext.2.2) (by simp [ready, setLocal])
    (by simpa [ready, upperEnv, setLocal] using finalContext.1)
  rw [range_answer] at finalTotal
  refine ⟨heap ++ extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (precompute_factorials.result base valid caller))
  apply BlockReturns.assign (value := number (maxFact base)) (middle := heap)
  · apply Evaluates.subscript (value := .ref caller.length) (position := number (base - 1))
      (middle := heap) (afterIndex := heap) (by intro a b c; simp) (Evaluates.lookup rfl)
    · apply Evaluates.binary (av := number base) (bv := number 1) (middle := heap) (afterRight := heap)
        (Evaluates.lookup (by simp [setLocal]; rfl)) Evaluates.constant
      have cast : ((base - 1 : Nat) : Int) = (base : Int) - 1 := by omega
      simp [Mutable.binary, scalarBinary, number, cast]
    · have inside : base - 1 < base := by omega
      have entry : (factorialTable base)[base - 1]? = some (number (maxFact base)) := by
        simp [factorialTable, facts, maxFact, inside]
      simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
        Mutable.getItem_list heap caller.length (base - 1) (factorialTable base) (by simp [heap]) (by simpa [factorialTable] using inside)
  · apply BlockReturns.assign Evaluates.constant
    apply bounded.returns
    apply BlockReturns.assign (value := number (upper base valid)) (middle := heap)
    · apply Evaluates.binary (av := number (stoppingWidth base valid - 1)) (bv := number (maxFact base))
        (middle := heap) (afterRight := heap)
      · apply Evaluates.binary (Evaluates.lookup width) Evaluates.constant
        have cast : ((stoppingWidth base valid - 1 : Nat) : Int) = (stoppingWidth base valid : Int) - 1 := by
          have positive := (stopping_spec base valid).1; omega
        simp [Mutable.binary, scalarBinary, number, cast]
      · exact Evaluates.lookup finalContext.2.1
      · simp [Mutable.binary, scalarBinary, number, upper]
    · apply BlockReturns.assign Evaluates.constant
      have executed : BlockContinues program ready
          [.forIn (.name "n") (.call (.name "range") [.name "base", .binOp (.name "upper") .add (.constant (.int 1))]) scanBody []]
          heap finalEnv (heap ++ extra) := by
        apply BlockContinues.for_range (middle := heap)
          (values := [number base, number (upper base valid + 1)])
        · refine .cons (Evaluates.lookup (by simpa [ready, upperEnv, setLocal] using finalContext.1)) (.cons ?_ .nil)
          exact Evaluates.binary (av := number (upper base valid)) (bv := number 1) (middle := heap) (afterRight := heap)
            (Evaluates.lookup (by simp [ready, upperEnv, setLocal])) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
        · rfl
        · exact scanned
      exact executed.returns (BlockReturns.return_value (Evaluates.lookup finalTotal))

end solvers.«34»
