import solvers.«21.py.lean.d».CachedRuntime
import lang.py.FormalSW.HeapRuntimeCostListCopies

/-! Word-cost execution of the source's final amicable-number scan. The table
and the independent factorization result identify both cached calls; the scan
itself only tracks an integer accumulator and the original immutable heap. -/
namespace solvers.«21».runtimeCost.Scan
open FormalSW.PyAst HeapCalls.Runtime
open ProjectEulerStatements.P21

abbrev program := Solvers.P21Ast.module
def number (value : Int) : Mutable.Value := .scalar (.int value)

def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

def loopStatement : Stmt :=
  (executableBody definition.2)[2]?.getD (.expr (.constant .none))

def body : List Stmt :=
  match loopStatement with
  | .forIn _ _ statements _ => statements
  | _ => []

def condition : Expr :=
  match (body[1]? : Option Stmt) with
  | some (.ifThen test _ _) => test
  | _ => .constant .none

theorem loopStatement_eq : loopStatement =
    .forIn (.name "a") (.call (.name "range") [.constant (.int 2), .name "limit"])
      body [] := rfl

theorem body_eq : body =
    [.assign [.name "b"] (.call (.name "__formalsw_local_solve_d") [.name "a", .name "divsum"]),
      .ifThen condition [.augAssign (.name "total") .add (.name "a")] []] := rfl

theorem condition_eq : condition = .boolOp .and
    [.compare (.name "b") [(.notEq, .name "a")],
      .compare (.call (.name "__formalsw_local_solve_d") [.name "b", .name "divsum"])
        [(.eq, .name "a")]] := rfl

theorem body_continuation (next : Stmt) : Calls.loopContinuation body next = next := rfl

structure Locals (env : HeapCalls.Env) (limit address : Nat) (total : Int) : Prop where
  bound : env "limit" = some (number limit)
  array : env "divsum" = some (.ref address)
  accumulated : env "total" = some (number total)

/-- The lifted local helper receives its integer input and captured table as
two actual arguments. Argument evaluation and call dispatch add six units. -/
theorem cached_call (name : String) (x limit address : Nat) (square : x ≤ limit * limit)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (input : env name = some (number x))
    (array : env "divsum" = some (.ref address))
    (found : heap[address]? = some (.list (solvers.«21».Sieve.table limit d)))
    (factorCorrect : ∀ (x : Nat) (heap : Mutable.Heap),
      HeapCalls.ReturnsValue program "proper_divisor_sum_factorization" [number x]
        heap (number (d x)) heap) :
    EvaluatesIn .word program env
      (.call (.name "__formalsw_local_solve_d") [.name name, .name "divsum"])
      heap (300 * (limit + 1) + 6) (number (d x)) heap := by
  have arguments : ArgumentsIn .word program env 0 [.name name, .name "divsum"]
      heap 5 [number x, .ref address] heap :=
    ArgumentsIn.cons (EvaluatesIn.lookup input)
      (ArgumentsIn.cons (EvaluatesIn.lookup array) ArgumentsIn.nil)
  have run := Cached.function_run x limit limit address square heap found (factorCorrect x heap)
  exact (EvaluatesIn.call arguments run).mono (by omega)

/-- The second divisor-sum call is evaluated only if the first comparison is
true. Its input is bounded using the shared divisor-sum value envelope. -/
theorem condition_cost (a limit address : Nat) (inside : a < limit)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (input : env "a" = some (number a))
    (partner : env "b" = some (number (d a)))
    (array : env "divsum" = some (.ref address))
    (found : heap[address]? = some (.list (solvers.«21».Sieve.table limit d)))
    (factorCorrect : ∀ (x : Nat) (heap : Mutable.Heap),
      HeapCalls.ReturnsValue program "proper_divisor_sum_factorization" [number x]
        heap (number (d x)) heap) :
    EvaluatesIn .word program env condition heap (300 * (limit + 1) + 22)
      (.scalar (.bool (decide (isAmicable a)))) heap := by
  have first : EvaluatesIn .word program env (.compare (.name "b") [(.notEq, .name "a")])
      heap 7 (.scalar (.bool (decide (d a ≠ a)))) heap := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup partner) (EvaluatesIn.lookup input)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, number]
    · rfl
  rw [condition_eq]
  by_cases different : d a ≠ a
  · have square : d a ≤ limit * limit :=
      (d_le_square a).trans (Nat.mul_le_mul (Nat.le_of_lt inside) (Nat.le_of_lt inside))
    have second : EvaluatesIn .word program env
        (.compare (.call (.name "__formalsw_local_solve_d") [.name "b", .name "divsum"])
          [(.eq, .name "a")]) heap (300 * (limit + 1) + 12)
        (.scalar (.bool (decide (isAmicable a)))) heap := by
      apply EvaluatesIn.compare (work := 2)
        (cached_call "b" (d a) limit address square heap env partner array found factorCorrect)
        (EvaluatesIn.lookup input)
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp, number, isAmicable, different]
      · rfl
    have pair := EvaluatesIn.boolean_pair (op := .and) first
      (flag := true) (by simp [Mutable.truth, scalarTruth, different]) rfl second
      (answerFlag := decide (isAmicable a)) rfl
    exact pair.mono (by omega)
  · have stopped : EvaluatesIn .word program env
        (.boolOp .and [.compare (.name "b") [(.notEq, .name "a")],
          .compare (.call (.name "__formalsw_local_solve_d") [.name "b", .name "divsum"])
            [(.eq, .name "a")]]) heap 9 (.scalar (.bool false)) heap :=
      EvaluatesIn.boolean_stop (by simpa [different] using first) (flag := false) rfl rfl
    simpa only [show decide (isAmicable a) = false by simp [isAmicable, different]] using
      stopped.mono (large := 300 * (limit + 1) + 22) (by omega)

/-- A source candidate pass keeps the table and limit, and changes only its
integer accumulator and temporary partner. No size bound on the accumulator
is needed for the word arithmetic tariff. -/
theorem candidate_cost (a limit address : Nat) (total : Int) (inside : a < limit)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env limit address total)
    (input : env "a" = some (number a))
    (found : heap[address]? = some (.list (solvers.«21».Sieve.table limit d)))
    (factorCorrect : ∀ (x : Nat) (heap : Mutable.Heap),
      HeapCalls.ReturnsValue program "proper_divisor_sum_factorization" [number x]
        heap (number (d x)) heap) :
    ∃ (finalTotal : Int) (finalEnv : HeapCalls.Env),
      Locals finalEnv limit address finalTotal ∧
      BlockContinuesIn .word program env body heap (700 * (limit + 1)) finalEnv heap := by
  have square : a ≤ limit * limit := by nlinarith
  let afterB := HeapCalls.setLocal env "b" (number (d a))
  have computed : BlockContinuesIn .word program env
      [.assign [.name "b"] (.call (.name "__formalsw_local_solve_d") [.name "a", .name "divsum"])]
      heap (300 * (limit + 1) + 8) afterB heap :=
    BlockContinuesIn.assign
      (cached_call "a" a limit address square heap env input locals.array found factorCorrect)
  have nextInput : afterB "a" = some (number a) := by
    simpa [afterB, HeapCalls.setLocal] using input
  have nextLocals : Locals afterB limit address total :=
    ⟨by simpa [afterB, HeapCalls.setLocal] using locals.bound,
      by simpa [afterB, HeapCalls.setLocal] using locals.array,
      by simpa [afterB, HeapCalls.setLocal] using locals.accumulated⟩
  have tested := condition_cost a limit address inside heap afterB nextInput
    (by simp [afterB, HeapCalls.setLocal]) nextLocals.array found factorCorrect
  by_cases accepted : isAmicable a
  · let finalEnv := HeapCalls.setLocal afterB "total" (number (total + a))
    have added : BlockContinuesIn .word program afterB
        [.augAssign (.name "total") .add (.name "a")] heap 7 finalEnv heap := by
      apply BlockContinuesIn.augAssign (work := 1) (left := .int total) (right := .int a)
        nextLocals.accumulated (EvaluatesIn.lookup nextInput)
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
      · rfl
    have branched : BlockContinuesIn .word program afterB
        [.ifThen condition [.augAssign (.name "total") .add (.name "a")] []]
        heap (300 * (limit + 1) + 31) finalEnv heap :=
      BlockContinuesIn.branch (cc := 300 * (limit + 1) + 22) (cb := 7) tested (flag := true)
        (by simp [Mutable.truth, scalarTruth, accepted]) added
    refine ⟨total + a, finalEnv, ?_, ?_⟩
    · exact ⟨by simpa [finalEnv, HeapCalls.setLocal] using nextLocals.bound,
        by simpa [finalEnv, HeapCalls.setLocal] using nextLocals.array,
        by simp [finalEnv, HeapCalls.setLocal]⟩
    · apply BlockContinuesIn.mono (small := (300 * (limit + 1) + 8) + (300 * (limit + 1) + 31))
      · simpa only [body_eq, List.cons_append, List.nil_append] using computed.append branched
      · omega
  · have branched : BlockContinuesIn .word program afterB
        [.ifThen condition [.augAssign (.name "total") .add (.name "a")] []]
        heap (300 * (limit + 1) + 24) afterB heap :=
      BlockContinuesIn.branch (cc := 300 * (limit + 1) + 22) (cb := 0) tested (flag := false)
        (by simp [Mutable.truth, scalarTruth, accepted]) BlockContinuesIn.nil
    refine ⟨total, afterB, nextLocals, ?_⟩
    apply BlockContinuesIn.mono (small := (300 * (limit + 1) + 8) + (300 * (limit + 1) + 24))
    · simpa only [body_eq, List.cons_append, List.nil_append] using computed.append branched
    · omega

/-- The exact remaining range state of the source scan, including each range
transition and the final failed range test. -/
theorem loop_cost (count start limit address : Nat) (total : Int) (stop : start + count = limit)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env limit address total)
    (found : heap[address]? = some (.list (solvers.«21».Sieve.table limit d)))
    (factorCorrect : ∀ (x : Nat) (heap : Mutable.Heap),
      HeapCalls.ReturnsValue program "proper_divisor_sum_factorization" [number x]
        heap (number (d x)) heap) :
    ∃ (finalTotal : Int) (finalEnv : HeapCalls.Env),
      Locals finalEnv limit address finalTotal ∧
      BlockContinuesIn .word program env [.rangeStepLoop "a" start limit 1 body []]
        heap ((700 * (limit + 1) + 4) * count + 4) finalEnv heap := by
  induction count generalizing start total env with
  | zero =>
      refine ⟨total, env, locals, ?_⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | succ count ih =>
      let indexed := HeapCalls.setLocal env "a" (number start)
      have indexedLocals : Locals indexed limit address total :=
        ⟨by simpa [indexed, HeapCalls.setLocal] using locals.bound,
          by simpa [indexed, HeapCalls.setLocal] using locals.array,
          by simpa [indexed, HeapCalls.setLocal] using locals.accumulated⟩
      obtain ⟨middleTotal, middleEnv, middleLocals, passed⟩ :=
        candidate_cost start limit address total (by omega) heap indexed indexedLocals
          (by simp [indexed, HeapCalls.setLocal]) found factorCorrect
      obtain ⟨finalTotal, finalEnv, finalLocals, rest⟩ :=
        ih (start + 1) middleTotal (by omega) middleEnv middleLocals
      refine ⟨finalTotal, finalEnv, finalLocals, ?_⟩
      have remainder : BlockContinuesIn .word program middleEnv
          [.rangeStepLoop "a" ((start : Int) + 1) limit 1 body []]
          heap ((700 * (limit + 1) + 4) * count + 4) finalEnv heap := by
        simpa only [Nat.cast_add, Nat.cast_one] using rest
      have run := BlockContinuesIn.range_step (env := env) (name := "a")
        (next := (start : Int)) (stop := (limit : Int)) (stride := 1)
        (body := body) (otherwise := [])
        (by simp [Calls.rangeContinues]; omega)
        (by simpa only [body_continuation, indexed, number] using passed.append remainder)
      apply run.mono
      simp only [Nat.mul_succ]
      omega

end solvers.«21».runtimeCost.Scan
