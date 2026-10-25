import «.formalsw».solvers.«56.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p56.py»
import FormalSW
import ProjectEulerSolutions.Lib.PowerDigitSum

namespace solvers.«56»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
set_option Elab.async false
private abbrev program := Solvers.P56Ast.module

/-- The unchanged ord-minus-48 generator sums the actual decimal string characters, with no heap effects. -/
theorem digit_sum.result (n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "digit_sum" [.scalar (.int n)] heap
      (.scalar (.int (PowerDigitSum.digitSum n))) heap globals := by
  let characters := (toString (n : Int)).toList
  have total : characters.attach.foldl
      (fun (total : Int) character => total + ((character.val.toNat : Int) - 48)) 0 =
      (PowerDigitSum.digitSum n : Int) := by
    rw [List.foldl_attach (f := fun (total : Int) (character : Char) => total + ((character.toNat : Int) - 48))]
    exact PowerDigitSum.printed_digit_sum n
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  rw [← total]
  apply Evaluates.sum_generator_sequence
    (items := characters.map (fun character => .scalar (.string (String.singleton character)))) (by intros; simp)
    (Evaluates.builtin (name := .str) (values := [.scalar (.int n)])
      (value := .scalar (.string (toString (n : Int))))
      (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil) rfl)
    (by simp [Mutable.sequence, characters]) Evaluates.constant rfl
  apply sum_map_loop (values := characters.attach)
    (encode := fun character => .scalar (.string (String.singleton character.val)))
    (number := fun character => (character.val.toNat : Int) - 48) characters.attach (by decide)
  · simp [sequenceValue, sequenceSource, Mutable.sequence, characters]
  · intro env character lookup
    apply Evaluates.binary
    · apply Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil)
      change HeapCalls.scalarResult (some (Strings.ord (.string (String.ofList [character.val])))) heap = _
      rw [Strings.ord_character]
      rfl
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary]
  · rfl
  · simp [setLocal]

end solvers.«56»


namespace solvers.«56».max_digital_sum
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
set_option Elab.async false

private abbrev program := Solvers.P56Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev scoreExpression : Expr := .call (.name "digit_sum") [.builtinCall .pow [.name "a", .name "b"]]
private abbrev comparison : Expr := .compare (.name "s") [(.gt, .name "best")]
private abbrev updates : List Stmt :=
  [.assign [.name "best"] (.name "s"), .assign [.name "best_a"] (.name "a"), .assign [.name "best_b"] (.name "b")]
private abbrev cellBody : List Stmt := [.assign [.name "s"] scoreExpression, .ifThen comparison updates []]
private abbrev rowBody : List Stmt := [.forIn (.name "b") (.call (.name "range") [.name "limit"]) cellBody []]

/-- Witnesses stay actual integer locals even when the range is empty. -/
private def Best (env : HeapCalls.Env) (best : Int) : Prop :=
  ∃ a b : Int, env "best" = some (number best) ∧ env "best_a" = some (number a) ∧ env "best_b" = some (number b)

private theorem Best.set_other {env : HeapCalls.Env} {best : Int} (known : Best env best)
    (name : String) (value : Mutable.Value) (noBest : "best" ≠ name)
    (noA : "best_a" ≠ name) (noB : "best_b" ≠ name) : Best (setLocal env name value) best := by
  obtain ⟨a, b, getBest, getA, getB⟩ := known
  exact ⟨a, b, by simpa [setLocal, noBest] using getBest,
    by simpa [setLocal, noA] using getA, by simpa [setLocal, noB] using getB⟩

/-- Each source cell evaluates pow and digit_sum before applying its original strict witness update. -/
private theorem cell (a b : Nat) (best : Int) (heap : Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getA : env "a" = some (number a)) (getB : env "b" = some (number b)) (known : Best env best) :
    ∃ finalEnv, BlockContinues program env cellBody heap finalEnv heap globals ∧
      Best finalEnv (max best (PowerDigitSum.score a b)) ∧
      finalEnv "a" = env "a" ∧ finalEnv "limit" = env "limit" := by
  have power : Evaluates program env (.builtinCall .pow [.name "a", .name "b"]) heap (number (a^b)) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getA) (.cons (Evaluates.lookup getB) .nil))
    simp [primitive, scalarResult, scalarBinary, number, show ¬ (b : Int) < 0 by omega]
  have scored : Evaluates program env scoreExpression heap (number (PowerDigitSum.score a b)) heap globals :=
    Evaluates.call (.cons power .nil) (digit_sum.result (a^b) heap globals)
  let ready := setLocal env "s" (number (PowerDigitSum.score a b))
  obtain ⟨previousA, previousB, getBest, getBestA, getBestB⟩ := known
  have readyBest : ready "best" = some (number best) := by simpa [ready, setLocal] using getBest
  have tested : Evaluates program ready comparison heap (flag (decide (best < PowerDigitSum.score a b))) heap globals := by
    apply Evaluates.compare
      (Evaluates.lookup (value := number (PowerDigitSum.score a b)) (by simp [ready, setLocal]))
      (Evaluates.lookup readyBest)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
  have assigned : BlockContinues program env [.assign [.name "s"] scoreExpression] heap ready heap globals :=
    BlockContinues.assign scored
  by_cases improved : best < PowerDigitSum.score a b
  · let afterBest := setLocal ready "best" (number (PowerDigitSum.score a b))
    let afterA := setLocal afterBest "best_a" (number a)
    let final := setLocal afterA "best_b" (number b)
    have bestSet : BlockContinues program ready [.assign [.name "best"] (.name "s")] heap afterBest heap globals :=
      BlockContinues.assign (Evaluates.lookup (by simp [ready, setLocal]))
    have aSet : BlockContinues program afterBest [.assign [.name "best_a"] (.name "a")] heap afterA heap globals :=
      BlockContinues.assign (Evaluates.lookup (by simpa [afterBest, ready, setLocal] using getA))
    have bSet : BlockContinues program afterA [.assign [.name "best_b"] (.name "b")] heap final heap globals :=
      BlockContinues.assign (Evaluates.lookup (by simpa [afterA, afterBest, ready, setLocal] using getB))
    have changed : BlockContinues program ready updates heap final heap globals := bestSet.append (aSet.append bSet)
    have branched : BlockContinues program ready [.ifThen comparison updates []] heap final heap globals := by
      apply BlockContinues.branch (flag := true) (by simpa [improved] using tested) rfl
      exact changed
    refine ⟨final, assigned.append branched, ?_, ?_, ?_⟩
    · rw [max_eq_right (le_of_lt improved)]
      exact ⟨a, b, by simp [final, afterA, afterBest, setLocal],
        by simp [final, afterA, setLocal], by simp [final, setLocal]⟩
    · simp [final, afterA, afterBest, ready, setLocal]
    · simp [final, afterA, afterBest, ready, setLocal]
  · have branched : BlockContinues program ready [.ifThen comparison updates []] heap ready heap globals := by
      apply BlockContinues.branch (flag := false) (by simpa [improved] using tested) rfl
      exact BlockContinues.nil
    refine ⟨ready, assigned.append branched, ?_, ?_, ?_⟩
    · rw [max_eq_left (by omega)]
      exact ⟨previousA, previousB, readyBest,
        by simpa [ready, setLocal] using getBestA, by simpa [ready, setLocal] using getBestB⟩
    · simp [ready, setLocal]
    · simp [ready, setLocal]

private theorem row_loop (length first a : Nat) (stop : Int) (equation : stop = ((first+length : Nat) : Int))
    (best : Int) (heap : Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getA : env "a" = some (number a)) (known : Best env best) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "b" first stop 1 cellBody []] heap finalEnv heap globals ∧
      Best finalEnv (PowerDigitSum.row a (List.range' first length) best) ∧
      finalEnv "a" = some (number a) ∧ finalEnv "limit" = env "limit" := by
  induction length generalizing first best env with
  | zero =>
      exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil,
        by simpa [PowerDigitSum.row] using known, getA, rfl⟩
  | succ length ih =>
      let bound := setLocal env "b" (number first)
      have boundA : bound "a" = some (number a) := by simpa [bound, setLocal] using getA
      have boundBest : Best bound best := known.set_other "b" (number first) (by decide) (by decide) (by decide)
      obtain ⟨afterCell, ranCell, cellBest, cellA, cellLimit⟩ := cell a first best heap bound globals boundA rfl boundBest
      have afterA : afterCell "a" = some (number a) := cellA.trans boundA
      obtain ⟨finalEnv, rest, finalBest, finalA, finalLimit⟩ :=
        ih (first+1) (by omega) (max best (PowerDigitSum.score a first)) afterCell afterA cellBest
      refine ⟨finalEnv, ?_, ?_, finalA, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, show Calls.loopControl cellBody = false from rfl,
          Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using ranCell.append rest
      · simpa only [PowerDigitSum.row, List.range'_succ, List.foldl_cons] using finalBest
      · rw [finalLimit, cellLimit]
        simp [bound, setLocal]

private theorem outer_loop (length first width : Nat) (stop : Int)
    (equation : stop = ((first+length : Nat) : Int)) (widthEq : stop = (width : Int))
    (best : Int) (heap : Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getLimit : env "limit" = some (number stop)) (known : Best env best) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "a" first stop 1 rowBody []] heap finalEnv heap globals ∧
      Best finalEnv (PowerDigitSum.scan (List.range width) (List.range' first length) best) ∧
      finalEnv "limit" = some (number stop) := by
  induction length generalizing first best env with
  | zero =>
      exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil,
        by simpa [PowerDigitSum.scan] using known, getLimit⟩
  | succ length ih =>
      let bound := setLocal env "a" (number first)
      have boundLimit : bound "limit" = some (number stop) := by simpa [bound, setLocal] using getLimit
      have boundBest : Best bound best := known.set_other "a" (number first) (by decide) (by decide) (by decide)
      obtain ⟨afterRow, inner, rowBest, rowA, rowLimit⟩ :=
        row_loop width 0 first stop (by omega) best heap bound globals rfl boundBest
      have rowKnown : Best afterRow (PowerDigitSum.row first (List.range width) best) := by
        simpa only [List.range_eq_range'] using rowBest
      have ranRow : BlockContinues program bound rowBody heap afterRow heap globals :=
        BlockContinues.for_range (.cons (Evaluates.lookup boundLimit) .nil) rfl inner
      obtain ⟨finalEnv, rest, finalBest, finalLimit⟩ := ih (first+1) (by omega)
        (PowerDigitSum.row first (List.range width) best) afterRow (rowLimit.trans boundLimit) rowKnown
      refine ⟨finalEnv, ?_, ?_, finalLimit⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, show Calls.loopControl rowBody = false from rfl,
          Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using ranRow.append rest
      · simpa only [PowerDigitSum.scan, List.range'_succ, List.foldl_cons] using finalBest

/-- The actual nested source loops compute the maximum and retain their actual strict-update witness locals. -/
theorem result (limit : Int) (heap : Heap) (globals : Mutable.Globals := []) :
    ∃ bestA bestB : Int, ReturnsValue program "max_digital_sum" [number limit] heap (.ref heap.length)
      (heap ++ [.tuple [number (PowerDigitSum.maximum limit.toNat), number bestA, number bestB]]) globals := by
  let args := ((findFunctionIn program.body "max_digital_sum").getD ([], [])).1
  let body := ((findFunctionIn program.body "max_digital_sum").getD ([], [])).2
  let initial := functionEnv args body [number limit] globals
  let e1 := setLocal initial "best" (number 0)
  let e2 := setLocal e1 "best_a" (number (-1))
  let ready := setLocal e2 "best_b" (number (-1))
  have getLimit : ready "limit" = some (number limit) := by cases globals <;> rfl
  have known : Best ready 0 := ⟨-1, -1, rfl, rfl, rfl⟩
  have loopResult : ∃ finalEnv, BlockContinues program ready
      [.rangeStepLoop "a" 0 limit 1 rowBody []] heap finalEnv heap globals ∧
      Best finalEnv (PowerDigitSum.maximum limit.toNat) := by
    by_cases positive : 0 < limit
    · obtain ⟨finalEnv, ran, best, _⟩ := outer_loop limit.toNat 0 limit.toNat limit (by omega) (by omega) 0 heap ready globals getLimit known
      exact ⟨finalEnv, ran, by simpa only [PowerDigitSum.maximum, List.range_eq_range'] using best⟩
    · have empty : limit.toNat = 0 := by omega
      exact ⟨ready, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil,
        by simpa [PowerDigitSum.maximum, PowerDigitSum.scan, empty] using known⟩
  obtain ⟨finalEnv, scanned, bestA, bestB, getBest, getA, getB⟩ := loopResult
  refine ⟨bestA, bestB, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign (Evaluates.unary Evaluates.constant rfl)
  apply BlockReturns.assign (Evaluates.unary Evaluates.constant rfl)
  apply BlockReturns.for_range (.cons (Evaluates.lookup getLimit) .nil) rfl
  apply scanned.returns
  apply BlockReturns.return_value
  exact Evaluates.tuple (.cons (Evaluates.lookup getBest) (.cons (Evaluates.lookup getA)
    (.cons (Evaluates.lookup getB) .nil))) (by simp)

end solvers.«56».max_digital_sum

namespace solvers.«56».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev program := Solvers.P56Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

/-- The public source reads the first element of the actual returned witness tuple. -/
theorem result (limit : Int) (heap : Heap) (globals : Mutable.Globals := []) :
    ∃ bestA bestB : Int, ReturnsValue program "solve" [number limit] heap (number (PowerDigitSum.maximum limit.toNat))
      (heap ++ [.tuple [number (PowerDigitSum.maximum limit.toNat), number bestA, number bestB]]) globals := by
  obtain ⟨a, b, called⟩ := max_digital_sum.result limit heap globals
  refine ⟨a, b, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.subscript (by intros; simp)
    (Evaluates.call (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil) called) Evaluates.constant
  simp [Mutable.getItem, Mutable.index, number, Except.map]

end solvers.«56».solve

namespace solvers.«56»
open FormalSW.PyAst

theorem digit_sum.«Returns» (n : Int)
    (condition : Calls.Condition Solvers.P56Ast.module (fun key => if key = "n" then some (.int n) else none)
      Solvers.P56Ast.requires_digit_sum_0) :
    (HeapCalls.scalarFunction Solvers.P56Ast.module "digit_sum" [.int n]).Returns := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
  have returned : (HeapCalls.scalarFunction Solvers.P56Ast.module "digit_sum" [.int (n.toNat : Int)]).Returns :=
    (digit_sum.result n.toNat []).scalar_returns
  simpa only [Int.toNat_of_nonneg nonnegative] using returned

theorem digit_sum.«Supported» (n : Int)
    (condition : Calls.Condition Solvers.P56Ast.module (fun key => if key = "n" then some (.int n) else none)
      Solvers.P56Ast.requires_digit_sum_0) :
    (HeapCalls.scalarFunction Solvers.P56Ast.module "digit_sum" [.int n]).Supported :=
  (digit_sum.«Returns» n condition).supported

end solvers.«56»

namespace solvers.«56».solve
open FormalSW.PyAst

theorem «Returns» (limit : Int) :
    (HeapCalls.scalarFunction Solvers.P56Ast.module "solve" [.int limit]).Returns := by
  obtain ⟨a, b, returned⟩ := result limit []
  exact returned.scalar_returns

end solvers.«56».solve

namespace solvers.«56».solve
open FormalSW.PyAst

/-- The actual Python programs have the same scalar outcome for every signed limit. -/
theorem «Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction Solvers.P56Ast.module "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P56Ast.module "naive" [.int limit]) := by
  obtain ⟨a, b, optimized⟩ := result limit []
  obtain ⟨extra, original⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p56.naive.result limit []
  have optimizedRun := optimized.scalar_finishes (values := [.int limit])
  have originalRun := original.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimizedRun]
    exact originalRun
  · intro finished
    rw [finished.unique originalRun]
    exact optimizedRun

theorem «Supported» :
    (∀ limit : Int, (HeapCalls.scalarFunction Solvers.P56Ast.module "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (HeapCalls.scalarFunction P56Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p56.naive.«Returns» limit).supported

end solvers.«56».solve
