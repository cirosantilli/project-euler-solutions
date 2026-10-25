import «.formalsw».solvers.«55.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p55.py»
import FormalSW
import ProjectEulerSolutions.Lib.Lychrel

namespace solvers.«55»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.Lychrel
set_option Elab.async false
private abbrev program := Solvers.P55Ast.module

/-- The real str/slice/int expression computes decimal digit reversal, including zero and trailing decimal zeroes. -/
theorem reverse_int.result (n : Nat) :
    ReturnsValue program "reverse_int" [.int n] (.int (reverse n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.builtin (values := [.string (String.ofList (toString (n : Int)).toList.reverse)])
  · apply Evaluates.Arguments.cons
    · apply Evaluates.slice (stepValue := .int (-1))
      · exact .cons (Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil) rfl)
          (.cons Evaluates.constant (.cons Evaluates.constant
            (.cons (Evaluates.unary Evaluates.constant rfl) .nil)))
      · exact Strings.slice_reverse (toString (n : Int))
    · exact .nil
  · exact integerConversion_reverse n

/-- The source palindrome helper has a direct scalar result for every signed integer. -/
theorem is_palindrome.text_result (n : Int) :
    ReturnsValue program "is_palindrome" [.int n]
      (.bool (decide (toString n = String.ofList (toString n).toList.reverse))) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .string (toString n))
  · exact Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil) rfl
  · apply BlockReturns.return_value
    apply Evaluates.compare (Evaluates.lookup rfl)
    · apply Evaluates.slice (stepValue := .int (-1))
      · exact .cons (Evaluates.lookup rfl) (.cons Evaluates.constant (.cons Evaluates.constant
          (.cons (Evaluates.unary Evaluates.constant rfl) .nil)))
      · exact Strings.slice_reverse (toString n)
    · simp [scalarCompare]

/-- On nonnegative inputs the actual string comparison is the clean model's decimal palindrome predicate. -/
theorem is_palindrome.result (n : Nat) :
    ReturnsValue program "is_palindrome" [.int n] (.bool (palindrome n)) := by
  simpa only [palindrome_string] using is_palindrome.text_result (n : Int)

end solvers.«55»

namespace solvers.«55».is_lychrel_candidate
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions
set_option Elab.async false

private abbrev program := Solvers.P55Ast.module
private abbrev number (value : Int) : Constant := .int value
private abbrev flag (value : Bool) : Constant := .bool value
private abbrev loopBody : List Stmt :=
  [.assign [.name "x"] (.binOp (.name "x") .add (.call (.name "reverse_int") [.name "x"])),
    .ifThen (.call (.name "is_palindrome") [.name "x"]) [.return (.constant (.bool false))] []]

/-- The source loop uses fuel only to count remaining positive reverse-and-add tests. -/
private theorem loop (fuel n : Nat) (next stop : Int) (remaining : stop-next = (fuel : Int))
    (env : ScalarEnv) (getX : env "x" = some (number n)) :
    BlockReturns program env
      [.rangeStepLoop "_" next stop 1 loopBody [], .return (.constant (.bool true))]
      (flag (Lychrel.candidate n fuel)) := by
  induction fuel generalizing n next env with
  | zero =>
      simp only [Lychrel.candidate]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value Evaluates.constant
  | succ fuel ih =>
      let bound := env.set "_" (number next)
      have boundX : bound "x" = some (number n) := by simpa [bound, ScalarEnv.set] using getX
      have added : Evaluates program bound
          (.binOp (.name "x") .add (.call (.name "reverse_int") [.name "x"]))
          (number (Lychrel.step n)) := by
        apply Evaluates.binary (Evaluates.lookup boundX)
          (Evaluates.call (.cons (Evaluates.lookup boundX) .nil) (reverse_int.result n))
        simp [scalarBinary, Lychrel.step, number]
      let advanced := bound.set "x" (number (Lychrel.step n))
      have advancedX : advanced "x" = some (number (Lychrel.step n)) := by simp [advanced, ScalarEnv.set]
      have checked : Evaluates program advanced (.call (.name "is_palindrome") [.name "x"])
          (flag (Lychrel.palindrome (Lychrel.step n))) :=
        Evaluates.call (.cons (Evaluates.lookup advancedX) .nil) (is_palindrome.result (Lychrel.step n))
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturns.assign added
      apply BlockReturns.branch checked
      cases palindrome : Lychrel.palindrome (Lychrel.step n) with
      | false =>
          have continued := ih (Lychrel.step n) (next+1) (by omega) advanced advancedX
          simpa [Lychrel.candidate, palindrome, scalarTruth, loopBody, Calls.loopContinuation,
            Calls.loopControl, Calls.loopControlStmt] using continued
      | true =>
          simp [Lychrel.candidate, palindrome, scalarTruth]
          exact BlockReturns.return_value Evaluates.constant

private abbrev params := ((findFunctionIn program.body "is_lychrel_candidate").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn program.body "is_lychrel_candidate").getD ([], [])).2

private theorem body_result (n : Nat) (maxIters : Int) (env : ScalarEnv)
    (getN : env "n" = some (number n)) (getMax : env "max_iters" = some (number maxIters)) :
    BlockReturns program env (executableBody sourceBody) (flag (Lychrel.candidate n maxIters.toNat)) := by
  apply BlockReturns.assign (Evaluates.lookup getN)
  apply BlockReturns.for_range (.cons (Evaluates.lookup (by simpa [ScalarEnv.set] using getMax)) .nil) rfl
  by_cases positive : 0 < maxIters
  · exact loop maxIters.toNat n 0 maxIters (by omega) (env.set "x" (number n)) (by simp [ScalarEnv.set])
  · have empty : maxIters.toNat = 0 := by omega
    simp only [empty, Lychrel.candidate]
    apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
    exact BlockReturns.return_value Evaluates.constant

/-- Every nonnegative input executes the actual helper for the supplied signed iteration cap. -/
theorem result (n : Nat) (maxIters : Int) :
    ReturnsValue program "is_lychrel_candidate" [number n, number maxIters]
      (flag (Lychrel.candidate n maxIters.toNat)) := by
  apply BlockReturns.function rfl rfl
  exact body_result n maxIters _ rfl rfl

/-- The public one-argument call uses its source default of fifty additions. -/
theorem default_result (n : Nat) :
    ReturnsValue program "is_lychrel_candidate" [number n] (flag (Lychrel.candidate n 50)) := by
  apply BlockReturns.function_defaults (completed := [number n, number 50]) rfl (by simp) rfl
  exact body_result n 50 _ rfl rfl

end solvers.«55».is_lychrel_candidate

namespace solvers.«55».solve
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions
set_option Elab.async false

private abbrev program := Solvers.P55Ast.module
private abbrev number (value : Int) : Constant := .int value
private abbrev candidateCall : Expr := .call (.name "is_lychrel_candidate") [.name "n", .constant (.int 50)]
private def rangeSum (stop : Int) (env : ScalarEnv) : RangeSum :=
  ⟨"n", .constant (.int 1), [candidateCall], stop, 1, env⟩
private def indicator (n : Nat) : Int := if Lychrel.candidate n 50 then 1 else 0

/-- The original lazy generator tests every positive integer exactly once. -/
private theorem sum_loop (length first : Nat) (stop : Int) (equation : stop = ((first+length : Nat) : Int))
    (env : ScalarEnv) (initial : Int) :
    SumReturns program (rangeSum stop env) first initial
      (initial + ((List.range' first length).map indicator).sum) := by
  induction length generalizing first initial with
  | zero =>
      simp only [List.range'_zero, List.map_nil, List.sum_nil, Int.add_zero]
      exact SumReturns.done (by simp [Calls.rangeContinues, rangeSum]; omega)
  | succ length ih =>
      have evaluated : Evaluates program (env.set "n" (number first)) candidateCall
          (.bool (Lychrel.candidate first 50)) :=
        Evaluates.call (.cons (Evaluates.lookup (by simp [ScalarEnv.set])) (.cons Evaluates.constant .nil))
          (is_lychrel_candidate.result first 50)
      have tested : Evaluates program ((rangeSum stop env).env.set (rangeSum stop env).name (number first))
          (rangeSum stop env).filter (.bool (Lychrel.candidate first 50)) := by
        exact Evaluates.boolean_last evaluated
      cases accepted : Lychrel.candidate first 50 with
      | false =>
          have rest := ih (first+1) (by omega) initial
          apply SumReturns.skip (value := .bool false)
            (by simp [Calls.rangeContinues, rangeSum]; omega)
            (by simpa only [accepted] using tested) rfl
          simpa [List.range'_succ, indicator, accepted, Int.add_assoc] using rest
      | true =>
          have rest := ih (first+1) (by omega) (initial+1)
          apply SumReturns.keep (value := .bool true) (item := .int 1) (updated := initial+1)
            (by simp [Calls.rangeContinues, rangeSum]; omega)
            (by simpa only [accepted] using tested) rfl Evaluates.constant rfl
          simpa [List.range'_succ, indicator, accepted, Int.add_assoc] using rest

private abbrev params := ((findFunctionIn program.body "solve").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn program.body "solve").getD ([], [])).2

private theorem body_result (limit : Int) (env : ScalarEnv) (getLimit : env "limit" = some (number limit)) :
    BlockReturns program env (executableBody sourceBody) (number (Lychrel.count limit.toNat 50)) := by
  apply BlockReturns.return_value
  apply Evaluates.sum_range (.cons Evaluates.constant (.cons (Evaluates.lookup getLimit) .nil)) rfl
  by_cases nonempty : 1 < limit
  · have countEq : (Lychrel.count limit.toNat 50 : Int) =
        ((List.range' 1 (limit.toNat-1)).map indicator).sum := Lychrel.count_eq_sum limit.toNat 50
    rw [countEq]
    simpa only [Int.zero_add] using sum_loop (limit.toNat-1) 1 limit (by omega) env 0
  · have short : limit.toNat-1 = 0 := by omega
    simp only [Lychrel.count, short, List.range'_zero, List.filter_nil, List.length_nil]
    exact SumReturns.done (by simp [Calls.rangeContinues]; omega)

/-- The actual public solver counts fifty-step candidates, including empty signed-limit ranges. -/
theorem result (limit : Int) :
    ReturnsValue program "solve" [number limit] (number (Lychrel.count limit.toNat 50)) := by
  apply BlockReturns.function rfl rfl
  exact body_result limit _ rfl

/-- Omitting the limit executes the source default without changing the solver API. -/
theorem default_result : ReturnsValue program "solve" [] (number (Lychrel.count 10000 50)) := by
  apply BlockReturns.function_defaults (completed := [number 10000]) rfl (by decide) rfl
  exact body_result 10000 _ rfl

theorem «Returns» (limit : Int) : (Calls.function program "solve" [number limit]).Returns := (result limit).returns

end solvers.«55».solve


namespace solvers.«55».solve
open FormalSW.PyAst
set_option Elab.async false

/-- The original Python programs reach the same scalar result for every signed limit.
The reference's private digit lists remain internal to its actual heap execution. -/
theorem «Equiv(naive_default)» (limit : Int) :
    (Calls.function Solvers.P55Ast.module "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P55Ast.module "naive_default" [.int limit]) := by
  have optimized := (result limit).finishes
  obtain ⟨extra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p55.naive_default.result limit []
  have original := reference.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

/-- Both original source calls are covered for every signed limit. -/
theorem «Supported» :
    (∀ limit : Int, (Calls.function Solvers.P55Ast.module "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (HeapCalls.scalarFunction P55Ast.module "naive_default" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    obtain ⟨extra, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p55.naive_default.result limit []
    exact (reference.scalar_returns (values := [.int limit])).supported

/-- The root's omitted limit executes its actual source default and agrees with the original reference at that limit. -/
theorem default_equiv :
    (Calls.function Solvers.P55Ast.module "solve" []).Equiv
      (HeapCalls.scalarFunction P55Ast.module "naive_default" [.int 10000]) := by
  have optimized := default_result.finishes
  obtain ⟨extra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p55.naive_default.result 10000 []
  have original := reference.scalar_finishes (values := [.int 10000])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

end solvers.«55».solve
