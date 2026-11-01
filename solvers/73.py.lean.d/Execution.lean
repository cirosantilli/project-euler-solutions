import solvers.«73.py.lean.d».Bounds
import ProjectEulerSolutions.P73

/-! The optimized source scans every coprime numerator in its complete trimmed interval. -/
namespace solvers.«73»
open FormalSW.PyAst Calls ProjectEulerStatements.FractionIntervalCount
open data.«project-euler-statements».data.py.project_euler_statements.p73
  (Captured values helperArgs naiveArgs)
set_option Elab.async false

abbrev innerBody : List Stmt :=
  [.ifThen gcdCondition [.augAssign (.name "total") .add (.constant (.int 1))] []]

namespace count_between

theorem loop_result (d first stop : Int) (steps total : Nat) (env : ScalarEnv)
    (length : (stop - first).toNat = steps) (getD : env "d" = some (number d))
    (getTotal : env "total" = some (.int total)) :
    BlockReturns program env
      [.rangeStepLoop "n" first stop 1 innerBody [], .return (.name "total")]
      (.int (total + count (fun n => Int.gcd n d = 1) first stop : Nat)) := by
  induction steps generalizing first total env with
  | zero =>
      rw [count_empty _ first stop (by omega), Nat.add_zero]
      apply BlockReturns.range_done (by simp [rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | succ steps ih =>
      let entered := env.set "n" (number first)
      have enteredD : entered "d" = some (number d) := by simpa [entered, ScalarEnv.set] using getD
      have enteredTotal : entered "total" = some (.int total) := by
        simpa [entered, ScalarEnv.set] using getTotal
      have tested := gcd_result first d entered (by simp [entered, ScalarEnv.set]) enteredD
      apply BlockReturns.range_step (by simp [rangeContinues]; omega)
      by_cases chosen : Int.gcd first d = 1
      · let next := entered.set "total" (.int (total + 1 : Nat))
        have rest := ih (first + 1) (total + 1) next (by omega)
          (by simpa [next, ScalarEnv.set] using enteredD) (by simp [next, ScalarEnv.set])
        apply BlockReturns.branch (value := .bool true)
          (by simpa only [chosen, decide_true] using tested)
        apply BlockReturns.augAssign (value := .int (total + 1 : Nat))
          (Evaluates.binary (Evaluates.lookup enteredTotal) Evaluates.constant
            (by simp [scalarBinary]))
        simpa only [count_step _ first stop (by omega), if_pos chosen, Nat.cast_add,
          Nat.cast_one, Int.add_assoc, loopContinuation,
          show loopControl innerBody = false from rfl, Bool.false_eq_true, if_false] using rest
      · have rest := ih (first + 1) total entered (by omega) enteredD enteredTotal
        apply BlockReturns.branch (value := .bool false)
          (by simpa only [chosen, decide_false] using tested)
        simpa only [count_step _ first stop (by omega), if_neg chosen, Nat.zero_add,
          loopContinuation, show loopControl innerBody = false from rfl,
          Bool.false_eq_true, if_false] using rest

theorem result (d : Int) (p : Bounds) (positive : Positive p) :
    ReturnsValue program "count_between" (values d p) (number (row p d)) := by
  let env := argumentEnv helperArgs (values d p)
  let started := env.set "n_start" (number (lower p d))
  let bounded := started.set "n_end" (number (upper p d))
  have captured : Captured p env := ⟨rfl, rfl, rfl, rfl⟩
  have afterStart := captured.set "n_start" (number (lower p d))
    (by decide) (by decide) (by decide) (by decide)
  have tested : Evaluates program bounded
      (.compare (.name "n_start") [(.gt, .name "n_end")])
      (.bool (decide (upper p d < lower p d))) :=
    Evaluates.compare (av := number (lower p d)) (bv := number (upper p d))
      (Evaluates.lookup (by simp [bounded, started, ScalarEnv.set]))
      (Evaluates.lookup (by simp [bounded, ScalarEnv.set])) rfl
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (lower_result p d env positive captured rfl)
  apply BlockReturns.assign (upper_result p d started positive afterStart
    (by simp [started, ScalarEnv.set, env, helperArgs, values, argumentEnv]))
  by_cases empty : upper p d < lower p d
  · have zero : row p d = 0 := by
      rw [← trimmed_eq_row p positive d]
      exact count_empty _ _ _ (by omega)
    rw [zero]
    apply BlockReturns.branch (value := .bool true)
      (by simpa only [empty, decide_true] using tested)
    exact BlockReturns.return_value Evaluates.constant
  · apply BlockReturns.branch (value := .bool false)
      (by simpa only [empty, decide_false] using tested)
    apply BlockReturns.assign (value := .int 0) Evaluates.constant
    apply BlockReturns.for_range (values := [number (lower p d), number (upper p d + 1)])
      (.cons (Evaluates.lookup (by rfl))
        (.cons (Evaluates.binary (av := number (upper p d)) (bv := number 1)
          (Evaluates.lookup (by rfl)) Evaluates.constant rfl) .nil)) rfl
    have scanned := loop_result d (lower p d) (upper p d + 1)
      (upper p d + 1 - lower p d).toNat 0 (bounded.set "total" (.int 0)) (by omega)
      (by simp [bounded, started, env, helperArgs, values, argumentEnv, ScalarEnv.set])
      (by simp [ScalarEnv.set])
    simpa only [Nat.zero_add, trimmed_eq_row p positive d] using scanned

end count_between

abbrev countExpression : Expr := .call (.name "count_between")
  [.name "d", .name "lower_num", .name "lower_den", .name "upper_num", .name "upper_den"]
abbrev outerBody : List Stmt := [.augAssign (.name "total") .add countExpression]

namespace solve

theorem loop_result (p : Bounds) (positive : Positive p) (stop first : Int)
    (steps total : Nat) (env : ScalarEnv) (captured : Captured p env)
    (getTotal : env "total" = some (.int total)) (length : (stop - first).toNat = steps) :
    BlockReturns program env
      [.rangeStepLoop "d" first stop 1 outerBody [], .return (.name "total")]
      (.int (total + rows p first steps : Nat)) := by
  induction steps generalizing first total env with
  | zero =>
      apply BlockReturns.range_done (by simp [rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | succ steps ih =>
      let entered := env.set "d" (number first)
      let next := entered.set "total" (.int (total + row p first : Nat))
      have enteredCapture := captured.set "d" (number first)
        (by decide) (by decide) (by decide) (by decide)
      have nextCapture := enteredCapture.set "total" (.int (total + row p first : Nat))
        (by decide) (by decide) (by decide) (by decide)
      have rest := ih (first + 1) (total + row p first) next nextCapture
        (by simp [next, ScalarEnv.set]) (by omega)
      apply BlockReturns.range_step (by simp [rangeContinues]; omega)
      apply BlockReturns.augAssign (value := .int (total + row p first : Nat))
      · apply Evaluates.binary
          (Evaluates.lookup (by simpa [ScalarEnv.set] using getTotal))
          (Evaluates.call (.cons (Evaluates.lookup (by simp [ScalarEnv.set]))
            (.cons (Evaluates.lookup enteredCapture.lowerNum)
              (.cons (Evaluates.lookup enteredCapture.lowerDen)
                (.cons (Evaluates.lookup enteredCapture.upperNum)
                  (.cons (Evaluates.lookup enteredCapture.upperDen) .nil)))))
            (count_between.result first p positive))
        simp [scalarBinary]
      · simpa only [rows, Nat.cast_add, Int.add_assoc, loopContinuation,
          show loopControl outerBody = false from rfl, Bool.false_eq_true, if_false] using rest

theorem body_result (limit : Int) (p : Bounds) (positive : Positive p) :
    BlockReturns program (argumentEnv naiveArgs (values limit p))
      (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)
      (number (ProjectEulerStatements.FractionIntervalCount.result p limit)) := by
  apply BlockReturns.assign (value := .int 0) Evaluates.constant
  apply BlockReturns.for_range (values := [.int 2, .int (limit + 1)])
    (.cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup rfl)
      Evaluates.constant rfl) .nil)) rfl
  simpa only [ProjectEulerStatements.FractionIntervalCount.result, Nat.zero_add] using
    loop_result p positive (limit + 1) 2 (limit - 1).toNat 0
      ((argumentEnv naiveArgs (values limit p)).set "total" (.int 0))
      ⟨rfl, rfl, rfl, rfl⟩ rfl (by omega)

theorem result (limit : Int) (p : Bounds) (positive : Positive p) :
    ReturnsValue program "solve" (values limit p)
      (number (ProjectEulerStatements.FractionIntervalCount.result p limit)) :=
  BlockReturns.function rfl rfl (body_result limit p positive)

theorem default_result (limit : Int) (p : Bounds) (positive : Positive p)
    (supplied : List Constant) (omitted : naiveArgs.length ≠ supplied.length)
    (completed : completeArguments naiveArgs supplied = some (values limit p)) :
    ReturnsValue program "solve" supplied
      (number (ProjectEulerStatements.FractionIntervalCount.result p limit)) :=
  ReturnsValue.with_defaults rfl omitted completed rfl (result limit p positive)

theorem model_result (limit : Int) (p : Bounds) (positive : Positive p) :
    ReturnsValue program "solve" (values limit p)
      (number (ProjectEulerSolutions.P73.solve
        limit p.lowerNum p.lowerDen p.upperNum p.upperDen)) := by
  rw [ProjectEulerSolutions.P73.solve_eq_result limit p positive]
  exact result limit p positive

end solve
end solvers.«73»
