import «.formalsw».solvers.«57.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p57.py»
import FormalSW
import ProjectEulerSolutions.Lib.SqrtConvergents

namespace solvers.«57»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.SqrtConvergents
set_option Elab.async false
private abbrev program := Solvers.P57Ast.module
private abbrev number (value : Int) : Constant := .int value
private abbrev natural (value : Nat) : Constant := .int (value : Int)

private abbrev lengthExpression (name : String) : Expr := .builtinCall .len [.builtinCall .str [.name name]]
private abbrev digitTest : Expr := .compare (lengthExpression "p") [(.gt, lengthExpression "q")]
private abbrev nextP : Expr := .binOp (.name "p") .add (.binOp (.constant (.int 2)) .mult (.name "q"))
private abbrev nextQ : Expr := .binOp (.name "p") .add (.name "q")
private abbrev update : Stmt := .assign [.tuple [.name "p", .name "q"]] (.tuple [nextP, nextQ])
private abbrev loopBody : List Stmt :=
  [.ifThen digitTest [.augAssign (.name "cnt") .add (.constant (.int 1))] [], update]

private structure Values (env : ScalarEnv) (first count : Nat) : Prop where
  numerator : env "p" = some (natural (convergent first).1)
  denominator : env "q" = some (natural (convergent first).2)
  counter : env "cnt" = some (natural count)

private theorem Values.bindIndex {env : ScalarEnv} {first count : Nat} (values : Values env first count) (next : Int) :
    Values (env.set "_" (number next)) first count :=
  ⟨by simpa [ScalarEnv.set] using values.numerator,
    by simpa [ScalarEnv.set] using values.denominator,
    by simpa [ScalarEnv.set] using values.counter⟩

private theorem Values.setCount {env : ScalarEnv} {first count : Nat} (values : Values env first count) (updated : Nat) :
    Values (env.set "cnt" (natural updated)) first updated :=
  ⟨by simpa [ScalarEnv.set] using values.numerator,
    by simpa [ScalarEnv.set] using values.denominator,
    by simp [ScalarEnv.set]⟩

/-- The actual str/len calls return the exact positional digit length of a positive component. -/
private theorem length_evaluates (env : ScalarEnv) (name : String) (n : Nat) (positive : 0 < n)
    (found : env name = some (natural n)) :
    Evaluates program env (lengthExpression name) (natural (Nat.digits 10 n).length) := by
  rw [← printed_length n positive]
  apply Evaluates.builtin (values := [.string (toString (n : Int))])
  · exact .cons (Evaluates.builtin (.cons (Evaluates.lookup found) .nil) rfl) .nil
  · rfl

private theorem test_evaluates (env : ScalarEnv) (first count : Nat) (values : Values env first count) :
    Evaluates program env digitTest (.bool (decide
      ((Nat.digits 10 (convergent first).2).length < (Nat.digits 10 (convergent first).1).length))) := by
  have positive := convergent_positive first
  apply Evaluates.compare
    (length_evaluates env "p" (convergent first).1 positive.1 values.numerator)
    (length_evaluates env "q" (convergent first).2 positive.2 values.denominator)
  simp [scalarCompare, evalCmpOp]

/-- Both new components are evaluated in the old environment before the two names are assigned. -/
private theorem advance (env : ScalarEnv) (first count : Nat) (values : Values env first count)
    (following : List Stmt) (answer : Constant)
    (after : ∀ updated, Values updated (first+1) count → BlockReturns program updated following answer) :
    BlockReturns program env (update :: following) answer := by
  apply BlockReturns.unpack (names := ["p", "q"])
    (results := [natural (convergent (first+1)).1, natural (convergent (first+1)).2]) rfl rfl
  · exact .cons
      (Evaluates.binary (Evaluates.lookup values.numerator)
        (Evaluates.binary Evaluates.constant (Evaluates.lookup values.denominator) rfl) rfl)
      (.cons (Evaluates.binary (Evaluates.lookup values.numerator) (Evaluates.lookup values.denominator) rfl) .nil)
  · apply after
    constructor
    · simp [assignNames, ScalarEnv.set]
    · simp [assignNames, ScalarEnv.set]
    · simpa [assignNames, ScalarEnv.set] using values.counter

/-- Remaining fuel, rather than growing numerators, proves termination of the unchanged source loop. -/
private theorem loop (remaining first count : Nat) (next stop : Int) (equation : stop-next = (remaining : Int))
    (env : ScalarEnv) (values : Values env first count) :
    BlockReturns program env
      [.rangeStepLoop "_" next stop 1 loopBody [], .return (.name "cnt")]
      (natural (count + sumFrom first remaining)) := by
  induction remaining generalizing first count next env with
  | zero =>
      simp only [sumFrom_zero, Nat.add_zero]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup values.counter)
  | succ remaining ih =>
      let bound := env.set "_" (number next)
      have boundValues : Values bound first count := values.bindIndex next
      have tested := test_evaluates bound first count boundValues
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturns.branch tested
      by_cases more : (Nat.digits 10 (convergent first).2).length < (Nat.digits 10 (convergent first).1).length
      · have won : weight first = 1 := by simp [weight, indicator, more]
        simp only [more, decide_true, scalarTruth, if_true, List.cons_append, List.nil_append]
        apply BlockReturns.augAssign (value := natural (count+1))
          (Evaluates.binary (Evaluates.lookup boundValues.counter) Evaluates.constant rfl)
        apply advance _ first (count+1) (boundValues.setCount (count+1))
        intro updated nextValues
        have continued := ih (first+1) (count+1) (next+1) (by omega) updated nextValues
        simpa only [sumFrom_succ, won, Nat.add_assoc, loopBody, Calls.loopContinuation,
          Calls.loopControl, Calls.loopControlStmt, Bool.false_eq_true, if_false] using continued
      · have lost : weight first = 0 := by simp [weight, indicator, more]
        simp only [more, decide_false, scalarTruth]
        apply advance bound first count boundValues
        intro updated nextValues
        have continued := ih (first+1) count (next+1) (by omega) updated nextValues
        simpa only [sumFrom_succ, lost, Nat.zero_add, loopBody, Calls.loopContinuation,
          Calls.loopControl, Calls.loopControlStmt, Bool.false_eq_true, if_false] using continued

private abbrev params := ((findFunctionIn program.body "solve").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn program.body "solve").getD ([], [])).2

private theorem body_result (limit : Int) (env : ScalarEnv) (getLimit : env "limit" = some (number limit)) :
    BlockReturns program env (executableBody sourceBody) (natural (ProjectEulerStatements.P57.naive limit.toNat)) := by
  let paired := assignNames env ["p", "q"] [natural 3, natural 2]
  let ready := paired.set "cnt" (natural 0)
  have readyValues : Values ready 0 0 := by
    constructor <;> simp [ready, paired, assignNames, ScalarEnv.set, convergent, ProjectEulerStatements.P57.convergent]
  have readyLimit : ready "limit" = some (number limit) := by
    simpa [ready, paired, assignNames, ScalarEnv.set] using getLimit
  apply BlockReturns.unpack (names := ["p", "q"]) (results := [natural 3, natural 2]) rfl rfl
  · exact .cons Evaluates.constant (.cons Evaluates.constant .nil)
  · apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.for_range (values := [number 1, number (limit+1)])
      (.cons Evaluates.constant
        (.cons (Evaluates.binary (Evaluates.lookup readyLimit) Evaluates.constant rfl) .nil)) rfl
    by_cases positive : 0 < limit
    · have executed := loop limit.toNat 0 0 1 (limit+1) (by omega) ready readyValues
      simpa only [Nat.zero_add, ← naive_eq_sumFrom] using executed
    · have empty : limit.toNat = 0 := by omega
      simp only [empty, ProjectEulerStatements.P57.naive, List.range_zero, List.foldl_nil]
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup readyValues.counter)

namespace solve

/-- The unchanged source agrees with the genuine structural reference model for every signed limit. -/
theorem result (limit : Int) :
    ReturnsValue program "solve" [number limit] (natural (ProjectEulerStatements.P57.naive limit.toNat)) := by
  apply BlockReturns.function rfl rfl
  exact body_result limit _ rfl

theorem «Returns» (limit : Int) : (Calls.function program "solve" [number limit]).Returns := (result limit).returns

/-- The interpreted iterative and memoized recursive sources have the same outcome for every signed limit. -/
theorem «Equiv(naive)» (limit : Int) :
    (Calls.function program "solve" [number limit]).Equiv
      (HeapMemoCalls.scalarFunction P57Ast.module "naive" [.int limit]) := by
  have optimized := (result limit).finishes
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p57.naive.result limit []
  obtain ⟨after, _, original⟩ := reference.scalar_finishes (values := [.int limit])
    (result := .int (ProjectEulerStatements.P57.naive limit.toNat))
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem «Supported» :
    (∀ limit : Int, (Calls.function program "solve" [number limit]).Supported) ∧
    (∀ limit : Int, (HeapMemoCalls.scalarFunction P57Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p57.naive.«Returns» limit).supported

end solve
end solvers.«57»
