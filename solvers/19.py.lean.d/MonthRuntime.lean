import solvers.«19.py.lean.d».CalendarRuntime
import lang.py.FormalSW.HeapRuntimeCostComparisons
import lang.py.FormalSW.HeapRuntimeCostBooleans

namespace solvers.«19».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P19Ast.module

def yearBody : List Stmt :=
  match executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 with
  | [_, _, .forIn _ _ body _, _] => body
  | _ => []

def monthBody : List Stmt :=
  match yearBody with
  | [.forIn _ _ body _] => body
  | _ => []

def CountEnv (env : HeapCalls.Env) (first last weekday count : Int) : Prop :=
  env "start_year" = some (.scalar (.int first)) ∧ env "end_year" = some (.scalar (.int last)) ∧
  env "dow_first_of_month" = some (.scalar (.int weekday)) ∧ env "count" = some (.scalar (.int count))

theorem CountEnv.set_other (h : CountEnv env first last weekday count) (name : String) (value : Mutable.Value)
    (ordinary : name ≠ "start_year" ∧ name ≠ "end_year" ∧ name ≠ "dow_first_of_month" ∧ name ≠ "count") :
    CountEnv (setLocal env name value) first last weekday count := by
  simpa [CountEnv, setLocal, Ne.symm ordinary.1, Ne.symm ordinary.2.1, Ne.symm ordinary.2.2.1,
    Ne.symm ordinary.2.2.2] using h

theorem CountEnv.set_count (h : CountEnv env first last weekday count) (next : Int) :
    CountEnv (setLocal env "count" (.scalar (.int next))) first last weekday next := by
  simpa [CountEnv, setLocal] using And.intro h.1 (And.intro h.2.1 h.2.2.1)

theorem CountEnv.set_weekday (h : CountEnv env first last weekday count) (next : Int) :
    CountEnv (setLocal env "dow_first_of_month" (.scalar (.int next))) first last next count := by
  simpa [CountEnv, setLocal] using And.intro h.1 (And.intro h.2.1 h.2.2.2)

private def yearTest : Expr :=
  match monthBody with
  | .ifThen (.boolOp _ (test :: _)) _ _ :: _ => test
  | _ => .constant .none

private def countTest : Expr :=
  match monthBody with
  | .ifThen test _ _ :: _ => test
  | _ => .constant .none

private theorem year_test_cost (first last year weekday count : Int) (heap : Heap) (env : HeapCalls.Env)
    (saved : CountEnv env first last weekday count) (getYear : env "year" = some (.scalar (.int year))) :
    ∃ flag : Bool, EvaluatesIn .word program env yearTest heap 13 (.scalar (.bool flag)) heap := by
  by_cases eligible : first ≤ year
  · refine ⟨decide (year ≤ last), ?_⟩
    apply EvaluatesIn.chain (ca := 1) (cb := 10) (EvaluatesIn.lookup saved.1)
    apply EvaluatesIn.ComparesIn.next (ca := 1) (cb := 5) (work := 2)
      (comparisonResult := .bool true) (EvaluatesIn.lookup getYear)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, eligible]
    · rfl
    · rfl
    · apply EvaluatesIn.ComparesIn.last (cost := 1) (work := 2) (EvaluatesIn.lookup saved.2.1)
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      · rfl
  · refine ⟨false, ?_⟩
    apply EvaluatesIn.mono (small := 8) (larger := by omega)
    apply EvaluatesIn.chain (ca := 1) (cb := 5) (EvaluatesIn.lookup saved.1)
    apply EvaluatesIn.ComparesIn.stop (cost := 1) (work := 2) (answer := .bool false)
      (EvaluatesIn.lookup getYear)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, eligible]
    · rfl
    · rfl

private theorem count_test_cost (first last year weekday count : Int) (heap : Heap) (env : HeapCalls.Env)
    (saved : CountEnv env first last weekday count) (getYear : env "year" = some (.scalar (.int year))) :
    ∃ flag : Bool, EvaluatesIn .word program env countTest heap 24 (.scalar (.bool flag)) heap := by
  obtain ⟨flag, tested⟩ := year_test_cost first last year weekday count heap env saved getYear
  cases flag with
  | false =>
      refine ⟨false, ?_⟩
      apply EvaluatesIn.mono (small := 15) (larger := by omega)
      exact EvaluatesIn.boolean_stop tested rfl rfl
  | true =>
      refine ⟨decide (weekday = 6), ?_⟩
      apply EvaluatesIn.boolean_next (ca := 13) (cb := 9) tested rfl rfl
      apply EvaluatesIn.boolean_last (cost := 7) (flag := decide (weekday = 6))
      · apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup saved.2.2.1) EvaluatesIn.constant
        · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        · rfl
      · rfl

/-- One month includes the guard, optional counter update, calendar call and remainder. -/
theorem month_body_cost (first last year month weekday count : Int) (heap : Heap) (env : HeapCalls.Env)
    (saved : CountEnv env first last weekday count) (getYear : env "year" = some (.scalar (.int year)))
    (getMonth : env "month" = some (.scalar (.int month))) :
    ∃ extra finalEnv, ∃ nextWeekday nextCount : Int,
      BlockContinuesIn .word program env monthBody heap 131 finalEnv (heap ++ extra) ∧
      CountEnv finalEnv first last nextWeekday nextCount ∧
      finalEnv "year" = some (.scalar (.int year)) := by
  obtain ⟨flag, tested⟩ := count_test_cost first last year weekday count heap env saved getYear
  let nextCount := if flag then count + 1 else count
  let middleEnv := if flag then setLocal env "count" (.scalar (.int nextCount)) else env
  have counted : BlockContinuesIn .word program env (monthBody.take 1) heap 33 middleEnv heap := by
    apply BlockContinuesIn.branch (cc := 24) (cb := 7) (flag := flag) tested (by cases flag <;> rfl)
    cases chosen : flag with
    | false =>
        simpa only [middleEnv, chosen, Bool.false_eq_true, reduceIte] using
          (BlockContinuesIn.nil (arithmetic := .word) (module := program) (env := env) (heap := heap)).mono (by omega : 0 ≤ 7)
    | true =>
        simpa only [middleEnv, nextCount, chosen, reduceIte] using
          (BlockContinuesIn.augAssign (module := program) (heap := heap) (work := 1) saved.2.2.2
            (EvaluatesIn.constant (value := .int 1)) (op := .add) rfl rfl)
  have middleSaved : CountEnv middleEnv first last weekday nextCount := by
    cases chosen : flag
    · simpa [middleEnv, nextCount, chosen] using saved
    · simpa [middleEnv, chosen] using saved.set_count nextCount
  have middleYear : middleEnv "year" = some (.scalar (.int year)) := by
    cases chosen : flag <;> simpa [middleEnv, chosen, setLocal] using getYear
  have middleMonth : middleEnv "month" = some (.scalar (.int month)) := by
    cases chosen : flag <;> simpa [middleEnv, chosen, setLocal] using getMonth
  obtain ⟨extra, days, calendar⟩ := month_cost year month heap
  let nextWeekday := (weekday + days) % 7
  have update : EvaluatesIn .word program middleEnv
      (.binOp (.binOp (.name "dow_first_of_month") .add
        (.call (.name "days_in_month") [.name "year", .name "month"])) .mod (.constant (.int 7)))
      heap 96 (.scalar (.int nextWeekday)) (heap ++ extra) := by
    apply EvaluatesIn.binary (ca := 91) (cb := 1) (work := 1)
      (av := .int (weekday + days)) (bv := .int 7)
    · apply EvaluatesIn.binary (ca := 1) (cb := 86) (work := 1) (av := .int weekday) (bv := .int days)
        (EvaluatesIn.lookup middleSaved.2.2.1)
      · exact EvaluatesIn.call (ArgumentsIn.cons (EvaluatesIn.lookup middleYear)
          (ArgumentsIn.cons (EvaluatesIn.lookup middleMonth) ArgumentsIn.nil)) calendar
      · rfl
      · rfl
    · exact EvaluatesIn.constant
    · rfl
    · rfl
  refine ⟨extra, setLocal middleEnv "dow_first_of_month" (.scalar (.int nextWeekday)), nextWeekday, nextCount, ?_,
    middleSaved.set_weekday nextWeekday, ?_⟩
  · exact counted.append (BlockContinuesIn.assign update)
  · simpa [setLocal] using middleYear

end solvers.«19».runtimeCost
