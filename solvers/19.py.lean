import «.formalsw».solvers.«19.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p19.py»
import FormalSW

namespace solvers.«19»
open FormalSW.PyAst
open ProjectEulerStatements.P19 (isLeapInt)

private abbrev program := Solvers.P19Ast.module

theorem is_leap_year.result (year : Int) :
    Finishes (scalarFunction program "is_leap_year" [.int year]).step
      (scalarFunction program "is_leap_year" [.int year]).initial (.ok (.bool (isLeapInt year))) := by
  apply Execution.finishes_of_observe (fuel := 4)
  rw [ProjectEulerStatements.P19.isLeapInt_eq_century_test]
  by_cases century : year % 400 = 0 <;> by_cases hundred : year % 100 = 0 <;>
    simp [Execution.observe, Execution.observeFrom, scalarFunction, Solvers.P19Ast.module, findFunctionIn,
      scalarBlock, executableBody, scalarStep, scalarExpr, scalarBinary, scalarCompare,
      argumentEnv, scalarTruth, evalCmpOp, century, hundred]

theorem is_leap_year.«Equiv(is_leap)» (year : Int) :
    (scalarFunction program "is_leap_year" [.int year]).Equiv
      (Calls.function P19Ast.module "is_leap" [.int year]) := by
  have optimized := is_leap_year.result year
  have reference :=
    (data.«project-euler-statements».data.py.project_euler_statements.p19.is_leap.result year).finishes
  intro outcome
  constructor
  · intro result
    rw [result.unique optimized]
    exact reference
  · intro result
    rw [result.unique reference]
    exact optimized

open FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P19 (daysInMonthInt)

private theorem remainderTest (year divisor : Int) (positive : 0 < divisor) (heap : Mutable.Heap)
    (env : Mutable.Env) (lookup : env "year" = some (.scalar (.int year))) :
    Evaluates program env (.compare (.binOp (.name "year") .mod (.constant (.int divisor)))
      [(.eq, .constant (.int 0))]) heap (.scalar (.bool (decide (year % divisor = 0)))) heap := by
  apply Evaluates.compare (av := .scalar (.int (year % divisor))) (bv := .scalar (.int 0))
    (middle := heap) (afterRight := heap)
    (Evaluates.binary (finalHeap := heap) (Evaluates.lookup lookup) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, show divisor ≠ 0 by omega, positive])) Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem is_leap_year.heap_result (year : Int) (heap : Mutable.Heap) :
    ReturnsValue program "is_leap_year" [.scalar (.int year)] heap (.scalar (.bool (isLeapInt year))) heap := by
  let env := functionEnv [{name := "year", annotation := some (.name "int")}]
    ((findFunctionIn program.body "is_leap_year").getD ([], [])).2 [.scalar (.int year)] []
  have fourHundred := remainderTest year 400 (by decide) heap env rfl
  have hundred := remainderTest year 100 (by decide) heap env rfl
  have fourth := remainderTest year 4 (by decide) heap env rfl
  apply BlockReturns.function rfl rfl
  rw [ProjectEulerStatements.P19.isLeapInt_eq_century_test]
  by_cases century : year % 400 = 0
  · simp only [century, ↓reduceIte]
    apply BlockReturns.branch (flag := true) (value := .scalar (.bool true))
    · simpa only [century, decide_true] using fourHundred
    · rfl
    · exact BlockReturns.return_value Evaluates.constant
  · simp only [century, ↓reduceIte]
    apply BlockReturns.branch (flag := false) (value := .scalar (.bool false))
    · simpa only [century, decide_false] using fourHundred
    · rfl
    · by_cases century : year % 100 = 0
      · simp only [century, ↓reduceIte]
        apply BlockReturns.branch (flag := true) (value := .scalar (.bool true))
        · simpa only [century, decide_true] using hundred
        · rfl
        · exact BlockReturns.return_value Evaluates.constant
      · simp only [century, ↓reduceIte]
        apply BlockReturns.branch (flag := false) (value := .scalar (.bool false))
        · simpa only [century, decide_false] using hundred
        · rfl
        · exact BlockReturns.return_value fourth

private abbrev shortMonths : List Mutable.Value := ([4, 6, 9, 11] : List Int).map (fun n => .scalar (.int n))

private theorem shortMonths.result (month : Int) (heap : Mutable.Heap) (env : Mutable.Env)
    (lookup : env "month" = some (.scalar (.int month))) :
    Evaluates program env (.membership (.name "month")
      (.tuple [.constant (.int 4), .constant (.int 6), .constant (.int 9), .constant (.int 11)]) false)
      heap (.scalar (.bool (decide (month ∈ ([4, 6, 9, 11] : List Int))))) (heap ++ [.tuple shortMonths]) := by
  apply Evaluates.membership
    (.cons (Evaluates.lookup lookup)
      (.cons (Evaluates.tuple (.cons .constant (.cons .constant (.cons .constant (.cons .constant .nil)))) (by simp)) .nil))
  exact membership_tuple_ints _ heap.length month [4, 6, 9, 11] (by simp) false

theorem days_in_month.heap_result (year : Int) (month : Nat) (positive : 0 < month) (bound : month ≤ 12)
    (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "days_in_month" [.scalar (.int year), .scalar (.int month)] heap
      (.scalar (.int (daysInMonthInt year month))) (heap ++ extra) := by
  let env := functionEnv
    [{name := "year", annotation := some (.name "int")}, {name := "month", annotation := some (.name "int")}]
    ((findFunctionIn program.body "days_in_month").getD ([], [])).2
    [.scalar (.int year), .scalar (.int month)] []
  have february : Evaluates program env (.compare (.name "month") [(.eq, .constant (.int 2))]) heap
      (.scalar (.bool (decide (month = 2)))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  by_cases second : month = 2
  · subst month
    refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := true) (value := .scalar (.bool true)) february rfl
    apply BlockReturns.return_value
    apply Evaluates.conditional (flag := isLeapInt year)
      (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (is_leap_year.heap_result year heap)) rfl
    cases leap : isLeapInt year <;> simpa [daysInMonthInt, leap] using
      (Evaluates.constant (module := program) (env := _) (heap := heap) (value := .int _))
  · have length : daysInMonthInt year month = if (month : Int) ∈ ([4, 6, 9, 11] : List Int) then 30 else 31 := by
      interval_cases month <;> simp_all [daysInMonthInt]
    have membership := shortMonths.result (month : Int) heap env rfl
    refine ⟨[.tuple shortMonths], ?_⟩
    rw [length]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := false) (value := .scalar (.bool false))
    · simpa only [second, decide_false] using february
    · rfl
    · apply BlockReturns.branch (flag := decide ((month : Int) ∈ ([4, 6, 9, 11] : List Int))) membership rfl
      by_cases short : (month : Int) ∈ ([4, 6, 9, 11] : List Int)
      · simp only [short, decide_true, ↓reduceIte]
        exact BlockReturns.return_value Evaluates.constant
      · simp only [short, decide_false, Bool.false_eq_true, ↓reduceIte]
        exact BlockReturns.return_value Evaluates.constant

open ProjectEulerStatements.P19 (dayOfWeek daysInMonth sundaysInMonths sundaysInYear sundaysInYears eligibleYears)

private def CountEnv (env : Mutable.Env) (startYear endYear weekday : Nat) (count : Int) : Prop :=
  env "start_year" = some (.scalar (.int startYear)) ∧ env "end_year" = some (.scalar (.int endYear)) ∧
    env "dow_first_of_month" = some (.scalar (.int weekday)) ∧ env "count" = some (.scalar (.int count))

private theorem CountEnv.set_other (h : CountEnv env startYear endYear weekday count) (name : String) (value : Mutable.Value)
    (ordinary : name ≠ "start_year" ∧ name ≠ "end_year" ∧ name ≠ "dow_first_of_month" ∧ name ≠ "count") :
    CountEnv (setLocal env name value) startYear endYear weekday count := by
  simpa [CountEnv, setLocal, Ne.symm ordinary.1, Ne.symm ordinary.2.1, Ne.symm ordinary.2.2.1,
    Ne.symm ordinary.2.2.2] using h

private theorem CountEnv.set_count (h : CountEnv env startYear endYear weekday count) (next : Int) :
    CountEnv (setLocal env "count" (.scalar (.int next))) startYear endYear weekday next := by
  simpa [CountEnv, setLocal] using And.intro h.1 (And.intro h.2.1 h.2.2.1)

private theorem CountEnv.set_weekday (h : CountEnv env startYear endYear weekday count) (next : Nat) :
    CountEnv (setLocal env "dow_first_of_month" (.scalar (.int next))) startYear endYear next count := by
  simpa [CountEnv, setLocal] using And.intro h.1 (And.intro h.2.1 h.2.2.2)

private abbrev yearRange : Expr :=
  .compare (.name "start_year") [(.ltE, .name "year"), (.ltE, .name "end_year")]

private theorem yearRange.result (startYear endYear year weekday : Nat) (count : Int)
    (heap : Mutable.Heap) (env : Mutable.Env) (saved : CountEnv env startYear endYear weekday count)
    (getYear : env "year" = some (.scalar (.int year))) (active : year ≤ endYear) :
    Evaluates program env yearRange heap (.scalar (.bool (decide (startYear ≤ year)))) heap := by
  by_cases eligible : startYear ≤ year
  · simp only [eligible, decide_true]
    apply Evaluates.chain (Evaluates.lookup saved.1)
    apply Evaluates.Compares.next (value := .scalar (.int year)) (compared := .scalar (.bool true))
      (afterComparison := heap) (Evaluates.lookup getYear)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, eligible]
    · rfl
    · apply Evaluates.Compares.last (middle := heap) (Evaluates.lookup saved.2.1)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, active]
  · simp only [eligible, decide_false]
    apply Evaluates.chain (Evaluates.lookup saved.1)
    apply Evaluates.Compares.stop (middle := heap) (Evaluates.lookup getYear)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, eligible]
    · rfl

private abbrev countTest : Expr := .boolOp .and
  [yearRange, .compare (.name "dow_first_of_month") [(.eq, .constant (.int 6))]]

private theorem countTest.result (startYear endYear year weekday : Nat) (count : Int)
    (heap : Mutable.Heap) (env : Mutable.Env) (saved : CountEnv env startYear endYear weekday count)
    (getYear : env "year" = some (.scalar (.int year))) (active : year ≤ endYear) :
    Evaluates program env countTest heap (.scalar (.bool (decide (startYear ≤ year ∧ weekday = 6)))) heap := by
  have range := yearRange.result startYear endYear year weekday count heap env saved getYear active
  by_cases eligible : startYear ≤ year
  · simp only [eligible, true_and]
    apply Evaluates.boolean_next (middle := heap) (value := .scalar (.bool true))
    · simpa only [eligible, decide_true] using range
    · rfl
    · rfl
    · apply Evaluates.boolean_last (flag := decide (weekday = 6))
      · apply Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup saved.2.2.1) Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
      · rfl
  · simp only [eligible, false_and, decide_false]
    apply Evaluates.boolean_stop (value := .scalar (.bool false))
    · simpa only [eligible, decide_false] using range
    · rfl
    · rfl

private abbrev countBody : List Stmt := [.ifThen countTest [.augAssign (.name "count") .add (.constant (.int 1))] []]
private abbrev nextWeekday : Expr := .binOp
  (.binOp (.name "dow_first_of_month") .add (.call (.name "days_in_month") [.name "year", .name "month"]))
  .mod (.constant (.int 7))
private abbrev monthBody : List Stmt := countBody ++ [.assign [.name "dow_first_of_month"] nextWeekday]

private theorem monthBody.result (startYear endYear year month : Nat) (count : Int)
    (heap : Mutable.Heap) (env : Mutable.Env) (saved : CountEnv env startYear endYear (dayOfWeek year month 1) count)
    (getYear : env "year" = some (.scalar (.int year))) (getMonth : env "month" = some (.scalar (.int month)))
    (active : year ≤ endYear) (positive : 0 < month) (bound : month ≤ 12) :
    ∃ extra finalEnv, BlockContinues program env monthBody heap finalEnv (heap ++ extra) ∧
      CountEnv finalEnv startYear endYear (dayOfWeek year (month + 1) 1)
        (count + if startYear ≤ year ∧ dayOfWeek year month 1 = 6 then 1 else 0) ∧
      finalEnv "year" = some (.scalar (.int year)) := by
  let selected := decide (startYear ≤ year ∧ dayOfWeek year month 1 = 6)
  let nextCount := count + if selected then 1 else 0
  let middleEnv := if selected then setLocal env "count" (.scalar (.int nextCount)) else env
  have test := countTest.result startYear endYear year (dayOfWeek year month 1) count heap env saved getYear active
  have counted : BlockContinues program env countBody heap middleEnv heap := by
    apply BlockContinues.branch (flag := selected) test rfl
    cases chosen : selected with
    | false => simpa [middleEnv, chosen] using (BlockContinues.nil (module := program) (env := env) (heap := heap))
    | true =>
        simpa [middleEnv, nextCount, chosen] using
          (BlockContinues.augAssign (module := program) (heap := heap) saved.2.2.2
            (Evaluates.constant (value := .int 1)) (op := .add) rfl)
  have savedMiddle : CountEnv middleEnv startYear endYear (dayOfWeek year month 1) nextCount := by
    cases chosen : selected with
    | false => simpa [middleEnv, nextCount, chosen] using saved
    | true => simpa [middleEnv, chosen] using saved.set_count nextCount
  have yearMiddle : middleEnv "year" = some (.scalar (.int year)) := by
    cases chosen : selected <;> simpa [middleEnv, chosen, setLocal] using getYear
  have monthMiddle : middleEnv "month" = some (.scalar (.int month)) := by
    cases chosen : selected <;> simpa [middleEnv, chosen, setLocal] using getMonth
  obtain ⟨extra, days⟩ := days_in_month.heap_result year month positive bound heap
  rw [ProjectEulerStatements.P19.daysInMonthInt_nat] at days
  have advanced : ((dayOfWeek year month 1 : Int) + (daysInMonth year month : Int)) % 7 =
      (dayOfWeek year (month + 1) 1 : Int) := by
    rw [ProjectEulerStatements.P19.firstOfMonth_succ year month positive]
    simp only [Int.natCast_emod, Nat.cast_add, Nat.cast_ofNat]
  have update : Evaluates program middleEnv nextWeekday heap
      (.scalar (.int (dayOfWeek year (month + 1) 1))) (heap ++ extra) := by
    refine Evaluates.binary (middle := heap ++ extra)
      (av := .scalar (.int ((dayOfWeek year month 1 : Int) + (daysInMonth year month : Int))))
      ?_ Evaluates.constant ?_
    · exact Evaluates.binary (Evaluates.lookup savedMiddle.2.2.1)
        (Evaluates.call (.cons (Evaluates.lookup yearMiddle) (.cons (Evaluates.lookup monthMiddle) .nil)) days) rfl
    · simp [Mutable.binary, scalarBinary, advanced]
  refine ⟨extra, setLocal middleEnv "dow_first_of_month" (.scalar (.int (dayOfWeek year (month + 1) 1))),
    counted.append (BlockContinues.assign update), ?_, ?_⟩
  · simpa only [nextCount, selected, decide_eq_true_eq] using savedMiddle.set_weekday (dayOfWeek year (month + 1) 1)
  · simpa [setLocal] using yearMiddle

private theorem monthLoop.result (startYear endYear year month count : Nat) (initial : Int)
    (heap : Mutable.Heap) (env : Mutable.Env) (saved : CountEnv env startYear endYear (dayOfWeek year month 1) initial)
    (getYear : env "year" = some (.scalar (.int year))) (active : year ≤ endYear)
    (positive : 0 < month) (stop : month + count = 13) :
    ∃ extra finalEnv, BlockContinues program env
      [.rangeStepLoop "month" (month : Int) ((month + count : Nat) : Int) 1 monthBody []]
      heap finalEnv (heap ++ extra) ∧
      CountEnv finalEnv startYear endYear (dayOfWeek year 13 1)
        (initial + if startYear ≤ year then (sundaysInMonths year month count : Int) else 0) ∧
      finalEnv "year" = some (.scalar (.int year)) := by
  induction count generalizing month initial heap env with
  | zero =>
      have last : month = 13 := by omega
      subst month
      refine ⟨[], env, ?_, ?_, getYear⟩
      · simp only [List.append_nil, Nat.add_zero]
        exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
      · simpa only [ProjectEulerStatements.P19.sundaysInMonths_zero, Nat.cast_zero, ite_self, Int.add_zero] using saved
  | succ count ih =>
      let withMonth := setLocal env "month" (.scalar (.int month))
      have savedMonth : CountEnv withMonth startYear endYear (dayOfWeek year month 1) initial :=
        saved.set_other "month" _ (by decide)
      have yearMonth : withMonth "year" = some (.scalar (.int year)) := by simpa [withMonth, setLocal] using getYear
      obtain ⟨allocated, middleEnv, body, savedBody, yearBody⟩ := monthBody.result startYear endYear year month initial
        heap withMonth savedMonth yearMonth (by simp [withMonth, setLocal]) active positive (by omega)
      obtain ⟨later, finalEnv, continued, savedFinal, yearFinal⟩ := ih (month + 1) _ (heap ++ allocated) middleEnv
        savedBody yearBody (by omega) (by omega)
      refine ⟨allocated ++ later, finalEnv, ?_, ?_, yearFinal⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues])
        change BlockContinues program withMonth
          (monthBody ++ [.rangeStepLoop "month" ((month : Int) + 1) ((month + (count + 1) : Nat) : Int) 1 monthBody []])
          heap finalEnv (heap ++ (allocated ++ later))
        simpa only [Nat.cast_add, Nat.cast_one, Int.add_assoc, Int.add_left_comm, Int.add_comm,
          List.append_assoc] using body.append continued
      · by_cases eligible : startYear ≤ year <;> by_cases sunday : dayOfWeek year month 1 = 6 <;>
          simpa [ProjectEulerStatements.P19.sundaysInMonths_succ, eligible, sunday, Nat.cast_add, Int.add_assoc] using savedFinal

private abbrev yearBody : List Stmt :=
  [.forIn (.name "month") (.call (.name "range") [.constant (.int 1), .constant (.int 13)]) monthBody []]

private theorem yearBody.result (startYear endYear year : Nat) (initial : Int) (heap : Mutable.Heap)
    (env : Mutable.Env) (saved : CountEnv env startYear endYear (dayOfWeek year 1 1) initial)
    (getYear : env "year" = some (.scalar (.int year))) (afterEpoch : 1900 ≤ year) (active : year ≤ endYear) :
    ∃ extra finalEnv, BlockContinues program env yearBody heap finalEnv (heap ++ extra) ∧
      CountEnv finalEnv startYear endYear (dayOfWeek (year + 1) 1 1)
        (initial + if startYear ≤ year then (sundaysInYear year : Int) else 0) := by
  obtain ⟨extra, finalEnv, loop, savedFinal, _⟩ := monthLoop.result startYear endYear year 1 12 initial heap env
    saved getYear active (by decide) rfl
  refine ⟨extra, finalEnv, BlockContinues.for_range (.cons .constant (.cons .constant .nil)) rfl loop, ?_⟩
  simpa only [sundaysInYear, ProjectEulerStatements.P19.firstOfYear_after_months year afterEpoch] using savedFinal

private theorem yearLoop.result (startYear endYear year count : Nat) (initial : Int)
    (heap : Mutable.Heap) (env : Mutable.Env) (saved : CountEnv env startYear endYear (dayOfWeek year 1 1) initial)
    (afterEpoch : 1900 ≤ year) (stop : year + count = endYear + 1) :
    ∃ extra finalEnv, BlockContinues program env
      [.rangeStepLoop "year" (year : Int) ((year + count : Nat) : Int) 1 yearBody []] heap finalEnv (heap ++ extra) ∧
      CountEnv finalEnv startYear endYear (dayOfWeek (year + count) 1 1)
        (initial + (eligibleYears startYear year count : Int)) := by
  induction count generalizing year initial heap env with
  | zero =>
      refine ⟨[], env, ?_, ?_⟩
      · simp only [List.append_nil, Nat.add_zero]
        exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
      · simpa only [Nat.add_zero, ProjectEulerStatements.P19.eligibleYears_zero, Nat.cast_zero, Int.add_zero] using saved
  | succ count ih =>
      let withYear := setLocal env "year" (.scalar (.int year))
      have savedYear : CountEnv withYear startYear endYear (dayOfWeek year 1 1) initial :=
        saved.set_other "year" _ (by decide)
      obtain ⟨allocated, middleEnv, body, savedBody⟩ := yearBody.result startYear endYear year initial heap withYear
        savedYear (by simp [withYear, setLocal]) afterEpoch (by omega)
      obtain ⟨later, finalEnv, continued, savedFinal⟩ := ih (year + 1) _ (heap ++ allocated) middleEnv
        savedBody (by omega) (by omega)
      refine ⟨allocated ++ later, finalEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues])
        change BlockContinues program withYear
          (yearBody ++ [.rangeStepLoop "year" ((year : Int) + 1) ((year + (count + 1) : Nat) : Int) 1 yearBody []])
          heap finalEnv (heap ++ (allocated ++ later))
        simpa only [Nat.cast_add, Nat.cast_one, Int.add_assoc, Int.add_left_comm, Int.add_comm,
          List.append_assoc] using body.append continued
      · have years : year + 1 + count = year + (count + 1) := by omega
        by_cases eligible : startYear ≤ year
        · simpa only [ProjectEulerStatements.P19.eligibleYears_succ, if_pos eligible, Nat.cast_add,
            years, Int.add_assoc] using savedFinal
        · simpa only [ProjectEulerStatements.P19.eligibleYears_succ, if_neg eligible, Nat.cast_add,
            Nat.cast_zero, years, Int.add_zero, Int.zero_add] using savedFinal

theorem solve.result (heap : Mutable.Heap) (startYear endYear : Nat)
    (afterEpoch : 1900 ≤ startYear) (ordered : startYear ≤ endYear) :
    ∃ extra, ReturnsValue program "solve" [.scalar (.int startYear), .scalar (.int endYear)] heap
      (.scalar (.int (sundaysInYears startYear (endYear + 1 - startYear)))) (heap ++ extra) := by
  let env := functionEnv
    [{name := "start_year", annotation := some (.name "int")}, {name := "end_year", annotation := some (.name "int")}]
    ((findFunctionIn program.body "solve").getD ([], [])).2
    [.scalar (.int startYear), .scalar (.int endYear)] []
  let initialEnv := setLocal (setLocal env "dow_first_of_month" (.scalar (.int 0))) "count" (.scalar (.int 0))
  have saved : CountEnv initialEnv startYear endYear (dayOfWeek 1900 1 1) 0 := ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨extra, finalEnv, loop, savedFinal⟩ := yearLoop.result startYear endYear 1900 (endYear + 1 - 1900) 0 heap
    initialEnv saved (by omega) (by omega)
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range
    (.cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl) .nil)) rfl
  have actualLoop : BlockContinues program initialEnv
      [.rangeStepLoop "year" 1900 ((endYear : Int) + 1) 1 yearBody []] heap finalEnv (heap ++ extra) := by
    simpa only [show (1900 + (endYear + 1 - 1900) : Nat) = endYear + 1 by omega, Nat.cast_add, Nat.cast_one] using loop
  apply actualLoop.returns
  apply BlockReturns.return_value
  exact Evaluates.lookup (by simpa only [Int.zero_add,
    ProjectEulerStatements.P19.eligibleYears_before startYear 1900 (endYear + 1) afterEpoch (by omega)] using savedFinal.2.2.2)

private abbrev CountCondition (startYear endYear : Int) := Calls.Condition program
  (fun name => if name = "start_year" then some (.int startYear) else if name = "end_year" then some (.int endYear) else none)
  Solvers.P19Ast.requires_solve_0

private theorem countConditionEvaluation (module : Module) (startYear endYear : Int) :
    Calls.Evaluates module
      (fun name => if name = "start_year" then some (.int startYear) else if name = "end_year" then some (.int endYear) else none)
      (.compare (.constant (.int 1900)) [(.ltE, .name "start_year"), (.ltE, .name "end_year")])
      (.bool (decide (1900 ≤ startYear ∧ startYear ≤ endYear))) := by
  intro stack
  let env : ScalarEnv := fun name => if name = "start_year" then some (.int startYear)
    else if name = "end_year" then some (.int endYear) else none
  apply Calls.Reaches.next rfl
  apply Calls.Reaches.next rfl
  apply Calls.Reaches.next rfl
  apply Calls.Reaches.next rfl
  by_cases lower : 1900 ≤ startYear
  · simp only [lower, true_and]
    apply Calls.Reaches.next (following :=
      ⟨.expr (.name "end_year") env, .chainRight .ltE (.int startYear) [] env :: stack⟩)
      (by simp [Calls.step, scalarCompare, evalCmpOp, lower, env])
    apply Calls.Reaches.next rfl
    by_cases upper : startYear ≤ endYear <;> simp only [upper, decide_true, decide_false]
    · exact Calls.Reaches.next (by simp [Calls.step, scalarCompare, evalCmpOp, upper]) .refl
    · exact Calls.Reaches.next (by simp [Calls.step, scalarCompare, evalCmpOp, upper]) .refl
  · simp only [lower, false_and, decide_false]
    exact Calls.Reaches.next (by simp [Calls.step, scalarCompare, evalCmpOp, lower]) .refl

private theorem CountCondition.domain (startYear endYear : Int) (condition : CountCondition startYear endYear) :
    1900 ≤ startYear ∧ startYear ≤ endYear := by
  have same := (countConditionEvaluation program startYear endYear).condition_iff.mp condition
  simpa only [Constant.bool.injEq, decide_eq_true_eq] using same

theorem solve.«Equiv(naive_years)» (startYear endYear : Int)
    (condition : CountCondition startYear endYear) :
    (HeapCalls.scalarFunction program "solve" [.int startYear, .int endYear]).Equiv
      (HeapCalls.scalarFunction P19Ast.module "naive_years" [.int startYear, .int endYear]) := by
  obtain ⟨afterEpoch, ordered⟩ := CountCondition.domain startYear endYear condition
  lift startYear to Nat using (by omega : 0 ≤ startYear)
  lift endYear to Nat using (by omega : 0 ≤ endYear)
  obtain ⟨extra, optimized⟩ := solve.result [] startYear endYear (by omega) (by omega)
  obtain ⟨other, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p19.naive_years.result
    [] startYear endYear (by omega) (by omega)
  have optimizedRun := ReturnsValue.scalar_finishes (values := [.int startYear, .int endYear]) optimized
  have referenceRun := ReturnsValue.scalar_finishes (values := [.int startYear, .int endYear]) reference
  intro outcome
  constructor
  · intro run
    rw [run.unique optimizedRun]
    exact referenceRun
  · intro run
    rw [run.unique referenceRun]
    exact optimizedRun

theorem solve.«Supported» :
    (∀ startYear endYear, CountCondition startYear endYear →
      (HeapCalls.scalarFunction program "solve" [.int startYear, .int endYear]).Supported) ∧
    (∀ startYear endYear, CountCondition startYear endYear →
      (HeapCalls.scalarFunction P19Ast.module "naive_years" [.int startYear, .int endYear]).Supported) := by
  constructor
  · intro startYear endYear condition
    obtain ⟨afterEpoch, ordered⟩ := CountCondition.domain startYear endYear condition
    lift startYear to Nat using (by omega : 0 ≤ startYear)
    lift endYear to Nat using (by omega : 0 ≤ endYear)
    obtain ⟨extra, optimized⟩ := solve.result [] startYear endYear (by omega) (by omega)
    exact optimized.scalar_returns.supported
  · intro startYear endYear _
    exact (data.«project-euler-statements».data.py.project_euler_statements.p19.naive_years.«Returns» startYear endYear).supported

end solvers.«19»
